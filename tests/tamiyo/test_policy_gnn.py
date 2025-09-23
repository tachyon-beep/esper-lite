from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest
import torch

from esper.leyline import leyline_pb2
from esper.tamiyo import (
    TamiyoGraphBuilder,
    TamiyoGraphBuilderConfig,
    TamiyoPolicy,
    TamiyoPolicyConfig,
)


def _sample_packet() -> leyline_pb2.SystemStatePacket:
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        global_step=10,
        training_run_id="run-gnn",
        packet_id="pkt-gnn",
        validation_loss=0.42,
        training_loss=0.4,
    )
    packet.training_metrics["loss"] = 0.4
    packet.training_metrics["gradient_norm"] = 1.2
    packet.training_metrics["samples_per_s"] = 4800.0
    packet.training_metrics["hook_latency_ms"] = 11.0
    seed = packet.seed_states.add()
    seed.seed_id = "seed-1"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed.gradient_norm = 1.1
    seed.learning_rate = 0.01
    seed.layer_depth = 3
    seed.age_epochs = 12
    seed.risk_score = 0.3
    # Seed-level WP9 metrics (blend allowance overrides stage if present)
    seed.metrics["blend_allowed"] = 1.0
    return packet


def test_graph_builder_handles_missing_seeds() -> None:
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig())
    data = builder.build(leyline_pb2.SystemStatePacket(version=1))
    assert data["global"].x.shape[1] == 16
    assert data["seed"].x.shape[0] == 0
    assert data["blueprint"].x.shape[0] == 1
    coverage = getattr(data, "feature_coverage", {})
    assert isinstance(coverage, dict)
    assert "global.loss" in coverage


def test_graph_builder_schema(tmp_path: pytest.PathLike) -> None:
    blueprint_id = "BP001"
    metadata = {
        "tier": "BLUEPRINT_TIER_SAFE",
        "tier_index": 1,
        "risk": 0.45,
        "stage": 2,
        "quarantine_only": False,
        "approval_required": True,
        "parameter_count": 2,
        "allowed_parameters": {
            "alpha": {"min": 0.1, "max": 0.9, "span": 0.8},
            "beta": {"min": -0.5, "max": 0.5, "span": 1.0},
        },
    }
    metadata["graph"] = {
        "layers": [
            {
                "layer_id": f"{blueprint_id}-L0",
                "type": "linear",
                "depth": 0,
                "latency_ms": 5.0,
                "parameter_count": 2048,
                "dropout_rate": 0.1,
                "weight_norm": 1.2,
                "gradient_norm": 0.8,
                "activation": "relu",
            },
            {
                "layer_id": f"{blueprint_id}-L1",
                "type": "layer_norm",
                "depth": 1,
                "latency_ms": 4.0,
                "parameter_count": 1024,
                "dropout_rate": 0.0,
                "weight_norm": 0.9,
                "gradient_norm": 0.6,
                "activation": "gelu",
            },
        ],
        "activations": [
            {
                "activation_id": f"{blueprint_id}-A0",
                "type": "relu",
                "saturation_rate": 0.2,
                "gradient_flow": 0.9,
                "computational_cost": 256.0,
                "nonlinearity_strength": 0.6,
            },
            {
                "activation_id": f"{blueprint_id}-A1",
                "type": "gelu",
                "saturation_rate": 0.15,
                "gradient_flow": 0.85,
                "computational_cost": 192.0,
                "nonlinearity_strength": 0.5,
            },
        ],
        "parameters": [
            {"name": "alpha", "min": 0.1, "max": 0.9, "span": 0.8, "default": 0.5},
            {"name": "beta", "min": -0.5, "max": 0.5, "span": 1.0, "default": 0.0},
        ],
        "capabilities": {"allowed_blending_methods": ["linear", "cosine"]},
        "adjacency": {"layer": [[0, 1]]},
    }

    cfg = TamiyoGraphBuilderConfig(
        normalizer_path=tmp_path / "norms.json",
        max_layers=2,
        max_activations=2,
        max_parameters=2,
        layer_feature_dim=14,  # include extra categorical mask scalars
    activation_feature_dim=9,  # include activation type mask scalar
        parameter_feature_dim=10,
        edge_feature_dim=3,
        blueprint_metadata_provider=lambda bp: metadata if bp == blueprint_id else {},
    )
    builder = TamiyoGraphBuilder(cfg)

    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=3,
        global_step=7,
        packet_id=blueprint_id,
        training_run_id="run-graph",
    )
    seed_a = packet.seed_states.add()
    seed_a.seed_id = "seed-A"
    seed_a.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING
    seed_a.learning_rate = 0.02
    seed_a.risk_score = 0.2
    seed_a.layer_depth = 4
    seed_b = packet.seed_states.add()
    seed_b.seed_id = "seed-B"
    seed_b.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed_b.learning_rate = 0.01
    seed_b.risk_score = 0.6
    seed_b.layer_depth = 2

    graph = builder.build(packet)

    assert graph["seed"].x.shape[1] == 14
    assert graph["blueprint"].x.shape[1] == 14
    assert graph["layer"].x.shape == (2, cfg.layer_feature_dim)
    assert graph["activation"].x.shape == (2, cfg.activation_feature_dim)
    assert graph["parameter"].x.shape == (2, 10)
    assert set(graph["parameter"].node_ids) == {"run-graph-alpha", "run-graph-beta"}
    assert list(graph["layer"].node_ids) == [f"{blueprint_id}-L0", f"{blueprint_id}-L1"]
    assert list(graph["activation"].node_ids) == [f"{blueprint_id}-A0", f"{blueprint_id}-A1"]
    assert torch.isfinite(graph["layer"].x).all()
    assert torch.isfinite(graph["activation"].x).all()
    assert torch.isfinite(graph["seed"].candidate_scores).all()
    assert torch.isfinite(graph["blueprint"].candidate_scores).all()

    assert graph[("layer", "activates", "activation")].edge_attr.shape == (4, 3)
    assert graph[("layer", "feeds", "activation")].edge_attr.shape == (4, 3)
    assert graph[("seed", "allowed", "parameter")].edge_attr.shape == (4, 3)
    assert graph[("global", "annotates", "blueprint")].edge_attr.shape == (1, 3)
    assert graph[("layer", "connects", "layer")].edge_attr.shape[0] == 1
    assert graph[("seed", "monitors", "layer")].edge_attr.shape[1] == 3

    blueprint_vec = graph["blueprint"].x[0]
    assert pytest.approx(float(blueprint_vec[5]), rel=1e-6) == metadata["risk"]
    assert (tmp_path / "norms.json").exists()
    with (tmp_path / "norms.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert "loss" in data
    assert "layer_latency_ms" in data
    assert "parameter_default" in data
    assert "epoch_progress" in data
    coverage = getattr(graph, "feature_coverage", {})
    assert coverage
    assert coverage["seed.learning_rate"] <= 1.0
    # Edge coverage markers included
    assert "edges.layer_connects" in coverage
    assert "edges.seed_monitors" in coverage


def test_policy_select_action_populates_annotations() -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    command = policy.select_action(_sample_packet())
    assert command.annotations["policy_version"] == policy.architecture_version
    assert "blending_method" in command.annotations
    last_action = policy.last_action
    assert "value_estimate" in last_action
    assert "blending_method" in last_action
    assert "blending_schedule_start" in last_action
    assert "blending_schedule_end" in last_action
    assert "selected_seed_index" in last_action
    assert "selected_seed_score" in last_action
    assert "selected_blueprint_index" in last_action
    assert "policy_param_vector" in last_action
    assert len(last_action["policy_param_vector"]) == TamiyoPolicyConfig().param_vector_dim
    assert 0.0 <= float(last_action["blending_schedule_start"]) <= 1.0
    assert 0.0 <= float(last_action["blending_schedule_end"]) <= 1.0
    assert float(last_action["blending_schedule_start"]) <= float(
        last_action["blending_schedule_end"]
    )
    assert command.annotations["blending_schedule_units"] == "fraction_0_1"
    assert 0.0 <= float(command.annotations["blending_schedule_start"]) <= 1.0
    assert 0.0 <= float(command.annotations["blending_schedule_end"]) <= 1.0
    assert float(command.annotations["blending_schedule_start"]) <= float(
        command.annotations["blending_schedule_end"]
    )
    coverage = policy.feature_coverage
    assert coverage
    assert set(coverage) & {"global.loss", "seed.learning_rate"}
    if command.command_type == leyline_pb2.COMMAND_SEED:
        params = command.seed_operation.parameters
        method_list = TamiyoPolicyConfig().blending_methods
        expected_index = float(method_list.index(command.annotations["blending_method"]))
    assert params["blending_method_index"] == pytest.approx(expected_index)


def test_seed_masks_absent_fields() -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-masks")
    seed = packet.seed_states.add()
    seed.seed_id = "seed-mask"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    # Intentionally do not set learning_rate, risk_score, gradient_norm, age_epochs, layer_depth
    graph = policy._graph_builder.build(packet)  # type: ignore[attr-defined]
    assert graph["seed"].x.shape[1] == TamiyoGraphBuilderConfig().seed_feature_dim
    vec = graph["seed"].x[0]
    # Masks for absent fields (positions: lr=1, risk=3, grad=5, age=7, depth=11)
    assert float(vec[1]) == 0.0
    assert float(vec[3]) == 0.0
    assert float(vec[5]) == 0.0
    assert float(vec[7]) == 0.0
    assert float(vec[11]) == 0.0
    # Stage mask (position 9) should be present (1.0)
    assert float(vec[9]) == 1.0


def test_layer_connects_edges_from_adjacency() -> None:
    blueprint_id = "BP-ADJ"
    meta = {
        "graph": {
            "layers": [
                {"layer_id": f"{blueprint_id}-L0", "type": "linear", "depth": 0},
                {"layer_id": f"{blueprint_id}-L1", "type": "relu", "depth": 1},
                {"layer_id": f"{blueprint_id}-L2", "type": "relu", "depth": 2},
            ],
            "adjacency": {"layer": [[0, 1], [1, 2]]},
        }
    }
    builder = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            max_layers=3,
            blueprint_metadata_provider=lambda bp: meta if bp == blueprint_id else {},
        )
    )
    packet = leyline_pb2.SystemStatePacket(version=1, packet_id=blueprint_id, training_run_id="run-adj")
    graph = builder.build(packet)
    edge_index = graph[("layer", "connects", "layer")].edge_index
    assert edge_index.shape[1] == 2


def test_layer_activation_parameter_masks_when_missing_metadata() -> None:
    builder = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            max_layers=2,
            max_activations=2,
            max_parameters=1,
            layer_feature_dim=14,
            activation_feature_dim=9,
        )
    )
    packet = leyline_pb2.SystemStatePacket(version=1, training_run_id="run-missing-meta")
    # No graph metadata provided; builder should fallback to zeros/masks and not crash
    graph = builder.build(packet)
    # Layers present with default count; mask bits indicate missing
    layer_vec = graph["layer"].x[0]
    assert graph["layer"].x.shape == (2, builder._cfg.layer_feature_dim)  # type: ignore[attr-defined]
    assert float(layer_vec[1]) == 0.0  # depth presence mask
    assert float(layer_vec[3]) == 0.0  # latency presence mask
    assert float(layer_vec[5]) == 0.0  # param_count presence mask
    assert float(layer_vec[11]) == 0.0  # descriptor presence sentinel
    # Activations present with default count; mask indicates descriptor missing
    act_vec = graph["activation"].x[0]
    assert graph["activation"].x.shape == (2, builder._cfg.activation_feature_dim)  # type: ignore[attr-defined]
    assert float(act_vec[1]) == 0.0  # descriptor presence mask
    # Parameters fallback to allowed/default descriptor; masks indicate presence
    param_vec = graph["parameter"].x[0]
    assert graph["parameter"].x.shape == (1, builder._cfg.parameter_feature_dim)  # type: ignore[attr-defined]
    assert float(param_vec[1]) == 1.0  # min present
    assert float(param_vec[3]) == 1.0  # max present
    assert float(param_vec[5]) == 1.0  # span present
    assert float(param_vec[7]) == 1.0  # default present


def test_layer_mask_toggles_with_metadata() -> None:
    # With metadata present
    blueprint_id = "BP-META"
    meta_with = {
        "graph": {
            "layers": [
                {"layer_id": f"{blueprint_id}-L0", "depth": 0, "latency_ms": 4.0, "parameter_count": 16},
            ]
        }
    }
    builder = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            max_layers=1,
            layer_feature_dim=14,
            blueprint_metadata_provider=lambda bp: meta_with if bp == blueprint_id else {},
        )
    )
    packet = leyline_pb2.SystemStatePacket(version=1, packet_id=blueprint_id, training_run_id="run-mask")
    graph_with = builder.build(packet)
    vec_with = graph_with["layer"].x[0]
    assert float(vec_with[1]) == 1.0  # depth present mask
    assert float(vec_with[3]) == 1.0  # latency present mask
    # Categorical presence masks at extended indices
    assert float(vec_with[12]) in {0.0, 1.0}
    assert float(vec_with[13]) in {0.0, 1.0}

    # Without those fields
    meta_without = {"graph": {"layers": [{"layer_id": f"{blueprint_id}-L0"}]}}
    builder2 = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            max_layers=1,
            layer_feature_dim=14,
            blueprint_metadata_provider=lambda bp: meta_without if bp == blueprint_id else {},
        )
    )
    graph_without = builder2.build(packet)
    vec_without = graph_without["layer"].x[0]
    assert float(vec_without[1]) == 0.0  # depth absent mask
    assert float(vec_without[3]) == 0.0  # latency absent mask
    # Categorical presence masks unset when not provided
    assert float(vec_without[12]) == 0.0
    assert float(vec_without[13]) == 0.0


def test_activation_type_mask_scalar() -> None:
    blueprint_id = "BP-ACT"
    meta_with = {"graph": {"activations": [{"activation_id": f"{blueprint_id}-A0", "type": "relu"}]}}
    builder = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            max_activations=1,
            activation_feature_dim=9,
            blueprint_metadata_provider=lambda bp: meta_with if bp == blueprint_id else {},
        )
    )
    packet = leyline_pb2.SystemStatePacket(version=1, packet_id=blueprint_id, training_run_id="run-act")
    graph = builder.build(packet)
    vec = graph["activation"].x[0]
    assert float(vec[8]) == 1.0  # type present mask
    # Now without type
    meta_without = {"graph": {"activations": [{"activation_id": f"{blueprint_id}-A0"}]}}
    builder2 = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            max_activations=1,
            activation_feature_dim=9,
            blueprint_metadata_provider=lambda bp: meta_without if bp == blueprint_id else {},
        )
    )
    graph2 = builder2.build(packet)
    vec2 = graph2["activation"].x[0]
    assert float(vec2[8]) == 0.0


def test_policy_compile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_compile(*_args, **_kwargs):  # pragma: no cover - deliberately triggered
        raise RuntimeError("compile disabled for test")

    monkeypatch.setattr(torch, "compile", _raise_compile)
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=True))
    assert not policy.compile_enabled


def test_policy_seed_selection_uses_fallback_scores() -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    fallback = torch.tensor([0.2, 1.4, 0.1], dtype=torch.float32)
    seed_id, idx, score = policy._select_seed(
        None,
        fallback,
        ["seed-a", "seed-b", "seed-c"],
        [
            {"blend_allowed": 0.0, "risk": 0.5},
            {"blend_allowed": 1.0, "risk": 0.2},
            {"blend_allowed": 0.0, "risk": 0.3},
        ],
    )
    assert seed_id == "seed-b"
    assert idx == 1
    assert pytest.approx(score, rel=1e-6) == fallback[1].item()


def test_policy_pause_when_no_seed_candidates() -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    packet = leyline_pb2.SystemStatePacket(version=1, training_run_id="run-no-seed")
    command = policy.select_action(packet)
    assert command.command_type != leyline_pb2.COMMAND_SEED
    assert command.annotations.get("risk_reason", "") == "no_seed_candidates"


def test_policy_select_action_ranks_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))

    class _Stub(torch.nn.Module):  # pragma: no cover - deterministic stub
        def forward(self, _data):
            seed_scores = torch.tensor([0.1, 2.5, 0.4], dtype=torch.float32)
            blueprint_scores = torch.tensor([0.3], dtype=torch.float32)
            policy_logits = torch.full((1, 32), -4.0)
            policy_logits[0, 0] = 8.0
            policy_logits[0, 1] = -2.0
            return {
                "policy_logits": policy_logits,
                "policy_params": torch.tensor([[0.02, -0.01, 0.0, 0.1]]),
                "param_delta": torch.tensor([[0.02]]),
                "blending_logits": torch.tensor([[0.1, 0.3, 0.6]]),
                "risk_logits": torch.tensor([[0.2, 0.3, 0.4, 0.05, 0.05]]),
                "value": torch.tensor([0.7]),
                "schedule_params": torch.tensor([[1.5, -2.5]]),
                "seed_scores": seed_scores,
                "blueprint_scores": blueprint_scores,
                "breaker_logits": torch.tensor([[0.0, 0.0]]),
            }

    monkeypatch.setattr(policy, "_gnn", _Stub())
    policy._compiled_model = None  # ensure stub is used

    packet = leyline_pb2.SystemStatePacket(
        version=1, current_epoch=4, training_run_id="run-rank", packet_id="pkt-rank"
    )
    for idx in range(3):
        seed = packet.seed_states.add()
        seed.seed_id = f"seed-{idx}"
        seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING
        seed.learning_rate = 0.01 + idx * 0.001
        seed.layer_depth = 2 + idx
        seed.risk_score = 0.2 + 0.1 * idx

    command = policy.select_action(packet)
    assert command.command_type == leyline_pb2.COMMAND_SEED
    assert command.target_seed_id == "seed-1"
    params = command.seed_operation.parameters
    assert 0.0 <= params["blending_schedule_start"] <= 1.0
    assert 0.0 <= params["blending_schedule_end"] <= 1.0
    assert params["blending_schedule_start"] <= params["blending_schedule_end"]
    assert command.annotations["blending_schedule_units"] == "fraction_0_1"
    assert float(command.annotations["blending_schedule_start"]) <= float(
        command.annotations["blending_schedule_end"]
    )
    assert "policy_param_vector" in policy.last_action


def test_policy_validate_state_dict_rejects_legacy() -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    checkpoint = policy.state_dict()
    checkpoint["_metadata"]["architecture_version"] = "legacy"
    with pytest.raises(ValueError):
        policy.validate_state_dict(checkpoint)


def test_registry_round_trip(tmp_path: pytest.PathLike) -> None:
    cfg = TamiyoPolicyConfig(enable_compile=False, registry_path=Path(tmp_path))
    policy = TamiyoPolicy(cfg)
    first = policy._seed_registry.get("seed-x")
    second = policy._seed_registry.get("seed-x")
    assert first == second


def test_schedule_registry_stable_indices(tmp_path: pytest.PathLike) -> None:
    cfg = TamiyoPolicyConfig(enable_compile=False, registry_path=Path(tmp_path))
    policy_a = TamiyoPolicy(cfg)
    idx_a = policy_a._schedule_registry.get("linear")  # type: ignore[attr-defined]
    # New policy same registry path should yield same index
    policy_b = TamiyoPolicy(cfg)
    idx_b = policy_b._schedule_registry.get("linear")  # type: ignore[attr-defined]
    assert idx_a == idx_b


def test_seed_registry_stable_across_instances(tmp_path: pytest.PathLike) -> None:
    cfg = TamiyoPolicyConfig(enable_compile=False, registry_path=Path(tmp_path))
    policy_a = TamiyoPolicy(cfg)
    idx_a = policy_a._seed_registry.get("seed-stable")
    # Recreate policy with the same registry path to force reload
    policy_b = TamiyoPolicy(cfg)
    idx_b = policy_b._seed_registry.get("seed-stable")
    assert idx_a == idx_b
    # Validate persistence content
    reg_file = Path(tmp_path) / "seed_registry.json"
    assert reg_file.exists()
    contents = json.loads(reg_file.read_text(encoding="utf-8"))
    assert str(idx_a) == str(contents.get("seed-stable") or contents.get("seed-stable"))


@pytest.mark.perf
def test_policy_inference_perf_budget() -> None:
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    packet = _sample_packet()
    for idx in range(4):
        seed = packet.seed_states.add()
        seed.seed_id = f"seed-extra-{idx}"
        seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING
        seed.learning_rate = 0.01 + 0.001 * idx
        seed.layer_depth = 3 + idx
        seed.risk_score = 0.25

    # warm-up
    policy.select_action(packet)

    durations: list[float] = []
    runs = 100
    for _ in range(runs):
        start = time.perf_counter()
        policy.select_action(packet)
        durations.append((time.perf_counter() - start) * 1000.0)

    p95 = statistics.quantiles(durations, n=100)[94]
    assert p95 <= 45.0
