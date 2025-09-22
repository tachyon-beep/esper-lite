from __future__ import annotations

import json

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
    return packet


def test_graph_builder_handles_missing_seeds() -> None:
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig())
    data = builder.build(leyline_pb2.SystemStatePacket(version=1))
    assert data["global"].x.shape[1] == 16
    assert data["seed"].x.shape[0] == 1  # placeholder when no seeds present
    assert data["blueprint"].x.shape[0] == 1


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

    builder = TamiyoGraphBuilder(
        TamiyoGraphBuilderConfig(
            normalizer_path=tmp_path / "norms.json",
            max_layers=2,
            max_activations=2,
            max_parameters=2,
            layer_feature_dim=6,
            activation_feature_dim=4,
            parameter_feature_dim=4,
            edge_feature_dim=3,
            blueprint_metadata_provider=lambda bp: metadata if bp == blueprint_id else {},
        )
    )

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

    assert graph["layer"].x.shape == (2, 6)
    assert graph["activation"].x.shape == (2, 4)
    assert graph["parameter"].x.shape == (2, 4)
    assert set(graph["parameter"].node_ids) == {"run-graph-alpha", "run-graph-beta"}

    assert graph[("layer", "activates", "activation")].edge_attr.shape == (4, 3)
    assert graph[("seed", "allowed", "parameter")].edge_attr.shape == (4, 3)
    assert graph[("global", "annotates", "blueprint")].edge_attr.shape == (1, 3)

    blueprint_vec = graph["blueprint"].x[0]
    assert pytest.approx(float(blueprint_vec[4]), rel=1e-6) == metadata["risk"]
    assert (tmp_path / "norms.json").exists()
    with (tmp_path / "norms.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert "loss" in data

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
    if command.command_type == leyline_pb2.COMMAND_SEED:
        params = command.seed_operation.parameters
        method_list = TamiyoPolicyConfig().blending_methods
        expected_index = float(method_list.index(command.annotations["blending_method"]))
        assert params["blending_method_index"] == pytest.approx(expected_index)


def test_policy_compile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_compile(*_args, **_kwargs):  # pragma: no cover - deliberately triggered
        raise RuntimeError("compile disabled for test")

    monkeypatch.setattr(torch, "compile", _raise_compile)
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=True))
    assert not policy.compile_enabled
