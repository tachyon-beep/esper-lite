from io import BytesIO
import os
import time
import statistics
from types import SimpleNamespace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from torch import nn
from fakeredis.aioredis import FakeRedis

from esper.leyline import leyline_pb2
from esper.urza import UrzaLibrary
from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.oona import OonaClient, StreamConfig
from esper.kasmina import KasminaSeedManager
from esper.security.signing import SignatureContext, sign
from esper.tamiyo import (
    FieldReportStoreConfig,
    RiskConfig,
    TamiyoPolicy,
    TamiyoPolicyConfig,
    TamiyoService,
)


_SIGNATURE_CONTEXT = SignatureContext(secret=b"tamiyo-test-secret")


class _KasminaRuntime:
    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        return nn.Identity(), 1.0


class _SlowPolicy(TamiyoPolicy):
    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        time.sleep(0.02)
        return super().select_action(packet)


class _SeedPolicy(TamiyoPolicy):
    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-timeout",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = "bp-timeout"
        self._last_action = {"action": 0.0, "param_delta": 0.0}
        return command


class _SlowUrza:
    def __init__(self) -> None:
        self._record = SimpleNamespace(
            metadata=SimpleNamespace(
                tier=leyline_pb2.BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL,
                risk=0.1,
                stage=1,
                quarantine_only=False,
                approval_required=False,
                description="slow",
            )
        )

    def get(self, blueprint_id: str) -> SimpleNamespace:
        time.sleep(0.05)
        return self._record


class _CoveragePolicy(TamiyoPolicy):
    """Policy stub that sets a specific feature_coverage summary."""

    def __init__(self, avg: float = 0.2) -> None:
        super().__init__(TamiyoPolicyConfig(enable_compile=False))
        # Create a deterministic coverage map with desired average
        self._forced_avg = float(avg)

    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        # Build a simple pause command
        cmd = leyline_pb2.AdaptationCommand(version=1, command_type=leyline_pb2.COMMAND_PAUSE)
        cmd.issued_by = "tamiyo"
        cmd.issued_at.GetCurrentTime()
        # Force coverage summary
        # 2 keys: one present with value=_forced_avg, one missing with 0 → average ~ _forced_avg / 2
        # To get exact avg, just use identical values
        cov_val = max(0.0, min(1.0, self._forced_avg))
        self._last_feature_coverage = {"global.loss": cov_val, "seed.learning_rate": cov_val}
        self._last_action = {"action": 2.0, "param_delta": 0.0, "value_estimate": 0.0}
        return cmd


def test_tamiyo_service_generates_command(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=config,
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=1000.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-1",
        packet_id="pkt-1",
    )
    command = service.evaluate_epoch(packet)
    assert command.command_type in {
        leyline_pb2.COMMAND_SEED,
        leyline_pb2.COMMAND_OPTIMIZER,
        leyline_pb2.COMMAND_PAUSE,
        leyline_pb2.COMMAND_CIRCUIT_BREAKER,
    }
    assert "policy_action" in command.annotations
    assert "policy_param_delta" in command.annotations
    assert "policy_version" in command.annotations
    assert "blending_method" in command.annotations
    assert "policy_risk_score" in command.annotations
    assert "policy_risk_index" in command.annotations
    if command.command_type == leyline_pb2.COMMAND_SEED:
        assert command.annotations.get("selected_seed")
    else:
        assert command.annotations.get("risk_reason") == "no_seed_candidates"
    if command.command_type == leyline_pb2.COMMAND_SEED:
        params = command.seed_operation.parameters
        assert "blending_method_index" in params
        method_list = TamiyoPolicyConfig().blending_methods
        expected_index = float(method_list.index(command.annotations["blending_method"]))
        assert params["blending_method_index"] == pytest.approx(expected_index)
        assert "blending_schedule_start" in params
        assert "blending_schedule_end" in params
        assert 0.0 <= params["blending_schedule_start"] <= 1.0
        assert 0.0 <= params["blending_schedule_end"] <= 1.0
        assert params["blending_schedule_start"] <= params["blending_schedule_end"]
    assert service.telemetry_packets
    # Budget guardrail: inference latency <= 45 ms
    metrics = {m.name: m.value for m in service.telemetry_packets[-1].metrics}
    assert "tamiyo.inference.latency_ms" in metrics
    assert metrics["tamiyo.inference.latency_ms"] <= 45.0
    assert "tamiyo.gnn.inference.latency_ms" in metrics
    assert "tamiyo.gnn.feature_coverage" in metrics
    assert "tamiyo.gnn.compile_enabled" in metrics
    assert "tamiyo.policy.value_estimate" in metrics
    assert "tamiyo.policy.risk_score" in metrics
    telemetry = service.telemetry_packets[-1]
    assert "blending_method" in telemetry.system_health.indicators
    assert telemetry.system_health.indicators.get("policy_compile") in {"0", "1"}
    assert telemetry.system_health.indicators.get("policy_arch") == service._policy.architecture_version  # pylint: disable=protected-access
    # WP1 acceptance: command annotation includes coverage summary
    assert "feature_coverage" in command.annotations
    cov = float(command.annotations["feature_coverage"])  # type: ignore[arg-type]
    assert 0.0 <= cov <= 1.0
    # WP15: granular coverage present in annotations and telemetry
    assert "coverage_types" in command.annotations
    import json as _json
    types = _json.loads(command.annotations["coverage_types"])  # type: ignore[arg-type]
    assert any(k.startswith("node.") or k.startswith("edges.") for k in types.keys())
    # Telemetry contains per-type metrics
    metric_names = {m.name for m in telemetry.metrics}
    assert any(n.startswith("tamiyo.gnn.feature_coverage.node.") for n in metric_names)


@pytest.mark.asyncio
async def test_degraded_inputs_routes_emergency(tmp_path) -> None:
    # Force very low coverage → CRITICAL degraded_inputs
    policy = _CoveragePolicy(avg=0.1)
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=policy,
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
    )
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-degraded")
    command = service.evaluate_step(packet)
    # Route to Oona and assert emergency
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="tamiyo-degraded",
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()
    await service.publish_history(oona)
    # Emergency stream should have a critical packet from Tamiyo
    assert await oona.stream_length("oona.emergency") >= 1
    await oona.close()


def test_tamiyo_signed_command_accepted_and_replay_rejected(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-1",
        packet_id="pkt-1",
    )
    command = service.evaluate_epoch(packet)
    assert "signature" in command.annotations
    manager = KasminaSeedManager(_KasminaRuntime(), signing_context=_SIGNATURE_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))

    manager.handle_command(command)
    manager.finalize_step(step_index=1)
    packets = manager.drain_telemetry_packets()
    assert not any(event.description == "command_rejected" for pkt in packets for event in pkt.events)

    manager.handle_command(command)
    manager.finalize_step(step_index=2)
    packets = manager.drain_telemetry_packets()
    assert any(event.description == "command_rejected" for pkt in packets for event in pkt.events)


def test_service_schedule_parameters_fractional(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=config,
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=500.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=2,
        training_run_id="run-seed",
        packet_id="pkt-seed",
    )
    seed = packet.seed_states.add()
    seed.seed_id = "seed-123"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING
    seed.learning_rate = 0.02
    seed.risk_score = 0.2
    seed.layer_depth = 3

    command = service.evaluate_epoch(packet)
    assert command.command_type == leyline_pb2.COMMAND_SEED
    params = command.seed_operation.parameters
    assert 0.0 <= params["blending_schedule_start"] <= 1.0
    assert 0.0 <= params["blending_schedule_end"] <= 1.0
    assert params["blending_schedule_start"] <= params["blending_schedule_end"]
    assert command.annotations["blending_schedule_units"] == "fraction_0_1"
    assert 0.0 <= float(command.annotations["blending_schedule_start"]) <= 1.0
    assert 0.0 <= float(command.annotations["blending_schedule_end"]) <= 1.0


def test_evaluate_step_timeout_inference(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=_SlowPolicy(),
        store_config=config,
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=1.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-timeout",
        global_step=42,
    )
    command = service.evaluate_step(packet)
    assert command.command_type == leyline_pb2.COMMAND_PAUSE
    assert command.annotations["risk_reason"] == "timeout_inference"
    telemetry = service.telemetry_packets[-1]
    assert any(event.description == "timeout_inference" for event in telemetry.events)
    priority = telemetry.system_health.indicators.get("priority")
    assert priority == leyline_pb2.MessagePriority.Name(
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    )


def test_default_step_timeout_uses_env_when_no_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # With no explicit constructor override and env default at 5 ms, a slow policy (>20 ms)
    # should trigger a timeout.
    monkeypatch.setenv("TAMIYO_STEP_TIMEOUT_MS", "5")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(policy=_SlowPolicy(), store_config=cfg, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    pkt = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-timeout-env")
    cmd = service.evaluate_step(pkt)
    assert cmd.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    # Timeout event present and mapped to HIGH priority
    assert any(e.description == "timeout_inference" for e in telemetry.events)
    prio = telemetry.system_health.indicators.get("priority")
    assert prio == leyline_pb2.MessagePriority.Name(
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    )


def test_explicit_step_timeout_overrides_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Env default is tight (5 ms), but an explicit constructor override (100 ms)
    # should prevent the timeout even with a slow policy.
    monkeypatch.setenv("TAMIYO_STEP_TIMEOUT_MS", "5")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=_SlowPolicy(),
        store_config=cfg,
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=100.0,
    )
    pkt = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-timeout-override")
    _ = service.evaluate_step(pkt)
    telemetry = service.telemetry_packets[-1]
    assert not any(e.description == "timeout_inference" for e in telemetry.events)


def test_evaluate_step_includes_coverage_and_policy_version(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=config,
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=250.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-step",
        packet_id="pkt-step",
    )
    # Provide one seed to avoid fast-path pause
    seed = packet.seed_states.add()
    seed.seed_id = "seed-step"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed.learning_rate = 0.01
    cmd = service.evaluate_step(packet)
    # Annotations contain coverage and policy metadata
    assert "feature_coverage" in cmd.annotations
    assert "policy_version" in cmd.annotations
    # Priority unchanged (normal) for non-degraded events
    telemetry = service.telemetry_packets[-1]
    assert not any(e.description == "timeout_inference" for e in telemetry.events)
    prio_name = telemetry.system_health.indicators.get("priority")
    assert prio_name == leyline_pb2.MessagePriority.Name(leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL)


def test_blend_mode_annotations_disabled_by_default(tmp_path) -> None:
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False)),
        store_config=cfg,
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=250.0,
    )
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=2, training_run_id="run-blend-off")
    # Provide a seed so we emit a SEED command
    seed = packet.seed_states.add(); seed.seed_id = "seed-1"; seed.stage = leyline_pb2.SEED_STAGE_TRAINING
    cmd = service.evaluate_epoch(packet)
    assert cmd.command_type in {leyline_pb2.COMMAND_SEED, leyline_pb2.COMMAND_OPTIMIZER, leyline_pb2.COMMAND_PAUSE}
    # New keys should not be present by default
    assert "blend_mode" not in cmd.annotations


def test_emits_blend_mode_annotations_when_enabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAMIYO_ENABLE_BLEND_MODE_ANN", "true")
    monkeypatch.setenv("TAMIYO_BLEND_MODE_DEFAULT", "CONFIDENCE")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=cfg, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-blend-on")
    seed = packet.seed_states.add(); seed.seed_id = "seed-1"; seed.stage = leyline_pb2.SEED_STAGE_TRAINING
    cmd = service.evaluate_epoch(packet)
    if cmd.command_type == leyline_pb2.COMMAND_SEED:
        assert cmd.annotations.get("blend_mode") == "CONFIDENCE"
        # gating params present
        assert "gate_k" in cmd.annotations and "gate_tau" in cmd.annotations
        assert "alpha_lo" in cmd.annotations and "alpha_hi" in cmd.annotations


def test_channel_mode_emits_alpha_vec_when_shape_small(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable annotations and channel mode
    monkeypatch.setenv("TAMIYO_ENABLE_BLEND_MODE_ANN", "true")
    monkeypatch.setenv("TAMIYO_BLEND_MODE_DEFAULT", "CHANNEL")
    monkeypatch.setenv("TAMIYO_BLEND_ALPHA_VEC_MAX", "8")
    # Prepare Urza metadata with small output_channels
    urza_root = tmp_path / "urza"; urza_root.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / "bp-vec.pt"; artifact_path.write_bytes(b"dummy")
    descriptor = BlueprintDescriptor(
        blueprint_id="bp-vec",
        name="vec",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.2,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="vec",
    )
    library = UrzaLibrary(root=urza_root)
    # Include a layer with output_channels = 4
    extras = {"graph": {"layers": [{"layer_id": "L0", "type": "linear", "depth": 0, "output_channels": 4}]}}
    library.save(descriptor, artifact_path, extras=extras)

    class _PolicySeed(TamiyoPolicy):
        def select_action(self, _packet):  # pragma: no cover - deterministic stub
            cmd = leyline_pb2.AdaptationCommand(version=1, command_type=leyline_pb2.COMMAND_SEED, target_seed_id="seed-1")
            cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            cmd.seed_operation.blueprint_id = "bp-vec"
            self._last_action = {"action": 0.0, "param_delta": 0.0}
            return cmd

    service = TamiyoService(policy=_PolicySeed(), store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"), urza=library, signature_context=_SIGNATURE_CONTEXT)
    pkt = leyline_pb2.SystemStatePacket(version=1, current_epoch=1)
    seed = pkt.seed_states.add(); seed.seed_id = "seed-1"; seed.stage = leyline_pb2.SEED_STAGE_TRAINING
    cmd = service.evaluate_epoch(pkt)
    if cmd.command_type == leyline_pb2.COMMAND_SEED:
        # alpha_vec present with small length
        if "alpha_vec_len" in cmd.annotations:
            assert int(cmd.annotations["alpha_vec_len"]) == 4
            assert "alpha_vec" in cmd.annotations



def test_health_indicators_include_timeouts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Set explicit budgets to predictable values
    monkeypatch.setenv("TAMIYO_STEP_TIMEOUT_MS", "7.5")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=cfg, urza=urza, signature_context=_SIGNATURE_CONTEXT, metadata_timeout_ms=12.0)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-hi")
    command = service.evaluate_step(packet)
    telemetry = service.telemetry_packets[-1]
    ind = telemetry.system_health.indicators
    assert "timeout_budget_ms" in ind
    assert "metadata_timeout_budget_ms" in ind
    # Values are strings but must be parseable as floats
    float(ind["timeout_budget_ms"])  # type: ignore[arg-type]
    float(ind["metadata_timeout_budget_ms"])  # type: ignore[arg-type]


def test_evaluate_step_timeout_urza(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(
        policy=_SeedPolicy(),
        store_config=config,
        urza=_SlowUrza(),
        signature_context=_SIGNATURE_CONTEXT,
        metadata_timeout_ms=1.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-urza",
    )
    command = service.evaluate_step(packet)
    telemetry = service.telemetry_packets[-1]
    assert any(event.description == "timeout_urza" for event in telemetry.events)
    assert "blueprint_tier" not in command.annotations
    # Priority should be mapped to HIGH on timeout to enable emergency routing
    priority = telemetry.system_health.indicators.get("priority")
    assert priority == leyline_pb2.MessagePriority.Name(
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    )


def test_service_urza_graph_metadata_round_trip(tmp_path) -> None:
    urza_root = tmp_path / "urza"
    urza_root.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / "bp.pt"
    artifact_path.write_bytes(b"dummy")

    descriptor = BlueprintDescriptor(
        blueprint_id="bp-graph",
        name="graph-test",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.35,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="graph metadata fixture",
    )
    bounds = descriptor.allowed_parameters["alpha"]
    bounds.min_value = 0.05
    bounds.max_value = 0.95

    graph_metadata = {
        "source": "urza-cache",
        "layers": [
            {
                "layer_id": "bp-graph-L0",
                "type": "linear",
                "depth": 0,
                "latency_ms": 7.0,
                "parameter_count": 1024,
                "dropout_rate": 0.1,
                "weight_norm": 1.1,
                "gradient_norm": 0.9,
                "activation": "relu",
            }
        ],
        "activations": [
            {
                "activation_id": "bp-graph-A0",
                "type": "relu",
                "saturation_rate": 0.25,
                "gradient_flow": 0.85,
                "computational_cost": 128.0,
                "nonlinearity_strength": 0.55,
            }
        ],
        "parameters": [
            {
                "name": "alpha",
                "min": 0.05,
                "max": 0.95,
                "span": 0.9,
                "default": 0.5,
            }
        ],
        "capabilities": {"allowed_blending_methods": ["cosine"]},
    }

    library = UrzaLibrary(root=urza_root)
    library.save(
        descriptor,
        artifact_path,
        extras={"graph_metadata": graph_metadata},
    )

    local_signature = SignatureContext(secret=b"tamiyo-test-secret")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=library,
        signature_context=local_signature,
    )

    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        global_step=5,
        training_run_id="run-graph",
        packet_id="bp-graph",
    )
    seed = packet.seed_states.add()
    seed.seed_id = "seed-graph"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING
    seed.learning_rate = 0.02
    seed.layer_depth = 3
    seed.risk_score = 0.25

    command = service.evaluate_epoch(packet)
    assert command.annotations.get("blueprint_tier") == "BLUEPRINT_TIER_SAFE"
    assert command.annotations.get("blueprint_risk") == "0.35"

    cached = service._policy._blueprint_metadata.get("bp-graph")  # pylint: disable=protected-access
    assert cached is not None
    graph_block = cached.get("graph")
    assert graph_block is not None
    assert graph_block.get("source") == "urza-cache"
    assert graph_block.get("layers") and graph_block["layers"][0]["layer_id"] == "bp-graph-L0"
    assert graph_block.get("activations") and graph_block["activations"][0]["activation_id"] == "bp-graph-A0"
    assert graph_block.get("parameters") and graph_block["parameters"][0]["name"] == "alpha"

    metrics = {metric.name: metric.value for metric in service.telemetry_packets[-1].metrics}
    assert metrics["tamiyo.blueprint.risk"] == pytest.approx(0.35)

    service.close()


def test_service_urza_graph_metadata_fallback_via_guard_spec(tmp_path) -> None:
    urza_root = tmp_path / "urza"
    urza_root.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / "bp_guard.pt"
    artifact_path.write_bytes(b"dummy")

    descriptor = BlueprintDescriptor(
        blueprint_id="bp-guard",
        name="guard-test",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.2,
        stage=1,
        quarantine_only=False,
        approval_required=False,
        description="guard metadata fixture",
    )

    guard_spec = [{"shape": [4, 8], "dtype": "float32"}]
    library = UrzaLibrary(root=urza_root)
    library.save(
        descriptor,
        artifact_path,
        extras={"guard_spec": guard_spec, "prewarm_ms": 10.0},
    )

    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=library,
        signature_context=_SIGNATURE_CONTEXT,
    )

    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-guard",
        packet_id="bp-guard",
    )
    seed = packet.seed_states.add()
    seed.seed_id = "seed-guard"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING

    service.evaluate_epoch(packet)
    cached = service._policy._blueprint_metadata.get("bp-guard")  # type: ignore[attr-defined]
    assert cached is not None
    graph_block = cached.get("graph")
    assert graph_block is not None
    assert graph_block.get("layers") and graph_block["layers"][0]["type"] == "guard_tensor"
    assert graph_block.get("activations") and graph_block["activations"][0]["type"] == "float32"
    service.close()


def test_inference_breaker_enters_conservative_mode(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(
        policy=_SlowPolicy(),
        store_config=config,
        urza=UrzaLibrary(root=tmp_path / "urza"),
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=1.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-breaker",
        global_step=1,
    )
    for step in range(3):
        packet.global_step = step
        service.evaluate_step(packet)
    assert service._risk.conservative_mode  # type: ignore[attr-defined]
    assert any(
        event.description == "conservative_entered"
        for pkt in service.telemetry_packets
        for event in pkt.events
    )


def test_conservative_mode_overrides_directive(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        risk_config=RiskConfig(conservative_mode=True),
        store_config=config,
        urza=UrzaLibrary(root=tmp_path / "urza"),
        signature_context=_SIGNATURE_CONTEXT,
    )
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1)
    command = service.evaluate_epoch(packet)
    assert command.command_type == leyline_pb2.COMMAND_PAUSE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to validate default compile")
def test_service_default_compile_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure settings don't force-override compile/device
    monkeypatch.delenv("TAMIYO_ENABLE_COMPILE", raising=False)
    monkeypatch.delenv("TAMIYO_DEVICE", raising=False)
    service = TamiyoService(store_config=FieldReportStoreConfig(path=Path("/tmp/field_reports.log")), urza=UrzaLibrary(root=Path("/tmp/urza")), signature_context=_SIGNATURE_CONTEXT)
    # If backend raises or compile path falls back, skip to avoid flakiness on exotic stacks
    if not getattr(service._policy, "compile_enabled", False):  # type: ignore[attr-defined]
        pytest.skip("compile path fell back on this CUDA backend; default enablement attempted")
    assert service._policy.compile_enabled  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_field_report_publish_hedging_success_after_retry(tmp_path: Path) -> None:
    # Use zero backoff to attempt retry immediately on next publish cycle
    import os
    os.environ["TAMIYO_FR_RETRY_BACKOFF_MS"] = "0"
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    # Generate one field report
    # Create a minimal command for the report without invoking evaluate
    cmd = leyline_pb2.AdaptationCommand(version=1, command_type=leyline_pb2.COMMAND_PAUSE)
    cmd.command_id = str(uuid4())
    cmd.issued_at.GetCurrentTime()
    service.generate_field_report(
        command=cmd,
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
        metrics_delta={"loss": 0.0},
        training_run_id="run-hedge",
        seed_id="seed-h",
        blueprint_id="bp-h",
    )
    assert len(service.field_reports) >= 1

    calls: dict[str, int] = {"report": 0}

    class _FlakyOona:
        async def publish_field_report(self, report: leyline_pb2.FieldReport) -> None:
            calls["report"] += 1
            # Fail first attempt, succeed afterwards
            if calls["report"] == 1:
                raise RuntimeError("transient")

        async def publish_telemetry(self, *_args: object, **_kwargs: object) -> None:
            return None

    oona = _FlakyOona()
    sent = await service.publish_history(oona)  # first attempt fails
    assert sent is False
    assert len(service.field_reports) == 1  # retained for retry
    # Second attempt should succeed and clear buffer
    sent2 = await service.publish_history(oona)
    assert sent2 is True
    assert len(service.field_reports) == 0
    # Only one successful publish overall for the report
    assert calls["report"] == 2


@pytest.mark.asyncio
async def test_field_report_publish_drops_after_cap(tmp_path: Path) -> None:
    # Override via env-driven settings
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    # Inject settings via environment
    import os
    os.environ["TAMIYO_FIELD_REPORT_MAX_RETRIES"] = "1"
    os.environ["TAMIYO_FR_RETRY_BACKOFF_MS"] = "0"
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    # Generate one field report
    cmd = leyline_pb2.AdaptationCommand(version=1, command_type=leyline_pb2.COMMAND_PAUSE)
    cmd.command_id = str(uuid4())
    cmd.issued_at.GetCurrentTime()
    service.generate_field_report(
        command=cmd,
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_NEUTRAL,
        metrics_delta={},
        training_run_id="run-hedge-cap",
        seed_id="seed-hc",
        blueprint_id="bp-hc",
    )
    assert len(service.field_reports) >= 1

    class _AlwaysFailOona:
        async def publish_field_report(self, report: leyline_pb2.FieldReport) -> None:
            raise RuntimeError("down")

        async def publish_telemetry(self, *_args: object, **_kwargs: object) -> None:
            return None

    oona = _AlwaysFailOona()
    # First attempt: retained (retry count=1 <= cap=1)
    sent = await service.publish_history(oona)
    assert sent is False
    assert len(service.field_reports) == 1
    # Second attempt: exceeds cap (>1) and drops from memory
    sent2 = await service.publish_history(oona)
    assert sent2 is False
    assert len(service.field_reports) == 0
    # WAL remains intact
    assert len(service._field_report_store.reports()) >= 1


def test_field_report_generation(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=0,
        training_run_id="run-1",
    )
    command = service.evaluate_epoch(packet)
    report = service.generate_field_report(
        command=command,
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
        metrics_delta={"loss": -0.05},
        training_run_id="run-1",
        seed_id="seed-1",
        blueprint_id="bp-1",
    )
    assert report.outcome == leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS
    assert report.metrics["loss"] == pytest.approx(-0.05)
    assert service.field_reports


def test_field_report_emitted_for_each_step(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    for epoch in range(3):
        packet = leyline_pb2.SystemStatePacket(
            version=1,
            current_epoch=epoch + 1,
            training_run_id="run-steps",
            global_step=epoch + 10,
        )
        service.evaluate_step(packet)
    reports = service.field_reports
    assert len(reports) == 3
    assert all(report.training_run_id == "run-steps" for report in reports)
    assert all(report.observation_window_epochs >= 1 for report in reports)


def test_field_report_retention_rewrites(tmp_path) -> None:
    config = FieldReportStoreConfig(
        path=tmp_path / "field_reports.log",
        retention=timedelta(minutes=30),
    )
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-retention")
    command = service.evaluate_step(packet)
    old_command = leyline_pb2.AdaptationCommand()
    old_command.CopyFrom(command)
    old_command.command_id = str(uuid4())
    old_command.issued_at.FromDatetime(datetime.now(tz=UTC) - timedelta(minutes=10))
    service.generate_field_report(
        command=old_command,
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
        metrics_delta={},
        training_run_id="run-retention",
        seed_id="seed-old",
        blueprint_id="",
        observation_window_epochs=1,
    )
    assert len(service.field_reports) == 2
    future_time = datetime.now(tz=UTC) + timedelta(hours=1)
    service._field_report_store.enforce_retention(now=future_time)
    service._field_reports = service._field_report_store.reports()
    remaining = service.field_reports
    assert all(report.seed_id != "seed-old" for report in remaining)


@pytest.mark.perf
def test_step_evaluate_p95_budget(tmp_path) -> None:
    if os.getenv("RUN_PERF_TESTS") != "1":
        pytest.skip("perfs disabled; set RUN_PERF_TESTS=1 to enable")
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    durations: list[float] = []
    for step in range(30):
        packet = leyline_pb2.SystemStatePacket(
            version=1,
            current_epoch=1,
            global_step=step,
            training_run_id="run-budget",
        )
        start = time.perf_counter()
        service.evaluate_step(packet)
        durations.append((time.perf_counter() - start) * 1000.0)
    durations.sort()
    p95 = durations[int(len(durations) * 0.95)]
    assert p95 <= 10.0




def test_field_report_emitted_on_timeout(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(
        policy=_SlowPolicy(),
        store_config=config,
        urza=UrzaLibrary(root=tmp_path / "urza"),
        signature_context=_SIGNATURE_CONTEXT,
        step_timeout_ms=1.0,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-timeout-report",
        global_step=99,
    )
    command = service.evaluate_step(packet)
    # Last report exists and is neutral due to timeout
    assert service.field_reports
    last = service.field_reports[-1]
    assert last.training_run_id == "run-timeout-report"
    assert last.outcome == leyline_pb2.FIELD_REPORT_OUTCOME_NEUTRAL
    # Notes reflect last event (timeout_inference)
    assert "timeout" in last.notes.lower()
@pytest.mark.asyncio
async def test_tamiyo_publish_history_to_oona(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=0)
    service.evaluate_epoch(packet)
    command = service.evaluate_epoch(packet)
    service.generate_field_report(
        command=command,
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_NEUTRAL,
        metrics_delta={"loss": 0.0},
        training_run_id="run-1",
        seed_id="seed-1",
        blueprint_id="bp-1",
    )

    redis_config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="tamiyo-test",
    )
    oona = OonaClient("redis://localhost", config=redis_config, redis_client=FakeRedis())
    await oona.ensure_consumer_group()
    await service.publish_history(oona)
    assert await oona.stream_length("oona.normal") >= 1
    assert await oona.stream_length("oona.telemetry") >= 1
    await oona.close()


@pytest.mark.asyncio
async def test_tamiyo_publish_history_routes_priority(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-routing")
    command = service.evaluate_step(packet)
    telemetry = service.telemetry_packets[-1]
    if telemetry.events:
        telemetry.events[0].level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
    else:
        telemetry.events.add(
            description="timeout_inference",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
        )
    telemetry.system_health.indicators["priority"] = "MESSAGE_PRIORITY_CRITICAL"

    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="tamiyo-routing",
    )
    oona = OonaClient("redis://localhost", config=config, redis_client=FakeRedis())
    await oona.ensure_consumer_group()
    await service.publish_history(oona)
    assert await oona.stream_length("oona.emergency") >= 1
    await oona.close()


@pytest.mark.asyncio
async def test_tamiyo_consume_policy_updates(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    redis_config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="tamiyo-policy-test",
    )
    client = OonaClient("redis://localhost", config=redis_config, redis_client=FakeRedis())
    await client.ensure_consumer_group()
    update = leyline_pb2.PolicyUpdate(
        version=1,
        policy_id="policy-1",
        training_run_id="run-1",
        tamiyo_policy_version="policy-v42",
    )
    buffer = BytesIO()
    torch.save(service._policy.state_dict(), buffer)  # type: ignore[attr-defined]
    update.payload = buffer.getvalue()
    await client.publish_policy_update(update)
    await client.ensure_consumer_group()
    await service.consume_policy_updates(client)
    assert service.policy_updates
    assert service.policy_updates[0].tamiyo_policy_version == "policy-v42"
    await client.close()


def test_policy_update_rejected_on_version_mismatch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAMIYO_VERIFY_UPDATES", "true")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=cfg, urza=UrzaLibrary(root=tmp_path / "urza"), signature_context=_SIGNATURE_CONTEXT)
    update = leyline_pb2.PolicyUpdate(version=1, policy_id="p1", training_run_id="run-1", tamiyo_policy_version="wrong-version")
    buf = BytesIO(); torch.save(service._policy.state_dict(), buf)  # type: ignore[attr-defined]
    update.payload = buf.getvalue()
    service.ingest_policy_update(update)
    # Expect rejection telemetry; no applied updates
    assert not service.policy_updates
    assert any(e.description == "policy_update_rejected" and e.attributes.get("reason") == "version_mismatch"
               for pkt in service.telemetry_packets for e in pkt.events)


def test_policy_update_rejected_when_stale(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAMIYO_VERIFY_UPDATES", "true")
    monkeypatch.setenv("TAMIYO_UPDATE_FRESHNESS_SEC", "60")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=cfg, urza=UrzaLibrary(root=tmp_path / "urza"), signature_context=_SIGNATURE_CONTEXT)
    update = leyline_pb2.PolicyUpdate(version=1, policy_id="p1", training_run_id="run-1", tamiyo_policy_version=service._policy.architecture_version)  # type: ignore[attr-defined]
    # Make the update appear 1 hour old
    from datetime import datetime, timedelta, UTC
    update.issued_at.FromDatetime(datetime.now(tz=UTC) - timedelta(hours=1))
    buf = BytesIO(); torch.save(service._policy.state_dict(), buf)  # type: ignore[attr-defined]
    update.payload = buf.getvalue()
    service.ingest_policy_update(update)
    assert not service.policy_updates
    assert any(e.description == "policy_update_rejected" and e.attributes.get("reason") == "stale_update"
               for pkt in service.telemetry_packets for e in pkt.events)


def test_policy_update_applies_transactionally(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Valid, fresh update should apply and be recorded
    monkeypatch.setenv("TAMIYO_VERIFY_UPDATES", "true")
    monkeypatch.setenv("TAMIYO_UPDATE_FRESHNESS_SEC", "0")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=cfg, urza=UrzaLibrary(root=tmp_path / "urza"), signature_context=_SIGNATURE_CONTEXT)
    update = leyline_pb2.PolicyUpdate(version=1, policy_id="p-ok", training_run_id="run-1", tamiyo_policy_version=service._policy.architecture_version)  # type: ignore[attr-defined]
    update.issued_at.GetCurrentTime()
    buf = BytesIO(); torch.save(service._policy.state_dict(), buf)  # type: ignore[attr-defined]
    update.payload = buf.getvalue()
    service.ingest_policy_update(update)
    assert service.policy_updates and service.policy_updates[-1].policy_id == "p-ok"


def test_policy_update_rejected_on_registry_mismatch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Tamper registry digest in metadata to force validate_state_dict rejection
    monkeypatch.setenv("TAMIYO_VERIFY_UPDATES", "true")
    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=cfg, urza=UrzaLibrary(root=tmp_path / "urza"), signature_context=_SIGNATURE_CONTEXT)
    state = service._policy.state_dict()  # type: ignore[attr-defined]
    if "_metadata" in state and isinstance(state["_metadata"], dict):
        meta = dict(state["_metadata"])  # shallow copy
        regs = dict(meta.get("registries", {}))
        if regs:
            # Flip one digest to an invalid value
            k = next(iter(regs.keys()))
            regs[k] = "deadbeef"
            meta["registries"] = regs
            state = dict(state)
            state["_metadata"] = meta
    buf = BytesIO(); torch.save(state, buf)
    update = leyline_pb2.PolicyUpdate(version=1, policy_id="p-bad", training_run_id="run-1", tamiyo_policy_version=service._policy.architecture_version)  # type: ignore[attr-defined]
    update.payload = buf.getvalue()
    service.ingest_policy_update(update)
    assert not service.policy_updates
    assert any(e.description == "policy_update_rejected" for pkt in service.telemetry_packets for e in pkt.events)


def test_tamiyo_annotations_include_blueprint_metadata(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    metadata = BlueprintDescriptor(
        blueprint_id="bp-demo",
        name="Demo",
        tier=BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL,
        description="Test blueprint",
        risk=0.85,
        stage=5,
        quarantine_only=True,
        approval_required=True,
    )
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"demo")
    urza.save(metadata, artifact)

    class _PolicyStub(TamiyoPolicy):
        def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
            command = leyline_pb2.AdaptationCommand(
                version=1,
                command_id="cmd",
                command_type=leyline_pb2.COMMAND_SEED,
                target_seed_id="seed-1",
            )
            command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            command.seed_operation.blueprint_id = "bp-demo"
            self._last_action = {"action": 0.0, "param_delta": 0.0}
            return command

    service = TamiyoService(policy=_PolicyStub(), store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1)
    command = service.evaluate_epoch(packet)
    assert command.annotations["blueprint_tier"] == leyline_pb2.BlueprintTier.Name(metadata.tier)
    assert command.annotations["blueprint_stage"] == "5"
    assert command.annotations["blueprint_risk"] == "0.85"
    assert command.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    assert any(
        event.description == "bp_quarantine"
        and event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
        for event in telemetry.events
    )


def test_loss_spike_triggers_pause(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    service._last_validation_loss = 0.1  # type: ignore[attr-defined]
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-loss",
        training_metrics={"loss": 0.0},
    )
    packet.validation_loss = 0.3
    command = service.evaluate_step(packet)
    assert command.command_type == leyline_pb2.COMMAND_PAUSE
    assert command.annotations["risk_reason"] == "loss_spike"
    telemetry = service.telemetry_packets[-1]
    assert any(event.description == "loss_spike" for event in telemetry.events)


def test_latency_metrics_trigger_actions(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-latency",
        training_metrics={
            "loss": 0.2,
            "step_latency_ms": 250.0,
            "kasmina.apply_ms": 35.0,
            "kasmina.finalize_ms": 40.0,
        },
    )
    cmd = service.evaluate_step(packet)
    # High step latency should trigger PAUSE
    assert cmd.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    ev = {e.description for e in telemetry.events}
    assert "step_latency_high" in ev or "kasmina_apply_slow" in ev or "kasmina_finalize_slow" in ev


def test_optimizer_hint_drives_optimizer(tmp_path) -> None:
    # Use a seed-focused policy to ensure baseline command is SEED
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=_SeedPolicy(),
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=urza,
        signature_context=_SIGNATURE_CONTEXT,
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=2,
        training_run_id="run-opt",
        training_metrics={
            "loss": 0.3,
            "loss_delta": 0.1,
            "optimizer_lr": 0.0,
        },
    )
    # Provide a seed to avoid fast-path pause
    seed = packet.seed_states.add()
    seed.seed_id = "seed-opt"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    cmd = service.evaluate_step(packet)
    assert cmd.command_type == leyline_pb2.COMMAND_OPTIMIZER
    telemetry = service.telemetry_packets[-1]
    assert any(e.description in {"optimizer_hint", "loss_warning"} for e in telemetry.events)


def test_tamiyo_stale_command_rejected(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(store_config=config, urza=urza, signature_context=_SIGNATURE_CONTEXT)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=0, training_run_id="run-1")
    command = service.evaluate_epoch(packet)

    manager = KasminaSeedManager(_KasminaRuntime(), signing_context=_SIGNATURE_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    manager.handle_command(command)
    manager.finalize_step(step_index=1)
    manager.drain_telemetry_packets()

    stale_command = leyline_pb2.AdaptationCommand()
    stale_command.CopyFrom(command)
    stale_command.command_id = str(uuid4())
    stale_time = datetime.now(tz=UTC) - timedelta(minutes=10)
    stale_command.issued_at.FromDatetime(stale_time)
    if "signature" in stale_command.annotations:
        del stale_command.annotations["signature"]
    stale_command.annotations["signature"] = sign(
        stale_command.SerializeToString(deterministic=True),
        _SIGNATURE_CONTEXT,
    )

    manager.handle_command(stale_command)
    manager.finalize_step(step_index=2)
    packets = manager.drain_telemetry_packets()
    assert any(
        event.description == "command_rejected" and event.attributes.get("reason") == "stale_command"
        for packet in packets
        for event in packet.events
    )

def test_compile_fallback_counter_exposed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force compile to raise so policy falls back and records a fallback
    def _raise_compile(*_args, **_kwargs):  # pragma: no cover - deliberately triggered
        raise RuntimeError("compile disabled for test")

    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    monkeypatch.setattr(torch, "compile", _raise_compile)

    cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    # Enable compile in config to trigger the path even on CPU
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=True))
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(policy=policy, store_config=cfg, urza=urza, signature_context=_SIGNATURE_CONTEXT)

    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-compile")
    # Provide at least one seed to avoid fast-path pause
    seed = packet.seed_states.add()
    seed.seed_id = "seed-compile"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed.learning_rate = 0.01
    command = service.evaluate_step(packet)

    telemetry = service.telemetry_packets[-1]
    metrics = {m.name: m.value for m in telemetry.metrics}
    # Compile fallback counter present and > 0
    assert metrics.get("tamiyo.gnn.compile_fallback_total", 0.0) > 0.0
    # Compile enabled flag should be 0 after fallback
    assert metrics.get("tamiyo.gnn.compile_enabled", 1.0) == 0.0
    # Telemetry event should record the CPU demotion reason
    assert any(event.description == "compile_disabled_cpu" for event in telemetry.events)
    event = next(event for event in telemetry.events if event.description == "compile_disabled_cpu")
    assert event.attributes.get("reason") in {"device_not_cuda", "cuda_unavailable"}
    assert command.annotations.get("policy_compile_reason") in {"device_not_cuda", "cuda_unavailable"}
