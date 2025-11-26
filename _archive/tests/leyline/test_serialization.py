from __future__ import annotations

import os
import time
import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st

from esper.core.telemetry import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2


def _build_system_state() -> leyline_pb2.SystemStatePacket:
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=3,
        validation_accuracy=0.82,
        validation_loss=0.123,
        timestamp_ns=1_758_000_000_000_000_000,
        packet_id="run-epoch-3",
        global_step=1024,
        training_loss=0.456,
        source_subsystem="tolaria",
        training_run_id="run-demo",
        experiment_name="demo",
    )
    packet.training_metrics.update({"loss": 0.456, "grad_norm": 0.12})
    packet.hardware_context.device_type = "cuda"
    packet.hardware_context.device_id = "0"
    seed = packet.seed_states.add()
    seed.seed_id = "seed-1"
    seed.stage = leyline_pb2.SEED_STAGE_TRAINING
    seed.gradient_norm = 0.18
    seed.learning_rate = 0.002
    seed.metrics["loss"] = 0.321
    seed.age_epochs = 4
    seed.risk_score = 0.1
    return packet


def _build_adaptation_command() -> leyline_pb2.AdaptationCommand:
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-1",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
        execution_deadline_ms=250,
        issued_by="tamiyo",
    )
    command.issued_at.GetCurrentTime()
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "bp-demo"
    command.seed_operation.parameters["alpha"] = 0.5
    command.annotations["risk"] = "low"
    return command


def _build_field_report() -> leyline_pb2.FieldReport:
    report = leyline_pb2.FieldReport(
        version=1,
        report_id="fr-1",
        command_id="cmd-1",
        training_run_id="run-demo",
        seed_id="seed-1",
        blueprint_id="bp-demo",
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
        observation_window_epochs=1,
        tamiyo_policy_version="policy-stub",
        notes="synthetic",
    )
    report.issued_at.GetCurrentTime()
    report.metrics["loss_delta"] = -0.05
    mitigation = report.follow_up_actions.add()
    mitigation.action_type = "CONSERVATIVE_MODE"
    mitigation.rationale = "stability"
    return report


def _build_telemetry_packet() -> leyline_pb2.TelemetryPacket:
    metrics = [
        TelemetryMetric("tolaria.training.loss", 0.456, unit="loss"),
        TelemetryMetric("tolaria.training.latency_ms", 17.2, unit="ms"),
        TelemetryMetric("kasmina.isolation.violations", 0.0, unit="count"),
        TelemetryMetric("tamiyo.validation_loss", 0.321, unit="loss"),
    ]
    events = [
        TelemetryEvent(
            description="seed_operation",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            attributes={"seed_id": "seed-1"},
        )
    ]
    return build_telemetry_packet(
        packet_id="telemetry-sample",
        source="tolaria",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=metrics,
        events=events,
        health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
        health_summary="stable",
        health_indicators={"epoch": "3"},
    )


def _build_blueprint_descriptor() -> leyline_pb2.BlueprintDescriptor:
    descriptor = leyline_pb2.BlueprintDescriptor(
        blueprint_id="bp-42",
        name="UnitTest",
        tier=leyline_pb2.BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL,
        risk=0.75,
        stage=4,
        quarantine_only=False,
        approval_required=True,
        description="Test blueprint",
    )
    bounds = descriptor.allowed_parameters["alpha"]
    bounds.min_value = 0.1
    bounds.max_value = 0.9
    return descriptor


def _build_bus_envelope() -> leyline_pb2.BusEnvelope:
    descriptor = _build_blueprint_descriptor()
    payload = descriptor.SerializeToString()
    return leyline_pb2.BusEnvelope(
        message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_SYSTEM_STATE,
        payload=payload,
        attributes={"origin": "unit-test"},
    )


def test_system_state_roundtrip_size() -> None:
    packet = _build_system_state()
    payload = packet.SerializeToString()
    clone = leyline_pb2.SystemStatePacket()
    clone.ParseFromString(payload)
    assert clone == packet
    assert len(payload) < 280


def test_adaptation_command_roundtrip_size() -> None:
    command = _build_adaptation_command()
    payload = command.SerializeToString()
    clone = leyline_pb2.AdaptationCommand()
    clone.ParseFromString(payload)
    assert clone == command
    assert len(payload) < 280


def test_field_report_roundtrip() -> None:
    report = _build_field_report()
    payload = report.SerializeToString()
    clone = leyline_pb2.FieldReport()
    clone.ParseFromString(payload)
    assert clone == report


def test_telemetry_packet_roundtrip_size() -> None:
    packet = _build_telemetry_packet()
    payload = packet.SerializeToString()
    clone = leyline_pb2.TelemetryPacket()
    clone.ParseFromString(payload)
    assert clone == packet
    assert len(payload) < 280


def test_blueprint_descriptor_roundtrip() -> None:
    descriptor = _build_blueprint_descriptor()
    payload = descriptor.SerializeToString()
    clone = leyline_pb2.BlueprintDescriptor()
    clone.ParseFromString(payload)
    assert clone == descriptor


def test_bus_envelope_roundtrip() -> None:
    envelope = _build_bus_envelope()
    payload = envelope.SerializeToString()
    clone = leyline_pb2.BusEnvelope()
    clone.ParseFromString(payload)
    assert clone.message_type == envelope.message_type
    assert clone.attributes == envelope.attributes
    descriptor = leyline_pb2.BlueprintDescriptor()
    descriptor.ParseFromString(clone.payload)
    assert descriptor.blueprint_id == "bp-42"


_metric_names = (
    "tolaria.training.loss",
    "tolaria.training.accuracy",
    "tolaria.training.latency_ms",
    "kasmina.seeds.active",
    "kasmina.isolation.violations",
    "tamiyo.validation_loss",
    "tamiyo.loss_delta",
    "tamiyo.conservative_mode",
    "tamiyo.blueprint.risk",
)


@given(
    st.lists(
        st.builds(
            TelemetryMetric,
            name=st.sampled_from(_metric_names),
            value=st.floats(
                min_value=-10_000, max_value=10_000, allow_nan=False, allow_infinity=False
            ),
            unit=st.sampled_from(["loss", "ms", "count", "ratio", "bool", "delta"]),
        ),
        min_size=1,
        max_size=4,
    )
)
def test_telemetry_packet_property_roundtrip(metrics: list[TelemetryMetric]) -> None:
    packet_id = f"telemetry-prop-{uuid.uuid4()}"
    packet = build_telemetry_packet(
        packet_id=packet_id,
        source="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=metrics,
    )
    payload = packet.SerializeToString()
    clone = leyline_pb2.TelemetryPacket()
    clone.ParseFromString(payload)
    assert len(clone.metrics) == len(metrics)
    for proto_metric, metric in zip(clone.metrics, metrics, strict=True):
        assert proto_metric.name == metric.name
        assert proto_metric.value == pytest.approx(metric.value)
        assert proto_metric.unit == metric.unit


@pytest.mark.performance
def test_serialization_latency_budget() -> None:
    if not os.getenv("ESPER_RUN_PERF_TESTS"):
        pytest.skip("Set ESPER_RUN_PERF_TESTS=1 to enable serialization benchmarks")

    state = _build_system_state()
    command = _build_adaptation_command()
    report = _build_field_report()

    samples = 10_000
    start = time.perf_counter()
    for _ in range(samples):
        payload = state.SerializeToString()
        state.ParseFromString(payload)
    state_latency = (time.perf_counter() - start) / samples * 1e6

    start = time.perf_counter()
    for _ in range(samples):
        payload = command.SerializeToString()
        command.ParseFromString(payload)
    command_latency = (time.perf_counter() - start) / samples * 1e6

    start = time.perf_counter()
    for _ in range(samples):
        payload = report.SerializeToString()
        report.ParseFromString(payload)
    report_latency = (time.perf_counter() - start) / samples * 1e6

    assert state_latency < 80
    assert command_latency < 80
    assert report_latency < 120
