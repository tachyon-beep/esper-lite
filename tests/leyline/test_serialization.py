from __future__ import annotations

import os
import time

import pytest

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
