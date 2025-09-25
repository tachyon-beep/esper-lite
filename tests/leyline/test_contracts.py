from datetime import UTC, datetime
from pathlib import Path

from google.protobuf import struct_pb2

from esper.leyline import (
    DEFAULT_BUNDLE_NAME,
    DEFAULT_BUNDLE_VERSION,
    ContractRegistry,
    leyline_pb2,
    register_default_bundle,
)


def test_register_default_bundle_creates_entry() -> None:
    registry = ContractRegistry()
    register_default_bundle(registry)
    bundle = registry.get(DEFAULT_BUNDLE_NAME)
    assert bundle is not None
    assert bundle.version == DEFAULT_BUNDLE_VERSION
    assert bundle.schema_dir.exists()


def test_system_state_packet_serialization_roundtrip() -> None:
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        validation_accuracy=0.84,
        validation_loss=0.42,
        training_loss=0.43,
        training_run_id="run-1",
        experiment_name="exp-42",
        source_subsystem="tolaria",
        packet_id="pkt-1",
        global_step=1200,
    )
    packet.timestamp_ns = int(datetime.now(tz=UTC).timestamp() * 1e9)
    packet.training_metrics["loss"] = 0.43
    seed_state = packet.seed_states.add()
    seed_state.seed_id = "seed-1"
    seed_state.stage = leyline_pb2.SEED_STAGE_TRAINING
    seed_state.gradient_norm = 1.2
    seed_state.learning_rate = 3e-4
    seed_state.layer_depth = 12
    seed_state.metrics["sparsity"] = 0.1
    seed_state.age_epochs = 2
    seed_state.risk_score = 0.02
    packet.hardware_context.device_type = "cuda"
    packet.hardware_context.device_id = "0"
    packet.hardware_context.total_memory_gb = 24.0
    packet.hardware_context.available_memory_gb = 12.5
    packet.hardware_context.temperature_celsius = 65.0
    packet.hardware_context.utilization_percent = 80.0
    packet.hardware_context.compute_capability = 90

    payload = struct_pb2.Struct()
    payload.update({"seed_id": "seed-1"})
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-1",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
        execution_deadline_ms=18,
        issued_by="tamiyo",
    )
    command.issued_at.FromDatetime(datetime.now(tz=UTC))
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "blueprint-alpha"
    command.seed_operation.parameters["alpha"] = 0.5
    command.annotations["risk_mode"] = "normal"

    report = leyline_pb2.FieldReport(
        version=1,
        report_id="rpt-1",
        command_id=command.command_id,
        training_run_id="run-1",
        seed_id="seed-1",
        blueprint_id="blueprint-alpha",
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
        observation_window_epochs=1,
        tamiyo_policy_version="policy-v1",
        notes="Nominal",
    )
    report.metrics["loss_delta"] = -0.02
    mitigation = report.follow_up_actions.add()
    mitigation.action_type = "NONE"
    mitigation.rationale = "All good"
    report.issued_at.FromDatetime(datetime.now(tz=UTC))

    blob = report.SerializeToString()
    assert len(blob) < 512
    restored = leyline_pb2.FieldReport()
    restored.ParseFromString(blob)
    assert restored.report_id == "rpt-1"
    assert restored.outcome == leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS

    update = leyline_pb2.PolicyUpdate(
        version=1,
        policy_id="policy-1",
        training_run_id="run-1",
        tamiyo_policy_version="policy-v2",
    )
    update_blob = update.SerializeToString()
    restored_update = leyline_pb2.PolicyUpdate()
    restored_update.ParseFromString(update_blob)
    assert restored_update.policy_id == "policy-1"


def test_contract_bundle_path_relative() -> None:
    registry = register_default_bundle()
    bundle = registry.get(DEFAULT_BUNDLE_NAME)
    assert bundle is not None
    assert bundle.schema_dir.is_dir()
    expected = Path(__file__).resolve().parents[2] / "src" / "esper" / "leyline" / "_generated"
    assert bundle.schema_dir == expected
