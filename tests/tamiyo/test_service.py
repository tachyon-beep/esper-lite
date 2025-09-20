import pytest

from esper.leyline import leyline_pb2
from esper.tamiyo import RiskConfig, TamiyoPolicy, TamiyoService


def test_tamiyo_service_generates_command() -> None:
    service = TamiyoService(policy=TamiyoPolicy())
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-1",
        packet_id="pkt-1",
    )
    command = service.evaluate_epoch(packet)
    assert command.command_type == leyline_pb2.COMMAND_SEED
    assert command.target_seed_id == "seed-1"
    assert service.telemetry_packets


def test_conservative_mode_overrides_directive() -> None:
    service = TamiyoService(
        policy=TamiyoPolicy(),
        risk_config=RiskConfig(conservative_mode=True),
    )
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1)
    command = service.evaluate_epoch(packet)
    assert command.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    assert any(event.description == "Tamiyo pause triggered" for event in telemetry.events)


def test_field_report_generation() -> None:
    service = TamiyoService()
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
