from io import BytesIO

import pytest
import torch
from fakeredis.aioredis import FakeRedis

from esper.leyline import leyline_pb2
from esper.urza import UrzaLibrary
from esper.karn import BlueprintMetadata, BlueprintTier
from esper.oona import OonaClient, StreamConfig
from esper.tamiyo import (
    FieldReportStoreConfig,
    RiskConfig,
    TamiyoPolicy,
    TamiyoService,
)


def test_tamiyo_service_generates_command(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(policy=TamiyoPolicy(), store_config=config)
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
    }
    assert "policy_action" in command.annotations
    assert "policy_param_delta" in command.annotations
    assert service.telemetry_packets


def test_conservative_mode_overrides_directive(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        risk_config=RiskConfig(conservative_mode=True),
        store_config=config,
    )
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1)
    command = service.evaluate_epoch(packet)
    assert command.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    assert any(event.description == "pause_triggered" for event in telemetry.events)


def test_field_report_generation(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=config)
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


@pytest.mark.asyncio
async def test_tamiyo_publish_history_to_oona(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=config)
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
async def test_tamiyo_consume_policy_updates(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    service = TamiyoService(store_config=config)
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


def test_tamiyo_annotations_include_blueprint_metadata(tmp_path) -> None:
    config = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    metadata = BlueprintMetadata(
        blueprint_id="bp-demo",
        name="Demo",
        tier=BlueprintTier.EXPERIMENTAL,
        description="Test blueprint",
        allowed_parameters={},
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

    service = TamiyoService(policy=_PolicyStub(), store_config=config, urza=urza)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1)
    command = service.evaluate_epoch(packet)
    assert command.annotations["blueprint_tier"] == "experimental"
    assert command.annotations["blueprint_stage"] == "5"
    assert command.annotations["blueprint_risk"] == "0.85"
    assert command.command_type == leyline_pb2.COMMAND_PAUSE
