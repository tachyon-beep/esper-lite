import pytest
from fakeredis.aioredis import FakeRedis

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.simic import FieldReportReplayBuffer, SimicTrainer, SimicTrainerConfig
from esper.tamiyo import TamiyoPolicy


def _make_report(epoch: int) -> leyline_pb2.FieldReport:
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id=f"cmd-{epoch}",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    report = leyline_pb2.FieldReport(
        version=1,
        report_id=f"rpt-{epoch}",
        command_id=command.command_id,
        training_run_id="run",
        seed_id="seed-1",
        blueprint_id="bp-1",
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
    )
    report.metrics["loss"] = -0.01
    return report


def test_replay_buffer_stores_reports() -> None:
    buffer = FieldReportReplayBuffer(capacity=2)
    buffer.add(_make_report(0))
    buffer.add(_make_report(1))
    buffer.add(_make_report(2))
    assert len(list(buffer.sample(10))) == 2


def test_simic_trainer_runs() -> None:
    buffer = FieldReportReplayBuffer(capacity=4)
    for idx in range(4):
        buffer.add(_make_report(idx))
    trainer = SimicTrainer(
        policy=None,
        buffer=buffer,
        config=SimicTrainerConfig(epochs=1, batch_size=2),
    )
    trainer.run_training()


@pytest.mark.asyncio
async def test_field_report_buffer_ingest_from_oona() -> None:
    buffer = FieldReportReplayBuffer(capacity=4)
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="simic-buffer-test",
    )
    client = OonaClient("redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()
    report = _make_report(0)
    await client.publish_field_report(report)
    await client.ensure_consumer_group()
    await buffer.ingest_from_oona(client)
    reports = list(buffer.sample(1))
    assert len(reports) == 1
    await client.close()


def test_simic_create_policy_update() -> None:
    trainer = SimicTrainer(
        policy=None,
        buffer=FieldReportReplayBuffer(),
    )
    trainer.run_training()
    update = trainer.create_policy_update(
        policy_id="policy-1",
        training_run_id="run-1",
        policy_version="policy-v2",
    )
    assert update.tamiyo_policy_version == "policy-v2"
    assert update.payload
    assert trainer.policy_updates


@pytest.mark.asyncio
async def test_simic_publish_policy_updates() -> None:
    trainer = SimicTrainer(
        policy=None,
        buffer=FieldReportReplayBuffer(),
    )
    trainer.run_training()
    trainer.create_policy_update(
        policy_id="policy-1",
        training_run_id="run-1",
        policy_version="policy-v2",
    )
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="simic-test",
    )
    client = OonaClient("redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()
    await trainer.publish_policy_updates(client)
    assert await client.stream_length("oona.policy") == 1
    await client.close()
