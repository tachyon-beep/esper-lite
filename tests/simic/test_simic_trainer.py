from esper.leyline import leyline_pb2
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
        policy=TamiyoPolicy(),
        buffer=buffer,
        config=SimicTrainerConfig(epochs=1, batch_size=2),
    )
    trainer.run_training()
