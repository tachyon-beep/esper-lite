import pytest

from esper.core import (
    AdaptationCommand,
    AdaptationDirective,
    FieldReport,
    FieldReportOutcome,
    SystemStatePacket,
    TrainingPhase,
)


def test_system_state_packet_defaults() -> None:
    packet = SystemStatePacket(run_id="r1", epoch_index=0, phase=TrainingPhase.INIT)
    assert packet.host_metrics == {}
    assert packet.seed_snapshots == {}


def test_adaptation_command_roundtrip() -> None:
    command = AdaptationCommand(
        run_id="r1",
        epoch_index=1,
        directive=AdaptationDirective.NO_OP,
    )
    serialized = command.model_dump()
    restored = AdaptationCommand.model_validate(serialized)
    assert restored == command


def test_field_report_outcome_enum() -> None:
    report = FieldReport(
        run_id="r1",
        command=AdaptationCommand(run_id="r1", epoch_index=2, directive=AdaptationDirective.NO_OP),
        outcome=FieldReportOutcome.SUCCESS,
        metrics_delta={"loss": -0.1},
    )
    assert report.outcome is FieldReportOutcome.SUCCESS


@pytest.mark.parametrize("directive", list(AdaptationDirective))
def test_adaptation_directive_members(directive: AdaptationDirective) -> None:
    assert directive.value
