from __future__ import annotations

import pytest
import torch

from esper.leyline import (
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    SlotConfig,
    TempoAction,
)
from esper.simic.training.oracle_sandbox import (
    ORACLE_CI_FIXTURE_ID,
    ORACLE_CI_SLOT_ID,
    OracleSchedule,
    OracleScheduleStep,
    build_default_oracle_schedule,
    build_oracle_ci_fixture,
    run_oracle_schedule,
)


def _wait_action(*, slot_idx: int = 0) -> FactoredAction:
    return FactoredAction(
        slot_idx=slot_idx,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.INSTANT,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.WAIT,
    )


def test_default_oracle_schedule_is_stable_and_covers_lifecycle_contract() -> None:
    schedule = build_default_oracle_schedule()
    rebuilt = build_default_oracle_schedule()

    assert schedule.name == "oracle-ci-lifecycle-v1"
    assert schedule.proof_profile == "oracle-sandbox"
    assert schedule.fixture_id == ORACLE_CI_FIXTURE_ID
    assert schedule.identity == rebuilt.identity
    assert schedule.identity.startswith("oracle-sandbox:oracle-ci-lifecycle-v1:")

    ops = tuple(step.action.op for step in schedule.steps)
    assert ops == (
        LifecycleOp.GERMINATE,
        LifecycleOp.ADVANCE,
        LifecycleOp.ADVANCE,
        LifecycleOp.SET_ALPHA_TARGET,
        LifecycleOp.ADVANCE,
        LifecycleOp.FOSSILIZE,
        LifecycleOp.PRUNE,
    )
    assert schedule.steps[-1].expect_success is False

    manifest = schedule.to_manifest()
    assert manifest["name"] == schedule.name
    assert manifest["fixture_id"] == ORACLE_CI_FIXTURE_ID
    assert manifest["steps"][0]["op"] == "GERMINATE"
    assert manifest["steps"][-1]["expect_success"] is False


def test_oracle_schedule_validation_fails_loudly_for_malformed_entries() -> None:
    valid_step = OracleScheduleStep(step_id="wait", action=_wait_action())

    with pytest.raises(ValueError, match="at least one step"):
        OracleSchedule(name="empty", steps=())

    with pytest.raises(ValueError, match="duplicate step_id"):
        OracleSchedule(name="duplicate", steps=(valid_step, valid_step))

    with pytest.raises(ValueError, match="step_id must be non-empty"):
        OracleScheduleStep(step_id="", action=_wait_action())

    with pytest.raises(ValueError, match="slot_idx must be non-negative"):
        OracleScheduleStep(step_id="negative-slot", action=_wait_action(slot_idx=-1))


def test_oracle_ci_fixture_is_cpu_first_and_deterministic() -> None:
    fixture = build_oracle_ci_fixture()
    rebuilt = build_oracle_ci_fixture()

    assert fixture.fixture_id == ORACLE_CI_FIXTURE_ID
    assert fixture.slot_config == SlotConfig(slot_ids=(ORACLE_CI_SLOT_ID,))
    assert fixture.enabled_slots == (ORACLE_CI_SLOT_ID,)
    assert fixture.inputs.device.type == "cpu"
    assert fixture.targets.device.type == "cpu"
    assert fixture.inputs.shape == (2, 3, 8, 8)
    assert fixture.targets.equal(torch.tensor([0, 1]))
    assert fixture.model.training is True
    assert torch.equal(fixture.inputs, rebuilt.inputs)
    assert fixture.model.total_seeds() == 0


def test_oracle_schedule_runner_uses_masks_governor_and_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_oracle_ci_fixture()
    schedule = build_default_oracle_schedule()
    handler_calls: list[LifecycleOp] = []

    from esper.simic.training.handlers import get_handler as real_get_handler

    def recording_get_handler(op_idx: int):
        handler = real_get_handler(op_idx)

        def wrapped_handler(*args, **kwargs):
            handler_calls.append(LifecycleOp(op_idx))
            return handler(*args, **kwargs)

        return wrapped_handler

    monkeypatch.setattr(
        "esper.simic.training.oracle_sandbox.get_handler",
        recording_get_handler,
    )

    result = run_oracle_schedule(schedule=schedule, fixture=fixture)

    assert result.success is True
    assert tuple(step.op for step in result.steps) == tuple(
        step.action.op for step in schedule.steps
    )
    assert tuple(step.stage_after for step in result.steps) == (
        "GERMINATED",
        "TRAINING",
        "BLENDING",
        "BLENDING",
        "HOLDING",
        "FOSSILIZED",
        "FOSSILIZED",
    )
    assert handler_calls == [
        LifecycleOp.GERMINATE,
        LifecycleOp.ADVANCE,
        LifecycleOp.ADVANCE,
        LifecycleOp.SET_ALPHA_TARGET,
        LifecycleOp.ADVANCE,
        LifecycleOp.FOSSILIZE,
    ]
    assert all(step.mask_checked for step in result.steps)
    assert all(step.governor_checked for step in result.steps[:-1])
    assert result.steps[-1].blocked_by == "mask"
    assert result.steps[-1].handler_called is False
    assert result.steps[-1].success is False


def test_oracle_schedule_runner_blocks_governor_veto_before_handler() -> None:
    fixture = build_oracle_ci_fixture()
    germinate_step = build_default_oracle_schedule().steps[0]
    schedule = OracleSchedule(
        name="oracle-governor-veto",
        steps=(germinate_step,),
    )

    result = run_oracle_schedule(
        schedule=schedule,
        fixture=fixture,
        val_loss=float("nan"),
    )

    assert result.success is False
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.mask_checked is True
    assert step.governor_checked is True
    assert step.governor_approved is False
    assert step.blocked_by == "governor"
    assert step.handler_called is False
    assert fixture.model.total_seeds() == 0
