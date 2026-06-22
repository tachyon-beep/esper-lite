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
