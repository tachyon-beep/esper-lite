"""Scripted policy smoke test for factored action wiring."""

from __future__ import annotations

from esper.leyline import (
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    TempoAction,
)
from esper.simic.training.oracle_sandbox import (
    OracleSchedule,
    OracleScheduleStep,
    build_oracle_ci_fixture,
    run_oracle_schedule,
)


def _scripted_action(
    op: LifecycleOp,
    *,
    blueprint: BlueprintAction = BlueprintAction.NOOP,
    alpha_target: AlphaTargetAction = AlphaTargetAction.FULL,
    alpha_speed: AlphaSpeedAction = AlphaSpeedAction.INSTANT,
    alpha_curve: AlphaCurveAction = AlphaCurveAction.LINEAR,
) -> FactoredAction:
    return FactoredAction(
        slot_idx=0,
        blueprint=blueprint,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=alpha_target,
        alpha_speed=alpha_speed,
        alpha_curve=alpha_curve,
        op=op,
    )


def test_scripted_policy_runner_smoke() -> None:
    """Scripted action sequence should execute through the oracle schedule runner."""
    fixture = build_oracle_ci_fixture()
    schedule = OracleSchedule(
        name="scripted-policy-runner-smoke",
        steps=(
            OracleScheduleStep(
                step_id="germinate_norm",
                action=_scripted_action(
                    LifecycleOp.GERMINATE,
                    blueprint=BlueprintAction.NORM,
                    alpha_speed=AlphaSpeedAction.MEDIUM,
                ),
            ),
            OracleScheduleStep(
                step_id="advance_to_training",
                action=_scripted_action(LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="advance_to_blending",
                action=_scripted_action(LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="set_alpha_full",
                action=_scripted_action(
                    LifecycleOp.SET_ALPHA_TARGET,
                    alpha_target=AlphaTargetAction.FULL,
                    alpha_speed=AlphaSpeedAction.INSTANT,
                    alpha_curve=AlphaCurveAction.COSINE,
                ),
            ),
            OracleScheduleStep(
                step_id="advance_to_holding",
                action=_scripted_action(LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="prune_holding",
                action=_scripted_action(LifecycleOp.PRUNE),
            ),
        ),
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
        "PRUNED",
    )
    assert all(step.mask_checked for step in result.steps)
    assert not fixture.model.has_active_seed_in_slot("r0c1")
