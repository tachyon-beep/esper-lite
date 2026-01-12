"""SET_ALPHA_TARGET operation handler.

Handles retargeting the alpha (blend weight) for active seeds. This
operation allows the policy to adjust how much a seed's output
contributes to the final prediction.

SET_ALPHA_TARGET conditions:
- Seed must exist in the slot
- Seed must be in HOLD alpha mode (not mid-transition)
- Seed must be in BLENDING or HOLDING stage

This operation schedules an alpha transition to a new target value
using the specified speed, curve, and algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

from esper.leyline import (
    ALPHA_SPEED_TO_STEPS,
    ALPHA_TARGET_VALUES,
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    SeedStage,
    STYLE_ALPHA_ALGORITHMS,
)
from esper.simic.training.handlers.base import HandlerContext, HandlerResult


# Stages where SET_ALPHA_TARGET is valid
ALPHA_TARGET_STAGES = frozenset(
    {
        SeedStage.BLENDING,
        SeedStage.HOLDING,
    }
)


@dataclass(slots=True)
class AlphaTargetParams:
    """Parameters for the SET_ALPHA_TARGET operation.

    Attributes:
        alpha_target_idx: Index into ALPHA_TARGET_VALUES for target alpha.
        alpha_speed_idx: Index into ALPHA_SPEED_TO_STEPS for transition speed.
        alpha_curve_idx: Index into AlphaCurveAction for transition curve.
        style_idx: Index into STYLE_ALPHA_ALGORITHMS for alpha algorithm.
    """

    alpha_target_idx: int
    alpha_speed_idx: int
    alpha_curve_idx: int
    style_idx: int


def can_set_alpha_target(ctx: HandlerContext) -> bool:
    """Check if SET_ALPHA_TARGET operation is allowed.

    Preconditions:
    - Seed must exist in the slot
    - Seed must be in HOLD alpha mode (not mid-transition)
    - Seed must be in BLENDING or HOLDING stage

    Args:
        ctx: Handler context with slot and seed state.

    Returns:
        True if alpha target can be set.
    """
    if ctx.seed_state is None:
        return False

    # Must be in HOLD mode (not mid-transition)
    if ctx.seed_state.alpha_controller.alpha_mode != AlphaMode.HOLD:
        return False

    # Must be in a valid stage
    if ctx.seed_state.stage not in ALPHA_TARGET_STAGES:
        return False

    return True


def execute_set_alpha_target(
    ctx: HandlerContext,
    params: AlphaTargetParams,
) -> HandlerResult:
    """Execute the SET_ALPHA_TARGET operation.

    Sets a new alpha target for the seed, initiating a transition
    from the current alpha to the target using the specified parameters.

    Args:
        ctx: Handler context with environment and model state.
        params: Alpha target parameters (target, speed, curve, algorithm).

    Returns:
        HandlerResult with success=True if alpha target was set.
    """
    # Verify preconditions
    if ctx.seed_state is None:
        return HandlerResult(
            success=False,
            error="Cannot set alpha target: no active seed in slot",
        )

    if ctx.seed_state.alpha_controller.alpha_mode != AlphaMode.HOLD:
        return HandlerResult(
            success=False,
            error="Cannot set alpha target: seed not in HOLD alpha mode",
        )

    if ctx.seed_state.stage not in ALPHA_TARGET_STAGES:
        return HandlerResult(
            success=False,
            error=f"Cannot set alpha target: seed in {ctx.seed_state.stage.name}, must be BLENDING or HOLDING",
        )

    # Resolve parameters
    alpha_target = ALPHA_TARGET_VALUES[params.alpha_target_idx]
    speed_steps = ALPHA_SPEED_TO_STEPS[AlphaSpeedAction(params.alpha_speed_idx)]
    curve_action = AlphaCurveAction(params.alpha_curve_idx)
    curve = curve_action.to_curve()
    alpha_algorithm = STYLE_ALPHA_ALGORITHMS[params.style_idx]

    # Record current alpha for telemetry
    pre_alpha = ctx.slot.alpha

    # Set the new alpha target
    success = ctx.slot.set_alpha_target(
        alpha_target=alpha_target,
        steps=speed_steps,
        curve=curve,
        alpha_algorithm=alpha_algorithm,
        initiator="policy",
    )

    if not success:
        return HandlerResult(
            success=False,
            error="set_alpha_target() returned False",
        )

    return HandlerResult(
        success=True,
        telemetry={
            "pre_alpha": pre_alpha,
            "alpha_target": alpha_target,
            "speed_steps": speed_steps,
            "curve": curve.name if curve else None,
            "alpha_algorithm": alpha_algorithm.name if alpha_algorithm else None,
        },
    )


__all__ = [
    "ALPHA_TARGET_STAGES",
    "AlphaTargetParams",
    "can_set_alpha_target",
    "execute_set_alpha_target",
]
