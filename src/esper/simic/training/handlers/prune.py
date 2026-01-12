"""PRUNE operation handler.

Handles the removal of underperforming or unwanted seeds. Pruning
schedules a gradual alpha decay to 0, after which the seed is removed.

Prune conditions:
- Seed must be in GERMINATED, TRAINING, BLENDING, or HOLDING stage
- Seed must be in HOLD alpha mode (not mid-transition)
- Seed must meet MIN_PRUNE_AGE requirement (at least 1 epoch)

Pruning uses the policy-specified speed, curve, and algorithm to
control the decay trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from esper.leyline import (
    ALPHA_SPEED_TO_STEPS,
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    MIN_PRUNE_AGE,
    SeedStage,
)
from esper.simic.training.handlers.base import HandlerContext, HandlerResult

if TYPE_CHECKING:
    from esper.simic.rewards import SeedInfo


# Stages from which pruning is allowed
PRUNABLE_STAGES = frozenset(
    {
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.HOLDING,
    }
)


@dataclass(slots=True)
class PruneParams:
    """Parameters for the PRUNE operation.

    Attributes:
        alpha_speed_idx: Index into ALPHA_SPEED_TO_STEPS for decay speed.
        alpha_curve_idx: Index into AlphaCurveAction for decay curve shape.
    """

    alpha_speed_idx: int
    alpha_curve_idx: int


def can_prune(ctx: HandlerContext, seed_info: "SeedInfo | None") -> bool:
    """Check if PRUNE operation is allowed in the given context.

    Preconditions:
    - Slot must have an active seed
    - Seed must be in a prunable stage
    - Seed must be in HOLD alpha mode (not mid-transition)
    - Seed must meet MIN_PRUNE_AGE (BUG-020 fix)

    Args:
        ctx: Handler context with slot and seed state.
        seed_info: SeedInfo for age checking.

    Returns:
        True if prune can proceed.
    """
    if ctx.seed_state is None:
        return False

    if ctx.seed_state.stage not in PRUNABLE_STAGES:
        return False

    # Must be in HOLD mode (not mid-transition)
    if ctx.seed_state.alpha_controller.alpha_mode != AlphaMode.HOLD:
        return False

    # Must be able to transition to PRUNED
    if not ctx.seed_state.can_transition_to(SeedStage.PRUNED):
        return False

    # BUG-020 fix: enforce MIN_PRUNE_AGE to match masking invariant
    if seed_info is None:
        return False

    if seed_info.seed_age_epochs < MIN_PRUNE_AGE:
        return False

    return True


def execute_prune(
    ctx: HandlerContext,
    params: PruneParams,
    seed_info: "SeedInfo | None",
) -> HandlerResult:
    """Execute the PRUNE operation.

    Schedules an alpha decay to 0 using the specified speed and curve.
    The actual removal happens when alpha reaches 0 during step_epoch().

    Args:
        ctx: Handler context with environment and model state.
        params: Prune parameters (speed, curve).
        seed_info: SeedInfo for validation.

    Returns:
        HandlerResult with success=True if prune was scheduled.
    """
    # Verify preconditions
    if not can_prune(ctx, seed_info):
        if ctx.seed_state is None:
            return HandlerResult(
                success=False,
                error="Cannot prune: no active seed in slot",
            )
        if ctx.seed_state.stage not in PRUNABLE_STAGES:
            return HandlerResult(
                success=False,
                error=f"Cannot prune: seed in {ctx.seed_state.stage.name} is not prunable",
            )
        if ctx.seed_state.alpha_controller.alpha_mode != AlphaMode.HOLD:
            return HandlerResult(
                success=False,
                error="Cannot prune: seed not in HOLD alpha mode",
            )
        if seed_info is None or seed_info.seed_age_epochs < MIN_PRUNE_AGE:
            return HandlerResult(
                success=False,
                error=f"Cannot prune: seed must be at least {MIN_PRUNE_AGE} epoch(s) old",
            )
        return HandlerResult(
            success=False,
            error="Cannot prune: unknown precondition failure",
        )

    # Resolve prune parameters
    speed_steps = ALPHA_SPEED_TO_STEPS[AlphaSpeedAction(params.alpha_speed_idx)]
    curve_action = AlphaCurveAction(params.alpha_curve_idx)
    curve = curve_action.to_curve()

    # Schedule the prune (alpha decay to 0)
    success = ctx.slot.schedule_prune(
        steps=speed_steps,
        curve=curve,
        initiator="policy",
    )

    if not success:
        return HandlerResult(
            success=False,
            error="schedule_prune() returned False",
        )

    # Update episode counter
    ctx.env_state.prune_count += 1

    return HandlerResult(
        success=True,
        telemetry={
            "speed_steps": speed_steps,
            "curve": curve.name if curve else None,
            "pre_stage": ctx.seed_state.stage.name if ctx.seed_state else None,
            "seed_age_epochs": seed_info.seed_age_epochs if seed_info else 0,
        },
    )


__all__ = [
    "PRUNABLE_STAGES",
    "PruneParams",
    "can_prune",
    "execute_prune",
]
