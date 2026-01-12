"""ADVANCE operation handler.

Handles advancing seeds through lifecycle stages. ADVANCE triggers
gate evaluation and stage transitions when preconditions are met.

Valid ADVANCE transitions:
- GERMINATED -> TRAINING (G1 gate)
- TRAINING -> BLENDING (G2 gate)
- BLENDING -> HOLDING (G3 gate)

Note: HOLDING -> FOSSILIZED uses FOSSILIZE operation, not ADVANCE.
"""

from __future__ import annotations

from esper.leyline import SeedStage
from esper.simic.training.handlers.base import HandlerContext, HandlerResult

# Stages that can be advanced (excludes terminal stages and HOLDING)
ADVANCEABLE_STAGES = frozenset(
    {
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
    }
)


def can_advance(ctx: HandlerContext) -> bool:
    """Check if ADVANCE operation is allowed in the given context.

    Preconditions:
    - The slot must have an active seed (seed_state is not None)
    - The seed must be in an advanceable stage (GERMINATED, TRAINING, BLENDING)

    Args:
        ctx: Handler context with slot and seed state.

    Returns:
        True if advance can proceed (gate evaluation will occur).
    """
    if ctx.seed_state is None:
        return False

    return ctx.seed_state.stage in ADVANCEABLE_STAGES


def execute_advance(ctx: HandlerContext) -> HandlerResult:
    """Execute the ADVANCE operation.

    Triggers the slot's advance_stage() method which evaluates the
    appropriate quality gate (G1, G2, or G3) and transitions if passed.

    Args:
        ctx: Handler context with environment and model state.

    Returns:
        HandlerResult with success=True if gate passed and transition occurred.
    """
    # Verify preconditions
    if ctx.seed_state is None:
        return HandlerResult(
            success=False,
            error="Cannot advance: no active seed in slot",
        )

    if ctx.seed_state.stage not in ADVANCEABLE_STAGES:
        return HandlerResult(
            success=False,
            error=f"Cannot advance: seed in {ctx.seed_state.stage.name} is not advanceable",
        )

    # Record pre-advance state for telemetry
    pre_stage = ctx.seed_state.stage

    # Execute gate evaluation and potential transition
    gate_result = ctx.slot.advance_stage()

    # The slot's advance_stage() handles all internal state updates
    # We just need to report success based on gate result
    return HandlerResult(
        success=gate_result.passed,
        telemetry={
            "pre_stage": pre_stage.name,
            "post_stage": ctx.seed_state.stage.name,
            "gate_passed": gate_result.passed,
            "gate_level": gate_result.gate.name,
            "gate_message": gate_result.message,
        },
    )


__all__ = [
    "ADVANCEABLE_STAGES",
    "can_advance",
    "execute_advance",
]
