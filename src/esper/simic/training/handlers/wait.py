"""WAIT operation handler.

Handles the no-op WAIT operation. WAIT is always valid and successful -
it simply skips any lifecycle modification for this timestep.

Use cases:
- No slots require attention
- All slots are busy (mid-transition)
- Policy chooses to defer action
- Fallback when no valid operations are available
"""

from __future__ import annotations

from esper.simic.training.handlers.base import HandlerContext, HandlerResult


def can_wait(ctx: HandlerContext) -> bool:
    """Check if WAIT operation is allowed.

    WAIT is always allowed - it's the universal no-op.

    Args:
        ctx: Handler context (unused).

    Returns:
        Always True.
    """
    return True


def execute_wait(ctx: HandlerContext) -> HandlerResult:
    """Execute the WAIT operation.

    WAIT is a no-op that always succeeds. It doesn't modify any state.

    Args:
        ctx: Handler context (unused for state modification).

    Returns:
        HandlerResult with success=True.
    """
    return HandlerResult(
        success=True,
        telemetry={},
    )


__all__ = [
    "can_wait",
    "execute_wait",
]
