"""Base protocol and shared types for lifecycle operation handlers.

This module defines the contracts for action handlers extracted from
action_execution.py. Each handler is responsible for executing one
lifecycle operation (GERMINATE, ADVANCE, FOSSILIZE, PRUNE, WAIT, SET_ALPHA_TARGET).

The Strategy pattern enables:
- Independent testability of each operation
- Reduced cognitive load when reading action_execution.py
- Future extension without modifying the main execution loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from esper.leyline import SeedSlotProtocol, SeedStateProtocol, SlottedHostProtocol
    from esper.simic.training.parallel_env_state import ParallelEnvState


@dataclass(slots=True)
class HandlerContext:
    """Shared context for all lifecycle handlers.

    Contains the minimal state needed for handlers to make decisions
    and execute operations. This replaces the scattered local variables
    in the original execute_actions() loop.

    Attributes:
        env_idx: Index of the environment in the vectorized batch.
        slot_id: Target slot ID for the operation.
        env_state: Mutable per-environment state (episode tracking, accumulators).
        model: The slotted host model (MorphogeneticModel).
        slot: The target SeedSlot object (may be empty if slot_id not in model).
        seed_state: The SeedState if the slot has an active seed, else None.
        epoch: Current epoch number (1-indexed).
        max_epochs: Maximum epochs per episode.
        episodes_completed: Number of episodes completed so far.
    """

    env_idx: int
    slot_id: str
    env_state: "ParallelEnvState"
    model: "SlottedHostProtocol"
    slot: "SeedSlotProtocol"
    seed_state: "SeedStateProtocol | None"
    epoch: int
    max_epochs: int
    episodes_completed: int


@dataclass(slots=True)
class HandlerResult:
    """Result from executing a lifecycle handler.

    Each handler returns a HandlerResult indicating success/failure
    and any side effects that the caller should handle.

    Attributes:
        success: True if the operation executed successfully.
        telemetry: Optional telemetry data to emit.
        error: Error message if success=False.
    """

    success: bool = False
    telemetry: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class LifecycleHandler(Protocol):
    """Protocol for lifecycle operation handlers.

    Handlers follow the Strategy pattern: each implements can_execute()
    to check preconditions and execute() to perform the operation.

    This protocol is for documentation and type-checking purposes.
    Actual handlers are plain functions (can_X, execute_X) for simplicity.
    """

    def can_execute(self, ctx: HandlerContext) -> bool:
        """Check if this handler can execute given current state.

        Returns:
            True if preconditions are met and execution should proceed.
        """
        ...

    def execute(self, ctx: HandlerContext, **kwargs: Any) -> HandlerResult:
        """Execute the lifecycle operation.

        Args:
            ctx: Handler context with environment state.
            **kwargs: Operation-specific arguments.

        Returns:
            HandlerResult indicating success and any side effects.
        """
        ...


__all__ = [
    "HandlerContext",
    "HandlerResult",
    "LifecycleHandler",
]
