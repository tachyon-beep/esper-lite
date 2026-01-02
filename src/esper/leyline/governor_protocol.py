"""Leyline Governor Protocol - Contract for fail-safe training watchdogs.

GovernorProtocol defines the interface for training watchdog systems that
detect catastrophic failures (NaN, loss explosions) and can rollback.

Used by:
- tolaria: Implements TolariaGovernor
- simic: Uses governor interface for training safety
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GovernorReport:
    """Report from a rollback event.

    Note: consecutive_panics reflects the post-rollback state (always 0 after
    successful rollback). The pre-rollback count is captured in the
    GOVERNOR_ROLLBACK telemetry event's GovernorRollbackPayload.

    Note: loss_at_panic is NaN if rollback was triggered without a preceding
    panic (e.g., manual rollback). Use math.isnan() to check.
    """

    reason: str
    loss_at_panic: float
    loss_threshold: float
    consecutive_panics: int
    rollback_occurred: bool


class GovernorProtocol(Protocol):
    """Protocol for catastrophic failure detection and recovery.

    Governors monitor training for numerical disasters (NaN, loss explosions)
    and provide rollback capability to recover to a known-good state.

    Implementations:
        - tolaria.TolariaGovernor: Full implementation with statistical detection
    """

    def snapshot(self) -> None:
        """Save Last Known Good state for potential rollback.

        Called after validation passes to create a recovery point.
        """
        ...

    def check_vital_signs(self, current_loss: float) -> bool:
        """Check if the system has suffered catastrophic failure.

        Returns True only for truly catastrophic failures:
        - NaN or Inf in loss (immediate)
        - Loss exceeds detection thresholds after consecutive panics

        Args:
            current_loss: The current validation loss to check.

        Returns:
            True if catastrophic failure detected, False otherwise.
        """
        ...

    def reset(self) -> None:
        """Reset governor state for a new episode.

        Clears loss history, panic counters, and takes a fresh snapshot.
        """
        ...

    def get_punishment_reward(self) -> float:
        """Return the negative reward for RL agent punishment.

        Used to inject negative reward into PPO buffer when rollback occurs.
        """
        ...


__all__ = ["GovernorProtocol", "GovernorReport"]
