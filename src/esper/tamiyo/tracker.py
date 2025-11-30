"""Tamiyo Tracker - Training signal observation.

SignalTracker maintains running statistics for decision-making.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.leyline import TrainingSignals, TrainingMetrics

if TYPE_CHECKING:
    from esper.kasmina import SeedState


@dataclass
class SignalTracker:
    """Tracks training signals over time and computes derived metrics."""

    # Configuration
    plateau_threshold: float = 0.5  # Min improvement to not count as plateau
    history_window: int = 10

    # History windows (initialized in __post_init__ with history_window)
    _loss_history: deque[float] = field(default_factory=deque)
    _accuracy_history: deque[float] = field(default_factory=deque)

    # Best values seen
    _best_accuracy: float = 0.0
    _plateau_count: int = 0

    # Previous values for delta computation
    _prev_accuracy: float = 0.0
    _prev_loss: float = float('inf')

    def __post_init__(self):
        """Initialize deques with proper maxlen from history_window."""
        # Recreate deques with the correct maxlen from history_window parameter
        self._loss_history = deque(self._loss_history, maxlen=self.history_window)
        self._accuracy_history = deque(self._accuracy_history, maxlen=self.history_window)

    def update(
        self,
        epoch: int,
        global_step: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        active_seeds: list["SeedState"],
        available_slots: int = 1,
    ) -> TrainingSignals:
        """Update tracker and return current signals as TrainingSignals."""

        # Compute deltas
        loss_delta = self._prev_loss - val_loss  # Positive = improvement
        accuracy_delta = val_accuracy - self._prev_accuracy

        # Update plateau counter
        if accuracy_delta < self.plateau_threshold:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        # Update best
        if val_accuracy > self._best_accuracy:
            self._best_accuracy = val_accuracy

        # Update history
        self._loss_history.append(val_loss)
        self._accuracy_history.append(val_accuracy)

        # Build TrainingMetrics
        metrics = TrainingMetrics(
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            loss_delta=loss_delta,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            accuracy_delta=accuracy_delta,
            plateau_epochs=self._plateau_count,
            best_val_accuracy=self._best_accuracy,
            best_val_loss=min(self._loss_history) if self._loss_history else float('inf'),
        )

        # Build TrainingSignals (Leyline format with nested metrics)
        signals = TrainingSignals(
            metrics=metrics,
            active_seeds=[s.seed_id for s in active_seeds],
            available_slots=available_slots,
            loss_history=list(self._loss_history)[-5:],  # Last 5 for compat
            accuracy_history=list(self._accuracy_history)[-5:],
        )

        # Update previous values for next iteration
        self._prev_loss = val_loss
        self._prev_accuracy = val_accuracy

        return signals

    def reset(self) -> None:
        """Reset tracker state."""
        self._loss_history.clear()
        self._accuracy_history.clear()
        self._best_accuracy = 0.0
        self._plateau_count = 0
        self._prev_accuracy = 0.0
        self._prev_loss = float('inf')


__all__ = [
    "SignalTracker",
]
