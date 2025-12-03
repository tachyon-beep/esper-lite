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


# Stabilization detection: block germination until explosive growth phase ends
# This ensures seeds only get credit for improvements AFTER natural training gains exhaust.
#
# Even with counterfactual validation, stabilization gating helps during TRAINING stage
# (before alpha > 0) where germination during explosive growth can cause credit
# misattribution. Re-enabled per DRL expert review recommendation.
#
# Default values (can be overridden per-tracker):
# - CIFAR-10: 3% threshold, 3 epochs (standard)
# - TinyStories/LLMs: Consider lower threshold (~1%) since relative improvements are naturally smaller
STABILIZATION_THRESHOLD = 0.03  # 3% relative improvement (lower = stricter = stabilizes later)
STABILIZATION_EPOCHS = 3        # Require 3 consecutive stable epochs before germination allowed


@dataclass
class SignalTracker:
    """Tracks training signals over time and computes derived metrics.

    Stabilization Parameters:
        stabilization_threshold: Relative loss improvement threshold (default: 0.03 = 3%).
            Epochs with improvement >= threshold are considered "explosive growth".
            Set lower for LLMs (e.g., 0.01) where relative improvements are smaller.
        stabilization_epochs: Consecutive stable epochs required before germination (default: 3).
            Set to 0 to disable stabilization gating entirely.
    """

    # Configuration
    plateau_threshold: float = 0.5  # Min improvement to not count as plateau
    history_window: int = 10
    env_id: int | None = None  # Optional environment identifier for telemetry

    # Stabilization parameters (task-specific tuning)
    stabilization_threshold: float = STABILIZATION_THRESHOLD
    stabilization_epochs: int = STABILIZATION_EPOCHS

    # History windows (initialized in __post_init__ with history_window)
    _loss_history: deque[float] = field(default_factory=deque)
    _accuracy_history: deque[float] = field(default_factory=deque)

    # Best values seen
    _best_accuracy: float = 0.0
    _plateau_count: int = 0

    # Stabilization latch (for dynamic germination gating)
    # Once True, stays True - prevents re-locking after successful seeds
    _is_stabilized: bool = False
    _stable_count: int = 0

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

        # Stabilization tracking (latch behavior - once True, stays True)
        # Guards against germinating during explosive growth phase
        EPS = 1e-8
        if not self._is_stabilized and self._prev_loss < float('inf'):
            if self._prev_loss > EPS:
                relative_improvement = loss_delta / self._prev_loss
                # Check: improvement is small AND loss didn't spike (not diverging)
                is_stable_epoch = (
                    relative_improvement < self.stabilization_threshold and
                    val_loss < self._prev_loss * 1.5  # Sanity: not diverging
                )
                if is_stable_epoch:
                    self._stable_count += 1
                    if self._stable_count >= self.stabilization_epochs:
                        self._is_stabilized = True
                        env_str = f"ENV {self.env_id}" if self.env_id is not None else "Tamiyo"
                        if self.stabilization_epochs == 0:
                            # Stabilization disabled - just note when it happened
                            print(f"[{env_str}] Host stabilized at epoch {epoch} - germination now allowed")
                        else:
                            print(f"[{env_str}] Host stabilized at epoch {epoch} "
                                  f"({self._stable_count}/{self.stabilization_epochs} stable) - germination now allowed")
                else:
                    self._stable_count = 0

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
            host_stabilized=1 if self._is_stabilized else 0,
            best_val_accuracy=self._best_accuracy,
            best_val_loss=min(self._loss_history) if self._loss_history else float('inf'),
        )

        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0

        if active_seeds and active_seeds[0] is not None:
            first_seed = active_seeds[0]
            seed_stage = int(first_seed.stage)
            # SeedState contract guarantees metrics and epochs_in_stage integer.
            seed_epochs_in_stage = first_seed.epochs_in_stage
            seed_alpha = first_seed.alpha
            seed_improvement = first_seed.metrics.improvement_since_stage_start

        # Build TrainingSignals (Leyline format with nested metrics)
        signals = TrainingSignals(
            metrics=metrics,
            active_seeds=[s.seed_id for s in active_seeds],
            available_slots=available_slots,
            seed_stage=seed_stage,
            seed_epochs_in_stage=seed_epochs_in_stage,
            seed_alpha=seed_alpha,
            seed_improvement=seed_improvement,
            loss_history=list(self._loss_history)[-5:],  # Last 5 for compat
            accuracy_history=list(self._accuracy_history)[-5:],
        )

        # Update previous values for next iteration
        self._prev_loss = val_loss
        self._prev_accuracy = val_accuracy

        return signals

    def reset(self) -> None:
        """Reset tracker state."""
        # Recreate deques with current history_window (not just clear)
        self._loss_history = deque(maxlen=self.history_window)
        self._accuracy_history = deque(maxlen=self.history_window)
        self._best_accuracy = 0.0
        self._plateau_count = 0
        self._prev_accuracy = 0.0
        self._prev_loss = float('inf')

        # Reset stabilization latch
        self._is_stabilized = False
        self._stable_count = 0

    @property
    def is_stabilized(self) -> bool:
        """Host training has stabilized (latch - stays True once set)."""
        return self._is_stabilized


__all__ = [
    "SignalTracker",
    "STABILIZATION_THRESHOLD",
    "STABILIZATION_EPOCHS",
]
