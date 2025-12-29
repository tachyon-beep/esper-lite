"""Tamiyo Tracker - Training signal observation.

SignalTracker maintains running statistics for decision-making.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.leyline import (
    TrainingSignals,
    TrainingMetrics,
    TelemetryEvent,
    TelemetryEventType,
    TamiyoInitiatedPayload,
    DEFAULT_STABILIZATION_THRESHOLD,
    DEFAULT_STABILIZATION_EPOCHS,
)
from esper.nissa import get_hub

if TYPE_CHECKING:
    from esper.kasmina import SeedState

logger = logging.getLogger(__name__)


# Stabilization detection: block germination until explosive growth phase ends
# This ensures seeds only get credit for improvements AFTER natural training gains exhaust.
#
# Even with counterfactual validation, stabilization gating helps during TRAINING stage
# (before alpha > 0) where germination during explosive growth can cause credit
# misattribution. Re-enabled per DRL expert review recommendation.
#
# Default values imported from leyline (can be overridden per-tracker):
# - CIFAR-10: 3% threshold, 3 epochs (standard)
# - TinyStories/LLMs: Consider lower threshold (~1%) since relative improvements are smaller


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
    # P2-A: Renamed for clarity - this is percentage points on 0-100 scale
    plateau_threshold_pct: float = 0.5  # Min accuracy improvement (pct points) to not count as plateau
    history_window: int = 10
    env_id: int | None = None  # Optional environment identifier for telemetry

    # Stabilization parameters (from leyline, task-specific overrides allowed)
    stabilization_threshold: float = DEFAULT_STABILIZATION_THRESHOLD
    stabilization_epochs: int = DEFAULT_STABILIZATION_EPOCHS
    # B9-DRL-02: Max allowed regression (pct) to still count as stable/noise
    # Prevents divergence (>5%) from counting, but allows PPO noise (<5%)
    regression_threshold: float = 0.05

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

    def __post_init__(self) -> None:
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

        # P2-A: Validate accuracy scale (warn on 0-1 scale, expected 0-100)
        if val_accuracy > 1.0 and val_accuracy <= 100.0:
            pass  # Expected range
        elif val_accuracy > 100.0:
            logger.warning(
                f"val_accuracy {val_accuracy} exceeds 100 - expected 0-100 scale"
            )
        elif 0.0 < val_accuracy <= 1.0 and epoch > 0:
            # Only warn after epoch 0 (initial accuracy can legitimately be low)
            logger.warning(
                f"val_accuracy {val_accuracy} appears to be on 0-1 scale - "
                f"expected 0-100 scale (percentage points)"
            )

        # Update plateau counter
        if accuracy_delta < self.plateau_threshold_pct:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        # Stabilization tracking (latch behavior - once True, stays True)
        # Guards against germinating during explosive growth phase
        EPS = 1e-8
        if not self._is_stabilized and self._prev_loss < float('inf'):
            if self._prev_loss > EPS:
                relative_improvement = loss_delta / self._prev_loss
                # B9-DRL-02 FIX: Symmetric Stability Window
                # Old logic: loss_delta >= 0 was too strict (rejected normal PPO noise)
                # and val_loss < prev * 1.5 was too loose (50% regression allowed)
                #
                # New logic uses symmetric thresholds:
                # 1. Block explosive growth: rel_imp >= stabilization_threshold (>3%)
                # 2. Block divergence: rel_imp <= -regression_threshold (<-5%)
                # 3. Allow plateau/noise: -5% < rel_imp < 3%
                is_stable_epoch = (
                    relative_improvement > -self.regression_threshold and
                    relative_improvement < self.stabilization_threshold
                )
                if is_stable_epoch:
                    self._stable_count += 1
                    if self._stable_count >= self.stabilization_epochs:
                        self._is_stabilized = True
                        # Emit TAMIYO_INITIATED telemetry (console output via Nissa backend)
                        hub = get_hub()
                        # Only emit if env_id is set (telemetry requires env context)
                        if self.env_id is not None:
                            hub.emit(TelemetryEvent(
                                event_type=TelemetryEventType.TAMIYO_INITIATED,
                                epoch=epoch,
                                message=(
                                    f"Host stabilized - germination now allowed "
                                    f"(env_id={self.env_id}, stable_count={self._stable_count}, "
                                    f"stabilization_epochs={self.stabilization_epochs}, val_loss={val_loss:.4f})"
                                ),
                                data=TamiyoInitiatedPayload(
                                    env_id=self.env_id,
                                    epoch=epoch,
                                    stable_count=self._stable_count,
                                    stabilization_epochs=self.stabilization_epochs,
                                    val_loss=val_loss,
                                ),
                            ))
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
            # NOTE: best_val_loss is "best in window" (last N epochs), not global best
            best_val_loss=min(self._loss_history) if self._loss_history else float('inf'),
        )

        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0

        if active_seeds:
            # Multi-slot summary seed selection rule (deterministic, documented):
            # 1) Prefer highest stage
            # 2) Tie-break by highest alpha
            # 3) Tie-break by most negative counterfactual_contribution (safety)
            # 4) Final tie-break by seed_id for determinism
            seed_ids = [s.seed_id for s in active_seeds]
            if len(seed_ids) != len(set(seed_ids)):
                raise RuntimeError(f"Duplicate seed_id(s) in active_seeds: {seed_ids}")

            def summary_key(seed: "SeedState") -> tuple[int, float, float, str]:
                stage = int(seed.stage)
                alpha = float(seed.alpha)
                counterfactual = float("inf")
                if seed.metrics and seed.metrics.counterfactual_contribution is not None:
                    counterfactual = seed.metrics.counterfactual_contribution
                return (stage, alpha, -counterfactual, seed.seed_id)

            summary_seed = max(active_seeds, key=summary_key)
            seed_stage = int(summary_seed.stage)
            seed_epochs_in_stage = summary_seed.epochs_in_stage
            seed_alpha = summary_seed.alpha
            seed_improvement = (
                summary_seed.metrics.improvement_since_stage_start
                if summary_seed.metrics else 0.0
            )

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

    def peek(
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
        """Build TrainingSignals without modifying tracker state.

        Use this for bootstrap value computation where you need to query
        the value of a hypothetical state without advancing the tracker.
        Uses current tracker state (loss/accuracy history, best values)
        but with caller-provided seed information.

        This is read-only - does NOT update histories, deltas, or stabilization state.
        """
        # Compute deltas from current tracker state (same as update())
        loss_delta = self._prev_loss - val_loss  # Positive = improvement
        accuracy_delta = val_accuracy - self._prev_accuracy

        # Build TrainingMetrics using current tracker state
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
            best_val_accuracy=max(self._best_accuracy, val_accuracy),
            best_val_loss=min(
                min(self._loss_history) if self._loss_history else float('inf'),
                val_loss
            ),
        )

        # Same seed summary logic as update()
        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0

        if active_seeds:
            seed_ids = [s.seed_id for s in active_seeds]
            if len(seed_ids) != len(set(seed_ids)):
                raise RuntimeError(f"Duplicate seed_id(s) in active_seeds: {seed_ids}")

            def summary_key(seed: "SeedState") -> tuple[int, float, float, str]:
                stage = int(seed.stage)
                alpha = float(seed.alpha)
                counterfactual = float("inf")
                if seed.metrics and seed.metrics.counterfactual_contribution is not None:
                    counterfactual = seed.metrics.counterfactual_contribution
                return (stage, alpha, -counterfactual, seed.seed_id)

            summary_seed = max(active_seeds, key=summary_key)
            seed_stage = int(summary_seed.stage)
            seed_epochs_in_stage = summary_seed.epochs_in_stage
            seed_alpha = summary_seed.alpha
            seed_improvement = (
                summary_seed.metrics.improvement_since_stage_start
                if summary_seed.metrics else 0.0
            )

        return TrainingSignals(
            metrics=metrics,
            active_seeds=[s.seed_id for s in active_seeds],
            available_slots=available_slots,
            seed_stage=seed_stage,
            seed_epochs_in_stage=seed_epochs_in_stage,
            seed_alpha=seed_alpha,
            seed_improvement=seed_improvement,
            loss_history=list(self._loss_history)[-5:],
            accuracy_history=list(self._accuracy_history)[-5:],
        )

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
    # STABILIZATION_THRESHOLD and STABILIZATION_EPOCHS now in leyline as DEFAULT_*
]
