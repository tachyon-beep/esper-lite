"""Leyline Signals - Training state observations.

Two tiers of signal representation:
- FastTrainingSignals: NamedTuple for hot path (no GC pressure)
- TrainingSignals: Full dataclass for rich context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import NamedTuple


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TensorSchema(IntEnum):
    """Feature indices for the observation vector.

    Maps feature names to tensor indices for vectorized PPO training.
    Use this to slice state vectors by name without string lookups.

    Total: 27 base features (V1 compatible)
    """
    # Core state (2)
    EPOCH = 0
    GLOBAL_STEP = 1

    # Loss metrics (3)
    TRAIN_LOSS = 2
    VAL_LOSS = 3
    LOSS_DELTA = 4

    # Accuracy metrics (4)
    TRAIN_ACCURACY = 5
    VAL_ACCURACY = 6
    ACCURACY_DELTA = 7
    PLATEAU_EPOCHS = 8

    # Best tracking (2)
    BEST_VAL_ACCURACY = 9
    BEST_VAL_LOSS = 10

    # History (loss - 5 slots: 11-15)
    LOSS_HIST_0 = 11
    LOSS_HIST_1 = 12
    LOSS_HIST_2 = 13
    LOSS_HIST_3 = 14
    LOSS_HIST_4 = 15

    # History (accuracy - 5 slots: 16-20)
    ACC_HIST_0 = 16
    ACC_HIST_1 = 17
    ACC_HIST_2 = 18
    ACC_HIST_3 = 19
    ACC_HIST_4 = 20

    # Seed state (6)
    HAS_ACTIVE_SEED = 21
    SEED_STAGE = 22
    SEED_EPOCHS_IN_STAGE = 23
    SEED_ALPHA = 24
    SEED_IMPROVEMENT = 25
    AVAILABLE_SLOTS = 26


# Total feature count for V1 compatibility
TENSOR_SCHEMA_SIZE = 27


class FastTrainingSignals(NamedTuple):
    """Lightweight signals for PPO data plane.

    Contains ONLY numeric data required for policy network inference.
    No strings, no datetimes, fixed-size history tuples.
    Zero GC pressure - this is a named tuple (immutable, stack-allocated).
    """
    # Core state
    epoch: int
    global_step: int

    # Loss metrics
    train_loss: float
    val_loss: float
    loss_delta: float

    # Accuracy metrics
    train_accuracy: float
    val_accuracy: float
    accuracy_delta: float
    plateau_epochs: int

    # Best tracking
    best_val_accuracy: float
    best_val_loss: float

    # Fixed-size history (last 5 values)
    loss_history_5: tuple[float, float, float, float, float]
    accuracy_history_5: tuple[float, float, float, float, float]

    # Seed state
    has_active_seed: int  # 0 or 1
    seed_stage: int       # SeedStage value
    seed_epochs_in_stage: int
    seed_alpha: float
    seed_improvement: float
    available_slots: int

    def to_vector(self) -> list[float]:
        """Convert to flat feature vector matching TensorSchema."""
        return [
            float(self.epoch),
            float(self.global_step),
            self.train_loss,
            self.val_loss,
            self.loss_delta,
            self.train_accuracy,
            self.val_accuracy,
            self.accuracy_delta,
            float(self.plateau_epochs),
            self.best_val_accuracy,
            self.best_val_loss,
            *self.loss_history_5,
            *self.accuracy_history_5,
            float(self.has_active_seed),
            float(self.seed_stage),
            float(self.seed_epochs_in_stage),
            self.seed_alpha,
            self.seed_improvement,
            float(self.available_slots),
        ]

    @staticmethod
    def empty() -> "FastTrainingSignals":
        """Create empty/default signals."""
        return FastTrainingSignals(
            epoch=0, global_step=0,
            train_loss=0.0, val_loss=0.0, loss_delta=0.0,
            train_accuracy=0.0, val_accuracy=0.0, accuracy_delta=0.0,
            plateau_epochs=0, best_val_accuracy=0.0, best_val_loss=float('inf'),
            loss_history_5=(0.0, 0.0, 0.0, 0.0, 0.0),
            accuracy_history_5=(0.0, 0.0, 0.0, 0.0, 0.0),
            has_active_seed=0, seed_stage=0, seed_epochs_in_stage=0,
            seed_alpha=0.0, seed_improvement=0.0, available_slots=1,
        )


@dataclass(slots=True)
class TrainingMetrics:
    """Metrics from the training loop.

    Uses __slots__ for reduced memory footprint and faster attribute access.
    """

    epoch: int = 0
    global_step: int = 0

    # Loss metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    loss_delta: float = 0.0  # Change from previous epoch (positive = improvement)

    # Accuracy metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    accuracy_delta: float = 0.0

    # Trend indicators
    plateau_epochs: int = 0  # Epochs without improvement
    best_val_accuracy: float = 0.0
    best_val_loss: float = float('inf')

    # Gradient health
    grad_norm_host: float = 0.0
    grad_norm_seed: float = 0.0


@dataclass
class TrainingSignals:
    """Complete signals from training loop to Tamiyo.

    This is the observation space for Tamiyo's decision making.
    For high-frequency PPO training, use to_fast() to get a FastTrainingSignals.
    """

    # Training state
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)

    # Seed states
    active_seeds: list[str] = field(default_factory=list)  # seed_ids
    available_slots: int = 0
    seed_stage: int = 0
    seed_epochs_in_stage: int = 0
    seed_alpha: float = 0.0
    seed_improvement: float = 0.0

    # Resource state
    gpu_memory_used: float = 0.0  # GB
    gpu_utilization: float = 0.0  # 0-1

    # History (for trend analysis)
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)

    # Timing
    epoch_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=_utc_now)

    def to_fast(
        self,
        seed_stage: int | None = None,
        seed_epochs_in_stage: int | None = None,
        seed_alpha: float | None = None,
        seed_improvement: float | None = None,
    ) -> FastTrainingSignals:
        """Convert to FastTrainingSignals for PPO data plane.

        Args:
            seed_stage: Current seed stage (SeedStage value)
            seed_epochs_in_stage: Epochs in current stage
            seed_alpha: Current blending alpha
            seed_improvement: Improvement since stage start

        Returns:
            FastTrainingSignals with fixed-size history tuples
        """
        # Pad/truncate history to exactly 5 elements
        loss_hist = self.loss_history[-5:] if self.loss_history else []
        while len(loss_hist) < 5:
            loss_hist.insert(0, 0.0)
        acc_hist = self.accuracy_history[-5:] if self.accuracy_history else []
        while len(acc_hist) < 5:
            acc_hist.insert(0, 0.0)

        return FastTrainingSignals(
            epoch=self.metrics.epoch,
            global_step=self.metrics.global_step,
            train_loss=self.metrics.train_loss,
            val_loss=self.metrics.val_loss,
            loss_delta=self.metrics.loss_delta,
            train_accuracy=self.metrics.train_accuracy,
            val_accuracy=self.metrics.val_accuracy,
            accuracy_delta=self.metrics.accuracy_delta,
            plateau_epochs=self.metrics.plateau_epochs,
            best_val_accuracy=self.metrics.best_val_accuracy,
            best_val_loss=min(self.metrics.best_val_loss, 10.0),  # Clamp inf to reasonable max
            loss_history_5=tuple(loss_hist),
            accuracy_history_5=tuple(acc_hist),
            has_active_seed=1 if self.active_seeds else 0,
            seed_stage=self.seed_stage if seed_stage is None else seed_stage,
            seed_epochs_in_stage=self.seed_epochs_in_stage if seed_epochs_in_stage is None else seed_epochs_in_stage,
            seed_alpha=self.seed_alpha if seed_alpha is None else seed_alpha,
            seed_improvement=self.seed_improvement if seed_improvement is None else seed_improvement,
            available_slots=self.available_slots,
        )
