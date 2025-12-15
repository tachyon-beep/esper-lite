"""Leyline Signals - Training state observations.

TrainingSignals is the structured signals contract produced by training loops
and consumed by controllers (Tamiyo/Simic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


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
    host_stabilized: int = 0  # 1 if explosive growth phase ended (latch)
    best_val_accuracy: float = 0.0
    best_val_loss: float = float('inf')

    # Gradient health
    grad_norm_host: float = 0.0
    grad_norm_seed: float = 0.0


@dataclass
class TrainingSignals:
    """Complete signals from training loop to Tamiyo.

    This is the observation space for Tamiyo's decision making.
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
    seed_counterfactual: float = 0.0

    # Resource state
    gpu_memory_used: float = 0.0  # GB
    gpu_utilization: float = 0.0  # 0-1

    # History (for trend analysis)
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)

    # Timing
    epoch_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=_utc_now)
