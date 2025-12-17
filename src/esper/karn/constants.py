"""Karn Constants - Centralized thresholds for telemetry subsystem.

All magic numbers for anomaly detection, health monitoring, and TUI
display are defined here. This enables:
- Global tuning without hunting through multiple files
- Clear documentation of threshold rationale
- Type-safe access via class attributes

Usage:
    from esper.karn.constants import AnomalyThresholds, HealthThresholds

    if loss_ratio > AnomalyThresholds.LOSS_SPIKE_MULTIPLIER:
        trigger_dense_trace()
"""

from __future__ import annotations


class AnomalyThresholds:
    """Thresholds for dense trace triggering (Tier 3 capture).

    These control when the system captures detailed diagnostics.
    """

    # Loss spike: trigger if loss > N× rolling EMA
    LOSS_SPIKE_MULTIPLIER: float = 2.0

    # Accuracy drop: trigger if accuracy drops by N percentage points
    ACCURACY_DROP_POINTS: float = 5.0

    # Gradient explosion: trigger if grad norm > N× rolling EMA
    GRADIENT_EXPLOSION_MULTIPLIER: float = 100.0

    # Dense trace window: capture N epochs after trigger
    TRACE_WINDOW_EPOCHS: int = 3


class PolicyThresholds:
    """Thresholds for PPO policy anomaly detection.

    These detect pathological policy behavior during RL training.
    """

    # Value collapse: critic outputs have std below this → collapse
    VALUE_STD_COLLAPSE: float = 0.01

    # Entropy collapse: policy entropy below this → deterministic
    ENTROPY_COLLAPSE: float = 0.1

    # KL spike: policy change above this → large update
    KL_SPIKE: float = 0.1

    # Rolling window for anomaly detection
    WINDOW_SIZE: int = 10


class HealthThresholds:
    """Thresholds for system health monitoring.

    These trigger warnings and errors for resource/gradient issues.
    """

    # GPU memory utilization (0-1)
    GPU_UTILIZATION_WARNING: float = 0.9
    MEMORY_WARNING_THRESHOLD: float = 0.85
    MEMORY_WARNING_COOLDOWN_SECONDS: float = 60.0

    # Gradient norm thresholds
    GRAD_NORM_WARNING: float = 50.0
    GRAD_NORM_ERROR: float = 100.0

    # Gradient explosion indicator (likely Inf)
    GRAD_NORM_EXPLOSION: float = 1e10


class TUIThresholds:
    """Thresholds for TUI color-coded health display.

    These control green/yellow/red status indicators.
    """

    # Entropy (healthy starts near ln(4) ≈ 1.39 for 4 actions)
    ENTROPY_MAX: float = 1.39  # ln(4) for 4 actions
    ENTROPY_WARNING: float = 0.5
    ENTROPY_CRITICAL: float = 0.3

    # Clip fraction (target 0.1-0.2)
    CLIP_WARNING: float = 0.25
    CLIP_CRITICAL: float = 0.3

    # Explained variance (value learning quality)
    EXPLAINED_VAR_WARNING: float = 0.7
    EXPLAINED_VAR_CRITICAL: float = 0.5

    # Gradient norm
    GRAD_NORM_WARNING: float = 5.0
    GRAD_NORM_CRITICAL: float = 10.0

    # KL divergence (policy change magnitude)
    KL_WARNING: float = 0.05

    # Action distribution (WAIT dominance is suspicious)
    WAIT_DOMINANCE_WARNING: float = 0.7  # > 70% WAIT


class VitalSignsThresholds:
    """Thresholds for vital signs monitoring.

    These detect training failure patterns.
    """

    # Loss spike relative to recent average
    LOSS_SPIKE_MULTIPLIER: float = 2.0

    # Epochs without improvement before stagnation warning
    STAGNATION_EPOCHS: int = 20
