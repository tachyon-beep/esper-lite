"""Karn Constants - TUI display and presentation thresholds.

OWNERSHIP BOUNDARY:
    This module owns PRESENTATION/DISPLAY thresholds - anything that affects:
    - TUI color coding (green/yellow/red status indicators)
    - Dense trace trigger sensitivity (when to capture diagnostics)
    - Health monitoring display alerts (GPU, memory warnings)
    - Visual anomaly indicators (not training behavior)

    DO NOT add training behavior constants here. Those belong in leyline/__init__.py.
    When in doubt: if it affects what the USER SEES, it belongs here.
    If it affects TRAINING OUTCOMES, it belongs in leyline.

    Some thresholds are imported from leyline to ensure consistency between
    training detection and display. These are clearly marked.

Usage:
    from esper.karn.constants import AnomalyThresholds, HealthThresholds

    if loss_ratio > AnomalyThresholds.LOSS_SPIKE_MULTIPLIER:
        trigger_dense_trace()
"""

from __future__ import annotations

# Import shared thresholds from leyline (single source of truth for training behavior)
from esper.leyline import (
    DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
    DEFAULT_ENTROPY_WARNING_THRESHOLD,
    DEFAULT_GOVERNOR_LOSS_MULTIPLIER,
)


class AnomalyThresholds:
    """Thresholds for dense trace triggering (Tier 3 capture).

    These control when the system captures detailed diagnostics.
    """

    # Loss spike: trigger dense trace if loss > N× rolling EMA.
    # INTENTIONALLY LOWER than leyline's DEFAULT_GOVERNOR_LOSS_MULTIPLIER (3.0).
    # TUI should warn users BEFORE governor panics, giving time to investigate.
    # Governor: 3.0× = panic/rollback. TUI: 2.0× = "hey, something's off".
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
    Entropy threshold is imported from leyline for consistency with training detection.
    """

    # Value collapse: critic outputs have std below this → collapse
    VALUE_STD_COLLAPSE: float = 0.01

    # Entropy collapse: policy entropy below this → deterministic
    # (from leyline - single source of truth for training behavior)
    ENTROPY_COLLAPSE: float = DEFAULT_ENTROPY_COLLAPSE_THRESHOLD

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
    Entropy thresholds align with leyline for consistency.
    """

    # Entropy (healthy starts near ln(4) ≈ 1.39 for 4 actions)
    ENTROPY_MAX: float = 1.39  # ln(4) for 4 actions
    # Warning threshold from leyline (single source of truth)
    ENTROPY_WARNING: float = DEFAULT_ENTROPY_WARNING_THRESHOLD
    # Critical threshold from leyline (single source of truth)
    ENTROPY_CRITICAL: float = DEFAULT_ENTROPY_COLLAPSE_THRESHOLD

    # Clip fraction (target 0.1-0.2)
    CLIP_WARNING: float = 0.25
    CLIP_CRITICAL: float = 0.3

    # Explained variance (value learning quality)
    # In PPO, explained variance starts near 0 and improves as value function learns.
    # Negative values mean value function increases variance (harmful).
    EXPLAINED_VAR_WARNING: float = 0.0   # Value function not helping
    EXPLAINED_VAR_CRITICAL: float = -0.5  # Value function actively harmful

    # Gradient norm
    GRAD_NORM_WARNING: float = 5.0
    GRAD_NORM_CRITICAL: float = 10.0

    # KL divergence (policy change magnitude)
    KL_WARNING: float = 0.05

    # Action distribution (WAIT dominance is suspicious)
    WAIT_DOMINANCE_WARNING: float = 0.7  # > 70% WAIT


class VitalSignsThresholds:
    """Thresholds for vital signs monitoring.

    These detect training failure patterns for display purposes.
    """

    # Loss spike relative to recent average.
    # INTENTIONALLY LOWER than leyline's DEFAULT_GOVERNOR_LOSS_MULTIPLIER (3.0).
    # Same rationale as AnomalyThresholds: warn before governor panics.
    LOSS_SPIKE_MULTIPLIER: float = 2.0

    # Epochs without improvement before stagnation warning
    STAGNATION_EPOCHS: int = 20
