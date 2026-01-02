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

    Explained Variance thresholds follow DRL best practices:
    - EV=1.0: Perfect value prediction
    - EV=0.0: Value function explains nothing (useless)
    - EV<0.0: Value function increases variance (harmful)
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

    # Explained variance (value learning quality) - DRL CORRECTED
    # EV=0 means value function provides no advantage over REINFORCE
    EXPLAINED_VAR_WARNING: float = 0.3   # Value function weak but learning
    EXPLAINED_VAR_CRITICAL: float = 0.0   # Value function useless or harmful

    # Gradient norm
    GRAD_NORM_WARNING: float = 5.0
    GRAD_NORM_CRITICAL: float = 10.0

    # KL divergence (policy change magnitude) - ADDED per DRL review
    KL_WARNING: float = 0.015   # Mild policy drift
    KL_CRITICAL: float = 0.03   # Excessive policy change

    # Advantage normalization thresholds - ADDED per DRL review
    # Healthy advantage std is ~1.0 after normalization
    ADVANTAGE_STD_WARNING: float = 2.0      # High variance
    ADVANTAGE_STD_CRITICAL: float = 3.0     # Extreme variance
    ADVANTAGE_STD_LOW_WARNING: float = 0.5  # Too little variance
    ADVANTAGE_STD_COLLAPSED: float = 0.1    # Advantage normalization broken

    # Action distribution (WAIT dominance is suspicious)
    WAIT_DOMINANCE_WARNING: float = 0.7  # > 70% WAIT

    # Ratio statistics thresholds (PPO policy ratio should stay near 1.0)
    RATIO_MAX_CRITICAL: float = 2.0   # Policy changing too fast
    RATIO_MAX_WARNING: float = 1.5
    RATIO_MIN_CRITICAL: float = 0.3   # Policy changing too fast in other direction
    RATIO_MIN_WARNING: float = 0.5
    RATIO_STD_WARNING: float = 0.5    # High variance in updates

    # Gradient health percentage thresholds
    GRAD_HEALTH_WARNING: float = 0.8   # >80% healthy layers is OK
    GRAD_HEALTH_CRITICAL: float = 0.5  # <50% healthy is critical


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


class DisplayThresholds:
    """Thresholds for modal panel displays.

    These control styling in EnvDetailScreen, HistoricalEnvDetail, and other modals.
    All values are presentation-layer only and do not affect training behavior.
    """

    # Reward health (DRL Expert recommendations)
    # PBRS should be 10-40% of total reward to shape without dominating
    PBRS_HEALTHY_MIN: float = 0.1   # 10% of total reward
    PBRS_HEALTHY_MAX: float = 0.4   # 40% of total reward
    GAMING_RATE_HEALTHY_MAX: float = 0.05  # <5% anti-gaming triggers

    # Growth ratio (model size overhead from fossilized seeds)
    # >20% parameter growth = yellow warning
    GROWTH_RATIO_WARNING: float = 1.2

    # Stagnation (epochs without improvement)
    # >10 epochs = red momentum indicator
    MOMENTUM_STALL_THRESHOLD: int = 10

    # Blueprint success rates (graveyard display)
    # ≥50% fossilized = green, ≥25% = yellow, <25% = red
    BLUEPRINT_SUCCESS_GREEN: float = 0.50
    BLUEPRINT_SUCCESS_YELLOW: float = 0.25

    # Interaction synergy thresholds (SeedCard display)
    # Show synergy indicator if |interaction_sum| > 0.5
    INTERACTION_SYNERGY_THRESHOLD: float = 0.5
    # Show boost indicator if boost_received > 0.1
    BOOST_RECEIVED_THRESHOLD: float = 0.1

    # Contribution velocity (trend arrows)
    # Show trend if |velocity| > 0.01
    CONTRIBUTION_VELOCITY_EPSILON: float = 0.01
