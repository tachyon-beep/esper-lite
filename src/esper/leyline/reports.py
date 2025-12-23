"""Leyline Reports - Structured reporting contracts.

Defines the shape of reports emitted by various subsystems:
- TrainingMetrics: Per-epoch training statistics (exported from signals.py)
- SeedMetrics: Seed module performance
- SeedStateReport: Full seed state snapshot
- FieldReport: Episode summary for RL training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.stages import SeedStage
from esper.leyline.signals import TrainingSignals

if TYPE_CHECKING:
    from esper.leyline.telemetry import SeedTelemetry


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class SeedMetrics:
    """Metrics tracked for a specific seed.

    Uses __slots__ for reduced memory footprint and faster attribute access.
    """

    # Training progress
    epochs_total: int = 0
    epochs_in_current_stage: int = 0

    # Performance
    initial_val_accuracy: float = 0.0
    current_val_accuracy: float = 0.0
    best_val_accuracy: float = 0.0
    accuracy_at_stage_start: float = 0.0

    # Computed
    total_improvement: float = 0.0
    improvement_since_stage_start: float = 0.0

    # Causal attribution (set by counterfactual validation)
    counterfactual_contribution: float | None = None
    # Contribution velocity: EMA of Δcontribution for fossilize lookahead
    contribution_velocity: float = 0.0
    # Previous contribution for velocity computation
    _prev_contribution: float | None = None

    # Inter-slot interaction tracking (set by counterfactual engine)
    # interaction_sum: Σ I_ij for all j ≠ i (total synergy from interactions)
    interaction_sum: float = 0.0
    # boost_received: max(I_ij) for j ≠ i (strongest interaction partner)
    boost_received: float = 0.0
    # upstream_alpha_sum: Σ alpha_j for slots j < i (position-aware blending)
    upstream_alpha_sum: float = 0.0
    # downstream_alpha_sum: Σ alpha_j for slots j > i (position-aware blending)
    downstream_alpha_sum: float = 0.0

    # Gradient activity (parameter-normalized) for G2 gate
    seed_gradient_norm_ratio: float = 0.0

    # Parameter counts for normalization/analytics
    seed_param_count: int = 0
    host_param_count: int = 0

    # Health
    gradient_norm_avg: float = 0.0

    # Blending
    current_alpha: float = 0.0
    alpha_ramp_step: int = 0


@dataclass
class SeedStateReport:
    """Report of a seed's current state from Kasmina.

    This is the primary contract for Kasmina → Tamiyo communication about seed status.
    """

    # Identity
    seed_id: str = ""
    slot_id: str = ""
    blueprint_id: str = ""

    # Lifecycle
    stage: SeedStage = SeedStage.UNKNOWN
    previous_stage: SeedStage = SeedStage.UNKNOWN
    previous_epochs_in_stage: int = 0  # Epochs in previous stage at transition (for PBRS)
    stage_entered_at: datetime = field(default_factory=_utc_now)
    alpha_mode: int = AlphaMode.HOLD.value

    # Alpha controller state (for observation parity + action masking learnability)
    alpha_target: float = 0.0
    alpha_steps_total: int = 0
    alpha_steps_done: int = 0
    time_to_target: int = 0
    alpha_velocity: float = 0.0
    alpha_algorithm: int = AlphaAlgorithm.ADD.value

    # Seed tempo (chosen at germination)
    blend_tempo_epochs: int = 5

    # Metrics
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

    # Per-seed telemetry snapshot for PPO observations (optional)
    telemetry: SeedTelemetry | None = None

    # Status flags
    is_healthy: bool = True
    is_improving: bool = False
    needs_attention: bool = False

    # Annotations
    annotations: dict[str, Any] = field(default_factory=dict)

    # Timestamp
    reported_at: datetime = field(default_factory=_utc_now)


@dataclass
class FieldReport:
    """Report of a seed's complete lifecycle for Simic learning.

    Collected when a seed reaches a terminal state (FOSSILIZED or PRUNED).
    """

    # Identity
    report_id: str = field(default_factory=lambda: str(uuid4()))
    seed_id: str = ""
    blueprint_id: str = ""
    slot_id: str = ""

    # Lifecycle summary
    final_stage: SeedStage = SeedStage.UNKNOWN
    success: bool = False  # True if fossilized

    # Timeline
    germinated_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime = field(default_factory=_utc_now)
    total_epochs: int = 0
    epochs_per_stage: dict[str, int] = field(default_factory=dict)

    # Performance
    accuracy_at_germination: float = 0.0
    accuracy_at_completion: float = 0.0
    total_improvement: float = 0.0
    best_accuracy_achieved: float = 0.0

    # Context at germination
    signals_at_germination: TrainingSignals | None = None

    # Commands received
    commands_received: list[str] = field(default_factory=list)  # command_ids

    # Failure info (if pruned)
    failure_reason: str = ""
    failure_stage: SeedStage = SeedStage.UNKNOWN
