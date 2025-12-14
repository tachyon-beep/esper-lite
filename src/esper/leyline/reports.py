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
from typing import Any
from uuid import uuid4

from esper.leyline.stages import SeedStage
from esper.leyline.signals import TrainingSignals


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

    # Gradient activity (parameter-normalized) for G2 gate
    seed_gradient_norm_ratio: float = 0.0

    # Parameter counts for normalization/analytics
    seed_param_count: int = 0
    host_param_count: int = 0

    # Health
    isolation_violations: int = 0
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

    # Metrics
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

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

    Collected when a seed reaches a terminal state (FOSSILIZED or CULLED).
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

    # Failure info (if culled)
    failure_reason: str = ""
    failure_stage: SeedStage = SeedStage.UNKNOWN
