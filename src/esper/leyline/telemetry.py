"""Leyline Telemetry Contracts - Event definitions for system monitoring.

These are CONTRACTS only - the actual emission and handling
is done by domain-specific modules and Nissa.

Each domain emits events using these contracts:
- Kasmina: Seed lifecycle events
- Tamiyo: Decision events
- Simic: Training progress events
- Nissa: System health events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import uuid4


# Feature normalization constants for RL observation space
# These define the expected ranges for seed telemetry values
_GRADIENT_NORM_MAX: float = 10.0  # 99th percentile typical gradient norm
_EPOCHS_IN_STAGE_MAX: int = 50  # Typical max epochs in single stage
_ACCURACY_DELTA_SCALE: float = 10.0  # Scale factor for accuracy deltas


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TelemetryEventType(Enum):
    """Types of telemetry events."""

    # Lifecycle events
    SEED_GERMINATED = auto()
    SEED_STAGE_CHANGED = auto()
    SEED_GATE_EVALUATED = auto()
    SEED_FOSSILIZED = auto()
    SEED_CULLED = auto()

    # Training events
    EPOCH_COMPLETED = auto()
    PLATEAU_DETECTED = auto()
    DEGRADATION_DETECTED = auto()
    IMPROVEMENT_DETECTED = auto()
    TAMIYO_INITIATED = auto()  # Host stabilized, germination now allowed

    # Health events
    ISOLATION_VIOLATION = auto()
    GRADIENT_ANOMALY = auto()
    PERFORMANCE_DEGRADATION = auto()

    # === NEW: PPO Training Events (Ops Normal) ===
    PPO_UPDATE_COMPLETED = auto()
    MEMORY_WARNING = auto()
    REWARD_HACKING_SUSPECTED = auto()
    REWARD_COMPUTED = auto()  # Per-step reward breakdown for debugging

    # === NEW: Debug Events (triggered by anomalies) ===
    RATIO_EXPLOSION_DETECTED = auto()
    RATIO_COLLAPSE_DETECTED = auto()
    VALUE_COLLAPSE_DETECTED = auto()
    GRADIENT_PATHOLOGY_DETECTED = auto()
    NUMERICAL_INSTABILITY_DETECTED = auto()

    # === Governor Events (Tolaria) ===
    GOVERNOR_PANIC = auto()           # Vital signs check failed
    GOVERNOR_ROLLBACK = auto()        # Emergency rollback executed
    GOVERNOR_SNAPSHOT = auto()        # LKG checkpoint saved

    # === Training Progress Events ===
    BATCH_COMPLETED = auto()          # PPO batch finished
    TRAINING_STARTED = auto()         # Training run initialized
    CHECKPOINT_SAVED = auto()         # Model checkpoint saved
    CHECKPOINT_LOADED = auto()        # Model checkpoint restored

    # === Counterfactual Attribution Events ===
    COUNTERFACTUAL_COMPUTED = auto()  # Per-slot counterfactual contribution measured
    COUNTERFACTUAL_MATRIX_COMPUTED = auto()  # Full factorial matrix for env

    # === Analytics Events ===
    ANALYTICS_SNAPSHOT = auto()       # Full state snapshot for dashboard sync


@dataclass
class TelemetryEvent:
    """A telemetry event for observability."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: TelemetryEventType = TelemetryEventType.EPOCH_COMPLETED
    timestamp: datetime = field(default_factory=_utc_now)

    # Context
    seed_id: str | None = None
    slot_id: str | None = None
    epoch: int | None = None

    # Event data
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    # Severity
    severity: str = "info"  # debug, info, warning, error, critical


@dataclass(frozen=True)
class PerformanceBudgets:
    """Performance budget constants."""

    # Timing budgets (milliseconds)
    epoch_budget_ms: float = 18.0
    blending_budget_ms: float = 5.0
    gate_check_budget_ms: float = 2.0

    # Memory budgets (GB)
    seed_memory_budget_gb: float = 2.0
    total_memory_budget_gb: float = 12.0

    # Serialization budgets
    max_message_size_bytes: int = 280
    serialization_budget_us: float = 80.0


# Default budgets
DEFAULT_BUDGETS = PerformanceBudgets()


# =============================================================================
# Seed Telemetry Snapshot
# =============================================================================

@dataclass(slots=True)
class SeedTelemetry:
    """Per-seed telemetry snapshot - the seed's 'local picture'.

    Contract between seed implementations (Kasmina/Simic) and
    decision-makers (Tamiyo). Designed for:
    - Single seed (current): one instance
    - Multi-seed (future): collection managed by registry
    - Hierarchical (stretch): tactical aggregates for strategic

    Note: Uses slots=True for memory efficiency in multi-seed scenarios.
    """

    seed_id: str
    blueprint_id: str = ""
    layer_id: str = ""

    # Health signals (lightweight, always collected)
    gradient_norm: float = 0.0
    gradient_health: float = 1.0  # 0-1, higher is healthier
    has_vanishing: bool = False
    has_exploding: bool = False

    # Progress signals
    accuracy: float = 0.0  # percentage (0-100)
    accuracy_delta: float = 0.0  # positive = improving
    epochs_in_stage: int = 0

    # Stage context
    stage: int = 1  # SeedStage enum value (1-7)
    alpha: float = 0.0  # blending weight (0-1)

    # Temporal context
    epoch: int = 0
    max_epochs: int = 25

    # Timestamp for staleness detection
    captured_at: datetime = field(default_factory=_utc_now)

    # Tempo telemetry
    blend_tempo_epochs: int = 5
    blending_velocity: float = 0.0  # d(alpha) / d(epoch)

    def to_features(self) -> list[float]:
        """Convert to 10-dim feature vector for RL policies.

        All features normalized to approximately [0, 1] range.
        Uses module constants for normalization bounds.
        """
        return [
            min(self.gradient_norm, _GRADIENT_NORM_MAX) / _GRADIENT_NORM_MAX,
            self.gradient_health,
            float(self.has_vanishing),
            float(self.has_exploding),
            min(self.epochs_in_stage, _EPOCHS_IN_STAGE_MAX) / _EPOCHS_IN_STAGE_MAX,
            self.accuracy / 100.0,
            max(-1.0, min(1.0, self.accuracy_delta / _ACCURACY_DELTA_SCALE)),
            min((self.stage - 1) / 6.0, 1.0),  # stages 1-7 -> [0, 1], clamp overflow
            self.alpha,
            self.epoch / max(self.max_epochs, 1),  # temporal position
        ]

    @classmethod
    def feature_dim(cls) -> int:
        """Return current feature vector dimension."""
        return 10

    def to_dict(self) -> dict:
        """Convert to primitive dict for serialization."""
        return {
            "seed_id": self.seed_id,
            "blueprint_id": self.blueprint_id,
            "layer_id": self.layer_id,
            "gradient_norm": self.gradient_norm,
            "gradient_health": self.gradient_health,
            "has_vanishing": self.has_vanishing,
            "has_exploding": self.has_exploding,
            "accuracy": self.accuracy,
            "accuracy_delta": self.accuracy_delta,
            "epochs_in_stage": self.epochs_in_stage,
            "stage": self.stage,
            "alpha": self.alpha,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "blend_tempo_epochs": self.blend_tempo_epochs,
            "blending_velocity": self.blending_velocity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedTelemetry":
        """Reconstruct from primitive dict."""
        from datetime import datetime

        return cls(
            seed_id=data["seed_id"],
            blueprint_id=data.get("blueprint_id", ""),
            layer_id=data.get("layer_id", ""),
            gradient_norm=data.get("gradient_norm", 0.0),
            gradient_health=data.get("gradient_health", 1.0),
            has_vanishing=data.get("has_vanishing", False),
            has_exploding=data.get("has_exploding", False),
            accuracy=data.get("accuracy", 0.0),
            accuracy_delta=data.get("accuracy_delta", 0.0),
            epochs_in_stage=data.get("epochs_in_stage", 0),
            stage=data.get("stage", 1),
            alpha=data.get("alpha", 0.0),
            epoch=data.get("epoch", 0),
            max_epochs=data.get("max_epochs", 25),
            captured_at=datetime.fromisoformat(data["captured_at"]) if data.get("captured_at") else _utc_now(),
            blend_tempo_epochs=data.get("blend_tempo_epochs", 5),
            blending_velocity=data.get("blending_velocity", 0.0),
        )
