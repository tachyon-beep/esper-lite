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

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.stages import SeedStage

# Feature normalization constants for RL observation space
# These define the expected ranges for seed telemetry values
_GRADIENT_NORM_MAX: float = 10.0  # 99th percentile typical gradient norm
_EPOCHS_IN_STAGE_MAX: int = 50  # Typical max epochs in single stage
_ACCURACY_DELTA_SCALE: float = 10.0  # Scale factor for accuracy deltas
_ALPHA_MODE_MAX: int = max(mode.value for mode in AlphaMode)
_ALPHA_ALGO_MIN: int = min(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_MAX: int = max(algo.value for algo in AlphaAlgorithm)


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
    SEED_PRUNED = auto()

    # Training events
    EPOCH_COMPLETED = auto()  # Per-env epoch (has env_id in data)
    BATCH_EPOCH_COMPLETED = auto()  # Batch-level epoch summary + progress (commit barrier)
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
    group_id: str = "default"  # A/B testing group identifier (e.g., "A", "B")

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
    stage: int = 1  # SeedStage enum value (1-10); feature scaling clamps >=10 to 1.0
    alpha: float = 0.0  # blending weight (0-1)

    # Alpha controller context
    alpha_target: float = 0.0
    alpha_mode: int = 0
    alpha_steps_total: int = 0
    alpha_steps_done: int = 0
    time_to_target: int = 0
    alpha_velocity: float = 0.0
    alpha_algorithm: int = AlphaAlgorithm.ADD.value

    # Temporal context
    epoch: int = 0
    max_epochs: int = 25

    # Timestamp for staleness detection
    captured_at: datetime = field(default_factory=_utc_now)

    # Tempo telemetry
    blend_tempo_epochs: int = 5
    blending_velocity: float = 0.0  # d(alpha) / d(epoch)

    def to_features(self) -> list[float]:
        """Convert to 26-dim feature vector for RL policies.

        Features (schema v1, one-hot stage encoding):
        - [0-9]: Stage one-hot (10 dims) - categorical stage representation
        - [10]: gradient_norm (normalized to [0, 1])
        - [11]: gradient_health ([0, 1])
        - [12]: has_vanishing (0 or 1)
        - [13]: has_exploding (0 or 1)
        - [14]: epochs_in_stage (normalized)
        - [15]: accuracy (normalized to [0, 1])
        - [16]: accuracy_delta (clamped to [-1, 1])
        - [17]: alpha ([0, 1])
        - [18]: epoch/max_epochs (temporal position)
        - [19]: alpha_target ([0, 1])
        - [20]: alpha_mode (normalized)
        - [21]: alpha_steps_total (normalized)
        - [22]: alpha_steps_done (normalized)
        - [23]: time_to_target (normalized)
        - [24]: alpha_velocity (clamped to [-1, 1])
        - [25]: alpha_algorithm (normalized)
        """
        from esper.leyline.stage_schema import stage_to_one_hot, VALID_STAGE_VALUES

        steps_den = max(self.max_epochs, 1)
        alpha_algo_range = max(_ALPHA_ALGO_MAX - _ALPHA_ALGO_MIN, 1)

        # Stage one-hot (10 dims) - handles gap at value 5
        if self.stage in VALID_STAGE_VALUES:
            stage_one_hot = stage_to_one_hot(self.stage)
        else:
            # Fallback for invalid stage: all zeros (should not happen after Phase 0 validation)
            from esper.leyline.stage_schema import NUM_STAGES
            stage_one_hot = [0.0] * NUM_STAGES

        return stage_one_hot + [
            min(self.gradient_norm, _GRADIENT_NORM_MAX) / _GRADIENT_NORM_MAX,
            self.gradient_health,
            float(self.has_vanishing),
            float(self.has_exploding),
            min(self.epochs_in_stage, _EPOCHS_IN_STAGE_MAX) / _EPOCHS_IN_STAGE_MAX,
            self.accuracy / 100.0,
            max(-1.0, min(1.0, self.accuracy_delta / _ACCURACY_DELTA_SCALE)),
            self.alpha,
            self.epoch / max(self.max_epochs, 1),  # temporal position
            self.alpha_target,
            self.alpha_mode / max(_ALPHA_MODE_MAX, 1),
            min(self.alpha_steps_total, steps_den) / steps_den,
            min(self.alpha_steps_done, steps_den) / steps_den,
            min(self.time_to_target, steps_den) / steps_den,
            max(-1.0, min(1.0, self.alpha_velocity)),
            (self.alpha_algorithm - _ALPHA_ALGO_MIN) / alpha_algo_range,
        ]

    @classmethod
    def feature_dim(cls) -> int:
        """Return current feature vector dimension.

        Schema v1: 10 (stage one-hot) + 16 (other features) = 26
        """
        from esper.leyline.stage_schema import NUM_STAGES
        return NUM_STAGES + 16  # 10 + 16 = 26

    def to_dict(self) -> dict:
        """Convert to primitive dict for serialization."""
        from esper.leyline.stage_schema import STAGE_SCHEMA_VERSION
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
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
            "alpha_target": self.alpha_target,
            "alpha_mode": self.alpha_mode,
            "alpha_steps_total": self.alpha_steps_total,
            "alpha_steps_done": self.alpha_steps_done,
            "time_to_target": self.time_to_target,
            "alpha_velocity": self.alpha_velocity,
            "alpha_algorithm": self.alpha_algorithm,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "blend_tempo_epochs": self.blend_tempo_epochs,
            "blending_velocity": self.blending_velocity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedTelemetry":
        """Reconstruct from primitive dict.

        Raises KeyError/ValueError if required fields are missing (no silent defaults).
        """
        from datetime import datetime
        from esper.leyline.stage_schema import STAGE_SCHEMA_VERSION

        # Validate schema version if present (fail-fast on mismatch)
        schema_version = data.get("schema_version")
        if schema_version is not None and schema_version != STAGE_SCHEMA_VERSION:
            raise ValueError(
                f"SeedTelemetry schema version mismatch: "
                f"expected {STAGE_SCHEMA_VERSION}, got {schema_version}. "
                f"Telemetry from incompatible version cannot be loaded."
            )

        # Required: stage - validate type and enum membership
        stage_raw = data["stage"]  # KeyError if missing
        if not isinstance(stage_raw, int) or isinstance(stage_raw, bool):
            raise ValueError(
                f"SeedTelemetry.stage must be an int SeedStage.value, got {stage_raw!r}"
            )
        try:
            stage = SeedStage(stage_raw).value
        except ValueError as e:
            raise ValueError(
                f"SeedTelemetry.stage must be a valid SeedStage.value, got {stage_raw!r}"
            ) from e

        # Required: alpha_mode - validate type and enum membership
        alpha_mode_raw = data["alpha_mode"]  # KeyError if missing
        if not isinstance(alpha_mode_raw, int) or isinstance(alpha_mode_raw, bool):
            raise ValueError(
                f"SeedTelemetry.alpha_mode must be an int AlphaMode.value, got {alpha_mode_raw!r}"
            )
        try:
            alpha_mode = AlphaMode(alpha_mode_raw).value
        except ValueError as e:
            raise ValueError(
                f"SeedTelemetry.alpha_mode must be a valid AlphaMode.value, got {alpha_mode_raw!r}"
            ) from e

        # Required: alpha_algorithm - validate type and enum membership
        alpha_algorithm_raw = data["alpha_algorithm"]  # KeyError if missing
        if not isinstance(alpha_algorithm_raw, int) or isinstance(alpha_algorithm_raw, bool):
            raise ValueError(
                f"SeedTelemetry.alpha_algorithm must be an int AlphaAlgorithm.value, got {alpha_algorithm_raw!r}"
            )
        try:
            alpha_algorithm = AlphaAlgorithm(alpha_algorithm_raw).value
        except ValueError as e:
            raise ValueError(
                f"SeedTelemetry.alpha_algorithm must be a valid AlphaAlgorithm.value, got {alpha_algorithm_raw!r}"
            ) from e

        return cls(
            # Required fields - KeyError if missing
            seed_id=data["seed_id"],
            blueprint_id=data["blueprint_id"],
            layer_id=data["layer_id"],
            gradient_norm=data["gradient_norm"],
            gradient_health=data["gradient_health"],
            has_vanishing=data["has_vanishing"],
            has_exploding=data["has_exploding"],
            accuracy=data["accuracy"],
            accuracy_delta=data["accuracy_delta"],
            epochs_in_stage=data["epochs_in_stage"],
            stage=stage,
            alpha=data["alpha"],
            alpha_target=data["alpha_target"],
            alpha_mode=alpha_mode,
            alpha_steps_total=data["alpha_steps_total"],
            alpha_steps_done=data["alpha_steps_done"],
            time_to_target=data["time_to_target"],
            alpha_velocity=data["alpha_velocity"],
            alpha_algorithm=alpha_algorithm,
            epoch=data["epoch"],
            max_epochs=data["max_epochs"],
            captured_at=datetime.fromisoformat(data["captured_at"]),
            blend_tempo_epochs=data["blend_tempo_epochs"],
            blending_velocity=data["blending_velocity"],
        )
