"""Leyline: Shared Contracts for Esper-Lite

This module defines the data contracts (interfaces) between Esper components.
These are pure type definitions - no behavior, just the shapes of data.

Contracts defined here:
- Seed lifecycle stages and valid transitions
- Commands from Tamiyo to Kasmina (AdaptationCommand)
- Signals from training to Tamiyo (TrainingSignals)
- Reports from Kasmina about seeds (SeedStateReport)
- Field reports for Simic learning (FieldReport)
- Blueprint specifications

Design principles:
- Contracts are data, not behavior
- All fields have explicit types
- Enums for closed sets of values
- Dataclasses for structured data
- Protocols for capability contracts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Any, NamedTuple, Protocol, runtime_checkable
from uuid import uuid4


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


# =============================================================================
# Data Plane: Fast Path for PPO (High-Frequency, No GC Pressure)
# =============================================================================

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


# =============================================================================
# Version
# =============================================================================

LEYLINE_VERSION = "0.2.0"


# =============================================================================
# Seed Lifecycle Stages
# =============================================================================

class SeedStage(IntEnum):
    """Lifecycle stages for a seed.

    The full lifecycle represents a trust escalation model:

    DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► SHADOWING
                    │              │            │            │
                    ▼              ▼            ▼            ▼
                 CULLED ◄──────────────────────────────────────
                    │
                    ▼
               EMBARGOED ──► RESETTING ──► DORMANT (slot recycled)

    PROBATIONARY ──► FOSSILIZED (terminal success)
         │
         ▼
      CULLED (failure path)

    Stages explained:
    - DORMANT: Empty slot, waiting for a seed
    - GERMINATED: Seed attached, sanity checks passed, ready to train
    - TRAINING: Isolated training with gradient isolation from host
    - BLENDING: Alpha-managed grafting, gradually blending into host
    - SHADOWING: Running in shadow mode, comparing outputs without affecting host
    - PROBATIONARY: Final validation period before permanent integration
    - FOSSILIZED: Permanently integrated into the model (terminal success)
    - CULLED: Removed due to failure or poor performance
    - EMBARGOED: Cooldown period after culling to prevent thrashing
    - RESETTING: Cleanup before slot can be reused
    """

    UNKNOWN = 0
    DORMANT = 1
    GERMINATED = 2
    TRAINING = 3
    BLENDING = 4
    SHADOWING = 5
    PROBATIONARY = 6
    FOSSILIZED = 7      # Terminal state (success)
    CULLED = 8          # Failure state
    EMBARGOED = 9       # Post-cull cooldown
    RESETTING = 10      # Cleanup before reuse


# Valid transitions between stages
VALID_TRANSITIONS: dict[SeedStage, tuple[SeedStage, ...]] = {
    SeedStage.UNKNOWN: (SeedStage.DORMANT,),
    SeedStage.DORMANT: (SeedStage.GERMINATED,),
    SeedStage.GERMINATED: (SeedStage.TRAINING, SeedStage.CULLED),
    SeedStage.TRAINING: (SeedStage.BLENDING, SeedStage.CULLED),
    SeedStage.BLENDING: (SeedStage.SHADOWING, SeedStage.CULLED),
    SeedStage.SHADOWING: (SeedStage.PROBATIONARY, SeedStage.CULLED),
    SeedStage.PROBATIONARY: (SeedStage.FOSSILIZED, SeedStage.CULLED),
    SeedStage.FOSSILIZED: (),  # Terminal - no transitions out
    SeedStage.CULLED: (SeedStage.EMBARGOED,),
    SeedStage.EMBARGOED: (SeedStage.RESETTING,),
    SeedStage.RESETTING: (SeedStage.DORMANT,),
}


def is_valid_transition(from_stage: SeedStage, to_stage: SeedStage) -> bool:
    """Check if a stage transition is valid."""
    return to_stage in VALID_TRANSITIONS.get(from_stage, ())


def is_terminal_stage(stage: SeedStage) -> bool:
    """Check if a stage is terminal (no further transitions)."""
    return stage in (SeedStage.FOSSILIZED,)


def is_active_stage(stage: SeedStage) -> bool:
    """Check if a stage represents an active seed (contributing to forward pass)."""
    return stage in (
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.SHADOWING,
        SeedStage.PROBATIONARY,
        SeedStage.FOSSILIZED,
    )


def is_failure_stage(stage: SeedStage) -> bool:
    """Check if a stage represents a failed seed."""
    return stage in (SeedStage.CULLED, SeedStage.EMBARGOED, SeedStage.RESETTING)


# =============================================================================
# Command Types (Tamiyo → Kasmina)
# =============================================================================

class CommandType(Enum):
    """Types of commands Tamiyo can issue to Kasmina."""

    # Lifecycle commands
    GERMINATE = auto()          # Create a new seed
    ADVANCE_STAGE = auto()      # Move seed to next stage
    CULL = auto()               # Kill a seed

    # Parameter commands
    SET_ALPHA = auto()          # Set blending alpha directly
    SET_LEARNING_RATE = auto()  # Adjust seed learning rate

    # Control commands
    PAUSE_SEED = auto()         # Temporarily pause seed training
    RESUME_SEED = auto()        # Resume paused seed

    # Query commands
    REQUEST_STATE = auto()      # Request current seed state


class RiskLevel(IntEnum):
    """Risk assessment levels for commands."""
    GREEN = 1       # Safe, routine operation
    YELLOW = 2      # Caution, monitor closely
    ORANGE = 3      # Elevated risk, may need rollback
    RED = 4         # High risk, prepare for intervention
    CRITICAL = 5    # Emergency, immediate attention required


@dataclass(frozen=True)
class AdaptationCommand:
    """Command from Tamiyo to Kasmina.

    This is the primary contract for Tamiyo → Kasmina communication.
    Commands are immutable once created.
    """

    # Identity
    command_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=_utc_now)

    # Command specification
    command_type: CommandType = CommandType.GERMINATE
    target_seed_id: str | None = None
    target_slot_id: str | None = None

    # Parameters (command-specific)
    blueprint_id: str | None = None
    target_stage: SeedStage | None = None
    alpha_value: float | None = None
    learning_rate: float | None = None

    # Metadata
    reason: str = ""
    confidence: float = 1.0
    risk_level: RiskLevel = RiskLevel.GREEN

    # Annotations (extensible key-value pairs)
    annotations: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Seed Operations (for backwards compatibility)
# =============================================================================

class SeedOperation(Enum):
    """Operations that can be performed on a seed."""
    GERMINATE = auto()
    START_TRAINING = auto()
    START_BLENDING = auto()
    START_SHADOWING = auto()
    START_PROBATION = auto()
    FOSSILIZE = auto()
    CULL = auto()
    EMBARGO = auto()
    RESET = auto()


# Mapping from operation to target stage
OPERATION_TARGET_STAGE: dict[SeedOperation, SeedStage] = {
    SeedOperation.GERMINATE: SeedStage.GERMINATED,
    SeedOperation.START_TRAINING: SeedStage.TRAINING,
    SeedOperation.START_BLENDING: SeedStage.BLENDING,
    SeedOperation.START_SHADOWING: SeedStage.SHADOWING,
    SeedOperation.START_PROBATION: SeedStage.PROBATIONARY,
    SeedOperation.FOSSILIZE: SeedStage.FOSSILIZED,
    SeedOperation.CULL: SeedStage.CULLED,
    SeedOperation.EMBARGO: SeedStage.EMBARGOED,
    SeedOperation.RESET: SeedStage.RESETTING,
}


# =============================================================================
# Training Signals (Training Loop → Tamiyo)
# =============================================================================

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
        seed_stage: int = 0,
        seed_epochs_in_stage: int = 0,
        seed_alpha: float = 0.0,
        seed_improvement: float = 0.0,
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
            best_val_loss=self.metrics.best_val_loss if self.metrics.best_val_loss != float('inf') else 10.0,
            loss_history_5=tuple(loss_hist),
            accuracy_history_5=tuple(acc_hist),
            has_active_seed=1 if self.active_seeds else 0,
            seed_stage=seed_stage,
            seed_epochs_in_stage=seed_epochs_in_stage,
            seed_alpha=seed_alpha,
            seed_improvement=seed_improvement,
            available_slots=self.available_slots,
        )


# =============================================================================
# Seed State Report (Kasmina → Tamiyo)
# =============================================================================

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


# =============================================================================
# Field Report (Kasmina → Simic)
# =============================================================================

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


# =============================================================================
# Blueprint Specification
# =============================================================================

@runtime_checkable
class BlueprintProtocol(Protocol):
    """Protocol that all blueprints must satisfy."""

    @property
    def blueprint_id(self) -> str:
        """Unique identifier for this blueprint type."""
        ...

    @property
    def required_channels(self) -> int | None:
        """Number of input channels required, or None if flexible."""
        ...

    def create_module(self, in_channels: int, **kwargs) -> Any:
        """Create a seed module instance."""
        ...


@dataclass(frozen=True)
class BlueprintSpec:
    """Specification for a blueprint in the catalog."""

    blueprint_id: str
    name: str
    description: str = ""

    # Requirements
    min_channels: int = 1
    max_channels: int | None = None

    # Characteristics
    parameter_count_estimate: int = 0  # Rough parameter count
    compute_cost: float = 1.0  # Relative compute cost (1.0 = baseline)
    memory_cost: float = 1.0  # Relative memory cost

    # Recommended usage
    recommended_training_epochs: int = 5
    recommended_blending_epochs: int = 5

    # Tags for categorization
    tags: tuple[str, ...] = ()


# =============================================================================
# Gate Specifications (Quality Gates)
# =============================================================================

class GateLevel(IntEnum):
    """Quality gate levels for stage transitions."""
    G0 = 0  # Basic sanity (DORMANT → GERMINATED)
    G1 = 1  # Training readiness (GERMINATED → TRAINING)
    G2 = 2  # Blending readiness (TRAINING → BLENDING)
    G3 = 3  # Shadow readiness (BLENDING → SHADOWING)
    G4 = 4  # Probation readiness (SHADOWING → PROBATIONARY)
    G5 = 5  # Fossilization readiness (PROBATIONARY → FOSSILIZED)


@dataclass
class GateResult:
    """Result of a quality gate check."""

    gate: GateLevel
    passed: bool
    score: float = 0.0  # 0-1 confidence score

    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)

    message: str = ""
    timestamp: datetime = field(default_factory=_utc_now)


# =============================================================================
# Telemetry
# =============================================================================

class TelemetryEventType(Enum):
    """Types of telemetry events."""

    # Lifecycle events
    SEED_GERMINATED = auto()
    SEED_STAGE_CHANGED = auto()
    SEED_FOSSILIZED = auto()
    SEED_CULLED = auto()

    # Training events
    EPOCH_COMPLETED = auto()
    PLATEAU_DETECTED = auto()
    IMPROVEMENT_DETECTED = auto()

    # Health events
    ISOLATION_VIOLATION = auto()
    GRADIENT_ANOMALY = auto()
    PERFORMANCE_DEGRADATION = auto()

    # Command events
    COMMAND_ISSUED = auto()
    COMMAND_EXECUTED = auto()
    COMMAND_FAILED = auto()


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


# =============================================================================
# Performance Budgets
# =============================================================================

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
# Exports
# =============================================================================

__all__ = [
    # Version
    "LEYLINE_VERSION",

    # Data Plane (Fast Path for PPO)
    "TensorSchema",
    "TENSOR_SCHEMA_SIZE",
    "FastTrainingSignals",

    # Lifecycle
    "SeedStage",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "is_terminal_stage",
    "is_active_stage",
    "is_failure_stage",

    # Commands
    "CommandType",
    "RiskLevel",
    "AdaptationCommand",
    "SeedOperation",
    "OPERATION_TARGET_STAGE",

    # Signals
    "TrainingMetrics",
    "TrainingSignals",

    # Seed state
    "SeedMetrics",
    "SeedStateReport",

    # Field reports
    "FieldReport",

    # Blueprints
    "BlueprintProtocol",
    "BlueprintSpec",

    # Gates
    "GateLevel",
    "GateResult",

    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",

    # Budgets
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
]
