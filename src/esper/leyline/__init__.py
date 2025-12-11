"""Leyline - The invisible substrate of Esper.

Leyline defines the data contracts that flow between all Esper components.
Import from here for the public API.

Example:
    from esper.leyline import SimicAction, SeedStage, TrainingSignals
"""

# Version
LEYLINE_VERSION = "0.2.0"

# =============================================================================
# Lifecycle Constants (shared across simic modules)
# =============================================================================

# Minimum seed age before CULL is allowed (structural fix for uninformed culls)
# 10 epochs gives enough signal to evaluate seed quality
MIN_CULL_AGE = 10

# Epochs needed for confident seed quality assessment
FULL_EVALUATION_AGE = 10

# Minimum epochs in PROBATIONARY to earn full fossilize bonus
MIN_PROBATION_EPOCHS = 5

# Plateau gating: prevent germination until training has plateaued
MIN_GERMINATE_EPOCH = 5        # Let host get easy wins first
MIN_PLATEAU_TO_GERMINATE = 3   # Consecutive epochs with <0.5% improvement

# Seed limits (None = unlimited)
DEFAULT_MAX_SEEDS = None           # Global limit across all slots
DEFAULT_MAX_SEEDS_PER_SLOT = None  # Per-slot limit

# Actions
from esper.leyline.actions import (
    Action,
    SimicAction,  # SimicAction is alias
    build_action_enum,
    get_blueprint_from_action,
    is_germinate_action,
)

# Stages and transitions
from esper.leyline.stages import (
    SeedStage,
    CommandType,
    RiskLevel,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
)

# Signals (hot path)
from esper.leyline.signals import (
    TensorSchema,
    TENSOR_SCHEMA_SIZE,
    FastTrainingSignals,
    TrainingMetrics,
    TrainingSignals,
)

# Schemas and specifications
from esper.leyline.schemas import (
    SeedOperation,
    OPERATION_TARGET_STAGE,
    AdaptationCommand,
    GateLevel,
    GateResult,
    BlueprintProtocol,
    BlueprintSpec,
)

# Reports
from esper.leyline.reports import (
    SeedMetrics,
    SeedStateReport,
    FieldReport,
)

# Telemetry contracts
from esper.leyline.telemetry import (
    TelemetryEventType,
    TelemetryEvent,
    PerformanceBudgets,
    DEFAULT_BUDGETS,
    SeedTelemetry,
)

__all__ = [
    # Version
    "LEYLINE_VERSION",

    # Lifecycle constants
    "MIN_CULL_AGE",
    "FULL_EVALUATION_AGE",
    "MIN_PROBATION_EPOCHS",
    "MIN_GERMINATE_EPOCH",
    "MIN_PLATEAU_TO_GERMINATE",
    "DEFAULT_MAX_SEEDS",
    "DEFAULT_MAX_SEEDS_PER_SLOT",

    # Actions
    "Action",
    "SimicAction",  # deprecated alias
    "build_action_enum",
    "get_blueprint_from_action",
    "is_germinate_action",

    # Stages
    "SeedStage",
    "CommandType",
    "RiskLevel",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "is_terminal_stage",
    "is_active_stage",
    "is_failure_stage",

    # Signals
    "TensorSchema",
    "TENSOR_SCHEMA_SIZE",
    "FastTrainingSignals",
    "TrainingMetrics",
    "TrainingSignals",

    # Schemas
    "SeedOperation",
    "OPERATION_TARGET_STAGE",
    "AdaptationCommand",
    "GateLevel",
    "GateResult",
    "BlueprintProtocol",
    "BlueprintSpec",

    # Reports
    "SeedMetrics",
    "SeedStateReport",
    "FieldReport",

    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
    "SeedTelemetry",
]
