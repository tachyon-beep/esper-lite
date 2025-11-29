"""Leyline - The invisible substrate of Esper.

Leyline defines the data contracts that flow between all Esper components.
Import from here for the public API.

Example:
    from esper.leyline import SimicAction, SeedStage, TrainingSignals
"""

# Version
LEYLINE_VERSION = "0.2.0"

# Actions
from esper.leyline.actions import Action, SimicAction  # SimicAction is alias

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

    # Actions
    "Action",
    "SimicAction",  # deprecated alias

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
