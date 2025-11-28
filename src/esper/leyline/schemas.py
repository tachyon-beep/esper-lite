"""Leyline Schemas - Command and blueprint specifications.

Defines the structure of commands issued by controllers
and specifications for seed blueprints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from esper.leyline.stages import CommandType, RiskLevel, SeedStage


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


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
