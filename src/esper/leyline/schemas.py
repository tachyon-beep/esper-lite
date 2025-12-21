"""Leyline Schemas - Blueprint and gate specifications.

Defines specifications for seed blueprints and quality gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Any, Protocol, runtime_checkable

from esper.leyline.stages import SeedStage


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class SeedOperation(Enum):
    """Operations that can be performed on a seed."""
    GERMINATE = auto()
    START_TRAINING = auto()
    START_BLENDING = auto()
    START_HOLDING = auto()
    FOSSILIZE = auto()
    PRUNE = auto()
    EMBARGO = auto()
    RESET = auto()


# Mapping from operation to target stage
OPERATION_TARGET_STAGE: dict[SeedOperation, SeedStage] = {
    SeedOperation.GERMINATE: SeedStage.GERMINATED,
    SeedOperation.START_TRAINING: SeedStage.TRAINING,
    SeedOperation.START_BLENDING: SeedStage.BLENDING,
    SeedOperation.START_HOLDING: SeedStage.HOLDING,
    SeedOperation.FOSSILIZE: SeedStage.FOSSILIZED,
    SeedOperation.PRUNE: SeedStage.PRUNED,
    SeedOperation.EMBARGO: SeedStage.EMBARGOED,
    SeedOperation.RESET: SeedStage.RESETTING,
}


class GateLevel(IntEnum):
    """Quality gate levels for stage transitions."""
    G0 = 0  # Basic sanity (DORMANT → GERMINATED)
    G1 = 1  # Training readiness (GERMINATED → TRAINING)
    G2 = 2  # Blending readiness (TRAINING → BLENDING)
    G3 = 3  # Holding readiness (BLENDING → HOLDING)
    # Value 4 intentionally skipped (was G4/SHADOWING gate, removed)
    G5 = 5  # Fossilization readiness (HOLDING → FOSSILIZED)


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
