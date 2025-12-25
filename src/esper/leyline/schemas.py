"""Leyline Schemas - Gate specifications and seed operations.

Defines quality gate types and seed lifecycle operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto

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


