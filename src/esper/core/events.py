"""Leyline-aligned contracts used across subsystems.

Schemas are placeholders that should mirror the protobuf definitions stored in the
Leyline contract module (see `docs/project/backlog.md` TKT-002).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TrainingPhase(str, Enum):
    """Enumerates the high-level training loop phases (see `old/01-tolaria.md`)."""

    INIT = "init"
    EPOCH = "epoch"
    VALIDATION = "validation"
    ROLLBACK = "rollback"
    COMPLETE = "complete"


class SeedLifecycleStage(str, Enum):
    """Represents Kasmina's eleven-stage lifecycle (simplified placeholder)."""

    DORMANT = "dormant"
    REGISTERED = "registered"
    GERMINATING = "germinating"
    ACTIVE = "active"
    OBSERVING = "observing"
    DEGRADED = "degraded"
    ISOLATED = "isolated"
    ROLLING_BACK = "rolling_back"
    RETIRED = "retired"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class SystemStatePacket(BaseModel):
    """Telemetry emitted by Tolaria at each epoch boundary."""

    run_id: str = Field(..., description="Unique training run identifier")
    epoch_index: int = Field(..., ge=0, description="Zero-based epoch index")
    phase: TrainingPhase = Field(..., description="Current training loop phase")
    host_metrics: dict[str, float] = Field(default_factory=dict)
    seed_snapshots: dict[str, SeedLifecycleStage] = Field(default_factory=dict)
    emitted_at: datetime = Field(default_factory=datetime.utcnow)


class AdaptationDirective(str, Enum):
    """Possible directive types returned by Tamiyo."""

    NO_OP = "no_op"
    GRAFT_SEED = "graft_seed"
    RETIRE_SEED = "retire_seed"
    ADJUST_HYPERPARAMETERS = "adjust_hyperparameters"
    REQUEST_BLUEPRINT = "request_blueprint"


class AdaptationCommand(BaseModel):
    """Tamiyo adaptation decision sent to Kasmina."""

    run_id: str
    epoch_index: int
    directive: AdaptationDirective
    payload: dict[str, float | int | str] = Field(default_factory=dict)
    conservative_mode: bool = Field(default=False)
    issued_at: datetime = Field(default_factory=datetime.utcnow)


class FieldReportOutcome(str, Enum):
    """Outcome label for Tamiyo field reports."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    ABORTED = "aborted"
    RISK_REJECTED = "risk_rejected"


class FieldReport(BaseModel):
    """Tamiyo field report summarising adaptation impact."""

    run_id: str
    command: AdaptationCommand
    outcome: FieldReportOutcome
    metrics_delta: dict[str, float]
    notes: str | None = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)


__all__ = [
    "AdaptationCommand",
    "AdaptationDirective",
    "FieldReport",
    "FieldReportOutcome",
    "SeedLifecycleStage",
    "SystemStatePacket",
    "TrainingPhase",
]
