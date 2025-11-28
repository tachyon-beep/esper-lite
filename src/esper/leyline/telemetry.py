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


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


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
