"""Shared contracts and types for Karn subsystem.

Defines local protocols and data structures to decouple Karn from external
subsystems like Leyline/Nissa where appropriate (dependency inversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol


class TelemetryEventLike(Protocol):
    """Minimal TelemetryEvent-like Protocol for Karn internal use.
    
    Decouples Karn from esper.leyline.telemetry.TelemetryEvent.
    """
    event_type: Any  # Can be enum or string
    timestamp: datetime
    data: Optional[dict[str, Any]]
    epoch: Optional[int]
    seed_id: Optional[str]
    slot_id: Optional[str]
    severity: Optional[str]
    message: Optional[str]


@dataclass(frozen=True)
class KarnSlotConfig:
    """Minimal SlotConfig-like structure for Karn internal use.
    
    Decouples Karn from esper.leyline.slot_config.SlotConfig.
    """
    slot_ids: list[str] = field(default_factory=lambda: ["r0c0", "r0c1", "r0c2"])
    num_slots: int = 3

    @classmethod
    def default(cls) -> "KarnSlotConfig":
        return cls()
