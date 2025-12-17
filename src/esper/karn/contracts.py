"""Shared contracts and types for Karn subsystem.

Defines local protocols and data structures to decouple Karn from external
subsystems like Leyline/Nissa where appropriate (dependency inversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, Union


class TelemetryEventLike(Protocol):
    """Minimal TelemetryEvent-like Protocol for Karn internal use.

    Decouples Karn from esper.leyline.telemetry.TelemetryEvent.

    This Protocol defines the contract for telemetry events consumed by Karn.
    Any object with these attributes can be used, including:
    - Leyline's TelemetryEvent dataclass
    - Custom event objects implementing this interface
    - Named tuples or dataclasses with matching fields

    Note on event_type:
        Can be either an Enum (with .name attribute) or a string.
        Karn serialization handles both via hasattr checks.

    Note on serialization:
        Karn uses explicit field access (not dataclasses.asdict) for
        serialization, so non-dataclass implementations are supported.
    """

    @property
    def event_type(self) -> Union[str, Any]:
        """Event type identifier. Enum with .name or string."""
        ...

    @property
    def timestamp(self) -> datetime:
        """When the event occurred."""
        ...

    @property
    def data(self) -> Optional[dict[str, Any]]:
        """Event payload (varies by event_type)."""
        ...

    @property
    def epoch(self) -> Optional[int]:
        """Training epoch when event occurred."""
        ...

    @property
    def seed_id(self) -> Optional[str]:
        """Seed identifier (for seed-related events)."""
        ...

    @property
    def slot_id(self) -> Optional[str]:
        """Slot identifier (for slot-related events)."""
        ...

    @property
    def severity(self) -> Optional[str]:
        """Log severity: debug, info, warning, error, critical."""
        ...

    @property
    def message(self) -> Optional[str]:
        """Human-readable event description."""
        ...


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
