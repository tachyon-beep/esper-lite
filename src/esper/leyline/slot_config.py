"""SlotConfig dataclass for dynamic action spaces.

Replaces the fixed SlotAction enum with dynamic slot configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.leyline.slot_id import format_slot_id

if TYPE_CHECKING:
    from esper.leyline.injection_spec import InjectionSpec


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for slot action space.

    Replaces the fixed SlotAction enum with dynamic slot configuration.

    Attributes:
        slot_ids: Tuple of slot IDs in action space order.
        _channel_map: Internal mapping of slot_id -> channels (frozen dict workaround).
    """

    slot_ids: tuple[str, ...]
    _channel_map: tuple[tuple[str, int], ...] = field(default=())

    def __post_init__(self) -> None:
        """Validate slot configuration invariants."""
        if not self.slot_ids:
            raise ValueError("SlotConfig requires at least one slot")
        if len(self.slot_ids) != len(set(self.slot_ids)):
            duplicates = [s for s in self.slot_ids if self.slot_ids.count(s) > 1]
            raise ValueError(f"SlotConfig slot_ids must be unique, found duplicates: {set(duplicates)}")

    @property
    def num_slots(self) -> int:
        """Number of slots in this configuration."""
        return len(self.slot_ids)

    def slot_id_for_index(self, idx: int) -> str:
        """Get slot ID for action index.

        Args:
            idx: Action index (0-based).

        Returns:
            Slot ID at that index.

        Raises:
            IndexError: If index is out of range.
        """
        return self.slot_ids[idx]

    def index_for_slot_id(self, slot_id: str) -> int:
        """Get action index for slot ID.

        Args:
            slot_id: Slot ID to find.

        Returns:
            Index of that slot ID.

        Raises:
            ValueError: If slot ID not found in configuration.
        """
        return self.slot_ids.index(slot_id)

    def channels_for_slot(self, slot_id: str) -> int:
        """Get channel dimension for a slot.

        Args:
            slot_id: Slot ID to look up.

        Returns:
            Channel count, or 0 if not available.
        """
        for sid, channels in self._channel_map:
            if sid == slot_id:
                return channels
        return 0

    @classmethod
    def default(cls) -> "SlotConfig":
        """Default 3-slot configuration (legacy compatible).

        Returns:
            SlotConfig with 3 slots in a single row (r0c0, r0c1, r0c2).
        """
        return cls(slot_ids=("r0c0", "r0c1", "r0c2"))

    @classmethod
    def from_specs(cls, specs: list[InjectionSpec]) -> "SlotConfig":
        """Create config from host injection specs.

        Args:
            specs: List of InjectionSpec from host.

        Returns:
            SlotConfig with slots sorted by position.

        Raises:
            ValueError: If specs is empty.
        """
        if not specs:
            raise ValueError("SlotConfig.from_specs requires at least one InjectionSpec")

        # Sort by position (early -> late in network)
        sorted_specs = sorted(specs, key=lambda s: s.position)

        slot_ids = tuple(s.slot_id for s in sorted_specs)
        channel_map = tuple((s.slot_id, s.channels) for s in sorted_specs)

        return cls(slot_ids=slot_ids, _channel_map=channel_map)

    @classmethod
    def for_grid(cls, rows: int, cols: int) -> "SlotConfig":
        """Create config for a full grid.

        Args:
            rows: Number of rows in grid.
            cols: Number of columns in grid.

        Returns:
            SlotConfig with all slots in row-major order.
        """
        slot_ids = tuple(format_slot_id(r, c) for r in range(rows) for c in range(cols))
        return cls(slot_ids=slot_ids)
