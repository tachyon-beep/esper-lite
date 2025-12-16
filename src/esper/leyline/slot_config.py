"""SlotConfig dataclass for dynamic action spaces.

Replaces the fixed SlotAction enum with dynamic slot configuration.
"""

from dataclasses import dataclass

from esper.leyline.slot_id import format_slot_id


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for slot action space.

    Replaces the fixed SlotAction enum with dynamic slot configuration.

    Attributes:
        slot_ids: Tuple of slot IDs in action space order.
    """

    slot_ids: tuple[str, ...]

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

    @classmethod
    def default(cls) -> "SlotConfig":
        """Default 3-slot configuration (legacy compatible).

        Returns:
            SlotConfig with 3 slots in a single row (r0c0, r0c1, r0c2).
        """
        return cls(slot_ids=("r0c0", "r0c1", "r0c2"))

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
