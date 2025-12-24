"""Base class for Sanctum widgets with update protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class SanctumWidget(Protocol):
    """Protocol for Sanctum widgets that receive snapshot updates."""

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        ...
