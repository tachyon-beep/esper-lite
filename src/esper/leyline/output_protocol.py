"""Leyline Output Protocol - Contract for telemetry output backends.

OutputBackend defines the interface that telemetry backends must implement.
This decouples the hub (nissa) from backend implementations (karn, file, console).

Used by:
- nissa: Hub routes events through this protocol
- karn: Implements backends (Sanctum, Overwatch, file storage)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent


@runtime_checkable
class OutputBackend(Protocol):
    """Protocol for telemetry output backends.

    This is a structural typing protocol - any class with matching methods
    satisfies this type, no explicit inheritance required. Classes MAY
    inherit from this Protocol to get default implementations of start()
    and close().

    Methods:
        start: Initialize the backend (default: no-op)
        emit: Process a telemetry event (required)
        close: Release resources (default: no-op)
    """

    def start(self) -> bool | None:
        """Start the backend (e.g., open files, start threads).

        Returns:
            True if started successfully, False if failed (optional dependencies
            missing), or None for backends that don't report status.
        """
        ...

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit a telemetry event to this backend.

        Args:
            event: The telemetry event to emit.
        """
        ...

    def close(self) -> None:
        """Close the backend and release resources."""
        ...


__all__ = ["OutputBackend"]
