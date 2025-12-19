"""Overwatch Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the TelemetryAggregator for TUI consumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from esper.karn.overwatch.aggregator import TelemetryAggregator

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent
    from esper.karn.overwatch.schema import TuiSnapshot


class OverwatchBackend:
    """OutputBackend that feeds telemetry to Overwatch TUI.

    Thread-safe: emit() can be called from training thread while
    get_snapshot() is called from UI thread (aggregator handles locking).

    Usage:
        from esper.nissa import get_hub
        from esper.karn.overwatch.backend import OverwatchBackend

        backend = OverwatchBackend()
        get_hub().add_backend(backend)

        # In UI thread
        snapshot = backend.get_snapshot()
    """

    def __init__(self, num_envs: int = 4, max_feed_events: int = 100):
        """Initialize the backend.

        Args:
            num_envs: Expected number of training environments.
            max_feed_events: Maximum events to keep in feed.
        """
        self._aggregator = TelemetryAggregator(
            num_envs=num_envs,
            max_feed_events=max_feed_events,
        )
        self._started = False

    def start(self) -> None:
        """Start the backend (required by OutputBackend protocol)."""
        self._started = True

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit telemetry event to aggregator.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            return
        self._aggregator.process_event(event)

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def get_snapshot(self) -> "TuiSnapshot":
        """Get current TuiSnapshot for UI rendering.

        Returns:
            Snapshot of current aggregator state.
        """
        return self._aggregator.get_snapshot()
