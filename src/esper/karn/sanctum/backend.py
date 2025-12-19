"""Sanctum Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the SanctumAggregator for TUI consumption.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from esper.karn.sanctum.aggregator import SanctumAggregator

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent
    from esper.karn.sanctum.schema import SanctumSnapshot

_logger = logging.getLogger(__name__)


class SanctumBackend:
    """OutputBackend that feeds telemetry to Sanctum TUI.

    Thread-safe: emit() can be called from training thread while
    get_snapshot() is called from UI thread (aggregator handles locking).

    Usage:
        from esper.nissa import get_hub
        from esper.karn.sanctum.backend import SanctumBackend

        backend = SanctumBackend(num_envs=16)
        get_hub().add_backend(backend)

        # In UI thread
        snapshot = backend.get_snapshot()
    """

    def __init__(self, num_envs: int = 16, max_event_log: int = 100):
        """Initialize the backend.

        Args:
            num_envs: Expected number of training environments.
            max_event_log: Maximum events to keep in log.
        """
        self._aggregator = SanctumAggregator(
            num_envs=num_envs,
            max_event_log=max_event_log,
        )
        self._started = False
        self._event_count = 0

    def start(self) -> None:
        """Start the backend (required by OutputBackend protocol)."""
        self._started = True
        _logger.info("SanctumBackend started")

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit telemetry event to aggregator.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            _logger.warning("SanctumBackend.emit() called before start()")
            return
        self._event_count += 1
        self._aggregator.process_event(event)

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def get_snapshot(self) -> "SanctumSnapshot":
        """Get current SanctumSnapshot for UI rendering.

        Returns:
            Snapshot of current aggregator state.
        """
        snapshot = self._aggregator.get_snapshot()
        # Add event count for debugging
        snapshot.total_events_received = self._event_count
        return snapshot

    def toggle_decision_pin(self, decision_id: str) -> bool:
        """Toggle pin status for a decision.

        Args:
            decision_id: ID of the decision to toggle.

        Returns:
            New pin status (True if pinned, False if unpinned).
        """
        return self._aggregator.toggle_decision_pin(decision_id)
