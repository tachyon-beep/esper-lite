"""Sanctum Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the AggregatorRegistry for TUI consumption.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from esper.karn.sanctum.registry import AggregatorRegistry

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent
    from esper.karn.sanctum.schema import SanctumSnapshot

_logger = logging.getLogger(__name__)


class SanctumBackend:
    """OutputBackend that feeds telemetry to Sanctum TUI.

    Thread-safe: emit() can be called from training thread while
    get_snapshot()/get_all_snapshots() called from UI thread.

    Uses AggregatorRegistry internally to support A/B testing with
    multiple policy groups. Each group_id gets its own aggregator.
    """

    def __init__(self, num_envs: int = 16, max_event_log: int = 100):
        """Initialize the backend.

        Args:
            num_envs: Expected number of training environments.
            max_event_log: Maximum events to keep in log.
        """
        self._registry = AggregatorRegistry(
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
        """Emit telemetry event to appropriate aggregator.

        Routes to aggregator based on event.group_id.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            _logger.warning("SanctumBackend.emit() called before start()")
            return
        self._event_count += 1
        self._registry.process_event(event)

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def get_snapshot(self) -> "SanctumSnapshot":
        """Get merged SanctumSnapshot for backward compatibility.

        For single-policy mode or legacy callers. Returns the first
        group's snapshot, or empty snapshot if no events received.

        Returns:
            Snapshot of current aggregator state.
        """
        snapshots = self._registry.get_all_snapshots()
        if not snapshots:
            # No events yet - return empty snapshot
            from esper.karn.sanctum.schema import SanctumSnapshot
            snapshot = SanctumSnapshot()
        else:
            # Return first group's snapshot (alphabetically)
            group_id = sorted(snapshots.keys())[0]
            snapshot = snapshots[group_id]

        # Add event count for debugging
        snapshot.total_events_received = self._event_count
        return snapshot

    def get_all_snapshots(self) -> dict[str, "SanctumSnapshot"]:
        """Get snapshots for all policy groups.

        For A/B testing mode. Each group_id maps to its aggregator's snapshot.

        Returns:
            Dict mapping group_id to SanctumSnapshot.
        """
        snapshots = self._registry.get_all_snapshots()
        # Add event count to each snapshot
        for snapshot in snapshots.values():
            snapshot.total_events_received = self._event_count
        return snapshots

    def toggle_decision_pin(self, decision_id: str) -> bool:
        """Toggle pin status for a decision.

        Args:
            decision_id: ID of the decision to toggle.

        Returns:
            New pin status (True if pinned, False if unpinned).
        """
        # Pin applies to first group (or specific group if ID contains group prefix)
        snapshots = self._registry.get_all_snapshots()
        if not snapshots:
            return False
        group_id = sorted(snapshots.keys())[0]
        aggregator = self._registry.get_or_create(group_id)
        return aggregator.toggle_decision_pin(decision_id)

    def toggle_best_run_pin(self, record_id: str) -> bool:
        """Toggle pin status for a best run record.

        Args:
            record_id: ID of the record to toggle.

        Returns:
            New pin status (True if pinned, False if unpinned).
        """
        # Pin applies to first group
        snapshots = self._registry.get_all_snapshots()
        if not snapshots:
            return False
        group_id = sorted(snapshots.keys())[0]
        aggregator = self._registry.get_or_create(group_id)
        return aggregator.toggle_best_run_pin(record_id)
