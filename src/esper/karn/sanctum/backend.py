"""Sanctum Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the AggregatorRegistry for TUI consumption.
"""

from __future__ import annotations

import logging
import threading
import traceback
from typing import TYPE_CHECKING

from esper.karn.sanctum.errors import SanctumTelemetryFatalError
from esper.karn.sanctum.registry import AggregatorRegistry

if TYPE_CHECKING:
    from esper.karn.sanctum.widgets.reward_health import RewardHealthData
    from esper.karn.sanctum.schema import SanctumSnapshot
    from esper.leyline import TelemetryEvent

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
        self._fatal_error: SanctumTelemetryFatalError | None = None
        self._fatal_lock = threading.Lock()

    def start(self) -> None:
        """Start the backend (required by OutputBackend protocol)."""
        self._started = True
        with self._fatal_lock:
            self._fatal_error = None
        _logger.info("SanctumBackend started")

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit telemetry event to appropriate aggregator.

        Routes to aggregator based on event.group_id.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            raise RuntimeError("SanctumBackend.emit() called before start()")

        with self._fatal_lock:
            fatal = self._fatal_error
        if fatal is not None:
            return

        try:
            self._event_count += 1
            self._registry.process_event(event)
        except Exception as e:
            tb = traceback.format_exc()
            fatal = SanctumTelemetryFatalError(
                f"Sanctum telemetry processing failed: {e}",
                tb,
            )
            with self._fatal_lock:
                self._fatal_error = fatal
            _logger.exception("SanctumBackend encountered fatal telemetry error")
            raise fatal

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def _raise_if_fatal(self) -> None:
        with self._fatal_lock:
            fatal = self._fatal_error
        if fatal is not None:
            raise fatal

    def get_snapshot(self) -> "SanctumSnapshot":
        """Get merged SanctumSnapshot for backward compatibility.

        For single-policy mode or legacy callers. Returns the first
        group's snapshot, or empty snapshot if no events received.

        Returns:
            Snapshot of current aggregator state.
        """
        self._raise_if_fatal()
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
        self._raise_if_fatal()
        snapshots = self._registry.get_all_snapshots()
        # Add event count to each snapshot
        for snapshot in snapshots.values():
            snapshot.total_events_received = self._event_count
        return snapshots

    def compute_reward_health(self) -> "RewardHealthData":
        """Compute reward health metrics for one policy group.

        Returns:
            RewardHealthData with computed metrics.
        """
        from esper.karn.sanctum.widgets.reward_health import RewardHealthData

        self._raise_if_fatal()
        group_ids = self._registry.group_ids
        if not group_ids:
            return RewardHealthData()
        if "default" in group_ids:
            chosen_group_id = "default"
        else:
            chosen_group_id = sorted(group_ids)[0]
        aggregator = self._registry.get_or_create(chosen_group_id)
        return aggregator.compute_reward_health()

    def compute_reward_health_by_group(self) -> dict[str, "RewardHealthData"]:
        """Compute reward health metrics for all policy groups."""
        self._raise_if_fatal()
        group_ids = sorted(self._registry.group_ids)
        if not group_ids:
            return {}

        by_group: dict[str, "RewardHealthData"] = {}
        for group_id in group_ids:
            aggregator = self._registry.get_or_create(group_id)
            by_group[group_id] = aggregator.compute_reward_health()
        return by_group
