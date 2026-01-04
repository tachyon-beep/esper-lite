# src/esper/karn/sanctum/registry.py

"""Aggregator Registry for multi-policy A/B testing."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.karn.sanctum.schema import SanctumSnapshot
    from esper.leyline.telemetry import TelemetryEvent


class AggregatorRegistry:
    """Manages multiple SanctumAggregators for A/B testing.

    Each PolicyGroup gets its own aggregator, keyed by group_id.
    The registry creates aggregators on-demand when first accessed.
    """

    def __init__(self, num_envs: int = 4, max_event_log: int = 100) -> None:
        self._lock = threading.Lock()
        self._num_envs = num_envs
        self._max_event_log = max_event_log
        self._aggregators: dict[str, SanctumAggregator] = {}

    def get_or_create(self, group_id: str) -> "SanctumAggregator":
        """Get existing aggregator or create new one for group."""
        with self._lock:
            if group_id not in self._aggregators:
                from esper.karn.sanctum.aggregator import SanctumAggregator

                self._aggregators[group_id] = SanctumAggregator(
                    num_envs=self._num_envs,
                    max_event_log=self._max_event_log,
                )
            return self._aggregators[group_id]

    @property
    def group_ids(self) -> set[str]:
        """Return set of all registered group IDs."""
        with self._lock:
            return set(self._aggregators.keys())

    def get_all_snapshots(self) -> dict[str, "SanctumSnapshot"]:
        """Return snapshots from all aggregators."""
        with self._lock:
            items = list(self._aggregators.items())

        return {group_id: agg.get_snapshot() for group_id, agg in items}

    def process_event(self, event: "TelemetryEvent") -> None:
        """Route event to appropriate aggregator based on group_id."""
        agg = self.get_or_create(event.group_id)
        agg.process_event(event)
