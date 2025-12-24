# src/esper/karn/sanctum/registry.py

"""Aggregator Registry for multi-policy A/B testing."""

from __future__ import annotations

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

    def __init__(self, num_envs: int = 4) -> None:
        self._num_envs = num_envs
        self._aggregators: dict[str, SanctumAggregator] = {}

    def get_or_create(self, group_id: str) -> "SanctumAggregator":
        """Get existing aggregator or create new one for group."""
        if group_id not in self._aggregators:
            from esper.karn.sanctum.aggregator import SanctumAggregator
            self._aggregators[group_id] = SanctumAggregator(
                num_envs=self._num_envs
            )
        return self._aggregators[group_id]

    @property
    def group_ids(self) -> set[str]:
        """Return set of all registered group IDs."""
        return set(self._aggregators.keys())

    def get_all_snapshots(self) -> dict[str, "SanctumSnapshot"]:
        """Return snapshots from all aggregators."""
        return {
            group_id: agg.get_snapshot()
            for group_id, agg in self._aggregators.items()
        }

    def process_event(self, event: "TelemetryEvent") -> None:
        """Route event to appropriate aggregator based on group_id."""
        group_id = event.group_id
        agg = self.get_or_create(group_id)
        agg.process_event(event)
