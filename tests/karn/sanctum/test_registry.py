# tests/karn/sanctum/test_registry.py

import pytest
from esper.karn.sanctum.registry import AggregatorRegistry
from esper.karn.sanctum.aggregator import SanctumAggregator


def test_registry_creates_aggregator_on_demand():
    """Registry should create aggregator when first accessed."""
    registry = AggregatorRegistry(num_envs=4)

    # First access creates aggregator
    agg_a = registry.get_or_create("A")
    assert isinstance(agg_a, SanctumAggregator)

    # Second access returns same instance
    agg_a2 = registry.get_or_create("A")
    assert agg_a is agg_a2


def test_registry_manages_multiple_aggregators():
    """Registry should manage multiple independent aggregators."""
    registry = AggregatorRegistry(num_envs=4)

    agg_a = registry.get_or_create("A")
    agg_b = registry.get_or_create("B")

    # Different instances
    assert agg_a is not agg_b

    # Both tracked
    assert registry.group_ids == {"A", "B"}


def test_registry_list_snapshots():
    """Registry should return snapshots for all aggregators."""
    registry = AggregatorRegistry(num_envs=4)

    registry.get_or_create("A")
    registry.get_or_create("B")

    snapshots = registry.get_all_snapshots()

    assert len(snapshots) == 2
    assert "A" in snapshots
    assert "B" in snapshots
