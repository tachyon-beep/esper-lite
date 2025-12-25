# tests/karn/sanctum/test_registry.py

from esper.karn.sanctum.registry import AggregatorRegistry
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType


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


def test_registry_routes_events_by_group_id():
    """Registry should route events to correct aggregator based on group_id."""
    registry = AggregatorRegistry(num_envs=4)

    # Create events for different groups
    event_a = TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        group_id="A",
        message="Group A event"
    )
    event_b = TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        group_id="B",
        message="Group B event"
    )

    # Process events
    registry.process_event(event_a)
    registry.process_event(event_b)

    # Verify aggregators were created for both groups
    assert "A" in registry.group_ids
    assert "B" in registry.group_ids
    assert len(registry.group_ids) == 2


def test_registry_default_group_for_missing_group_id():
    """Registry should use default group when group_id is default."""
    registry = AggregatorRegistry(num_envs=4)

    # Create event with default group_id
    event = TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        group_id="default",
        message="Default group event"
    )

    # Process event
    registry.process_event(event)

    # Verify default aggregator was created
    assert "default" in registry.group_ids
    assert len(registry.group_ids) == 1
