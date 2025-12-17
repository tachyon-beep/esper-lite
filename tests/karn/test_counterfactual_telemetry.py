"""Tests for counterfactual telemetry emission."""

from unittest.mock import Mock, patch
import numpy as np


def test_shapley_computed_event_emitted():
    """Test that Shapley telemetry is emitted when enabled."""
    from esper.karn.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult
    from esper.leyline import TelemetryEventType

    events = []

    with patch("esper.karn.counterfactual.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        engine = CounterfactualEngine(emit_telemetry=True)

        # Create a mock counterfactual matrix with some results
        matrix = CounterfactualMatrix(
            epoch=10,
            strategy_used="full_factorial",
            compute_time_seconds=1.5,
            configs=[
                CounterfactualResult(
                    config=(False, False),
                    slot_ids=("r0c0", "r0c1"),
                    val_accuracy=0.70,
                ),
                CounterfactualResult(
                    config=(True, False),
                    slot_ids=("r0c0", "r0c1"),
                    val_accuracy=0.75,
                ),
                CounterfactualResult(
                    config=(False, True),
                    slot_ids=("r0c0", "r0c1"),
                    val_accuracy=0.72,
                ),
                CounterfactualResult(
                    config=(True, True),
                    slot_ids=("r0c0", "r0c1"),
                    val_accuracy=0.78,
                ),
            ]
        )

        # Compute Shapley values (should emit telemetry)
        values = engine.compute_shapley_values(matrix)

    # Check that event was emitted
    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and e.data.get("kind") == "shapley_computed"]
    assert len(shapley_events) == 1
    assert "shapley_values" in shapley_events[0].data
    assert "num_slots" in shapley_events[0].data
    assert shapley_events[0].data["num_slots"] == 2


def test_no_shapley_event_when_telemetry_disabled():
    """Test that Shapley telemetry is NOT emitted when disabled."""
    from esper.karn.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult

    events = []

    with patch("esper.karn.counterfactual.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        # emit_telemetry defaults to False
        engine = CounterfactualEngine()

        # Create a mock counterfactual matrix
        matrix = CounterfactualMatrix(
            epoch=10,
            configs=[
                CounterfactualResult(
                    config=(False, False),
                    slot_ids=("r0c0", "r0c1"),
                    val_accuracy=0.70,
                ),
                CounterfactualResult(
                    config=(True, True),
                    slot_ids=("r0c0", "r0c1"),
                    val_accuracy=0.78,
                ),
            ]
        )

        # Compute Shapley values (should NOT emit telemetry)
        values = engine.compute_shapley_values(matrix)

    # Check that NO event was emitted
    assert len(events) == 0


def test_shapley_event_includes_all_slots():
    """Test that Shapley event includes values for all slots."""
    from esper.karn.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult
    from esper.leyline import TelemetryEventType

    events = []

    with patch("esper.karn.counterfactual.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        engine = CounterfactualEngine(emit_telemetry=True)

        # Three-slot configuration
        matrix = CounterfactualMatrix(
            epoch=15,
            configs=[
                CounterfactualResult(
                    config=(False, False, False),
                    slot_ids=("r0c0", "r0c1", "r0c2"),
                    val_accuracy=0.65,
                ),
                CounterfactualResult(
                    config=(True, True, True),
                    slot_ids=("r0c0", "r0c1", "r0c2"),
                    val_accuracy=0.82,
                ),
            ]
        )

        values = engine.compute_shapley_values(matrix)

    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and e.data.get("kind") == "shapley_computed"]
    assert len(shapley_events) == 1

    # Should have all three slots in the data
    shapley_data = shapley_events[0].data["shapley_values"]
    assert len(shapley_data) >= 3  # At least the three slots (may have estimates)
    assert shapley_events[0].data["num_slots"] == 3


def test_counterfactual_helper_emits_shapley_telemetry():
    """Test that CounterfactualHelper emits Shapley telemetry when emit_events=True."""
    from esper.karn.counterfactual_helper import CounterfactualHelper
    from esper.leyline import TelemetryEventType

    events = []

    # Mock the hub to capture events
    with patch("esper.karn.counterfactual.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        # Create helper with telemetry enabled
        helper = CounterfactualHelper(
            strategy="full_factorial",
            shapley_samples=20,
            emit_events=True,
        )

        # Mock evaluate function
        def evaluate_fn(alpha_settings):
            # Simulate some variation based on which slots are enabled
            acc = 0.70
            for slot_id, alpha in alpha_settings.items():
                if alpha > 0.5:
                    acc += 0.03
            return 0.3, min(acc, 1.0)

        # Compute contributions
        slot_ids = ["r0c0", "r0c1"]
        results = helper.compute_contributions(slot_ids, evaluate_fn, epoch=5)

        # Verify results were computed
        assert len(results) == 2
        assert "r0c0" in results
        assert "r0c1" in results

    # Check that Shapley telemetry was emitted
    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and e.data.get("kind") == "shapley_computed"]
    assert len(shapley_events) == 1
    assert "shapley_values" in shapley_events[0].data
    assert shapley_events[0].data["num_slots"] == 2


def test_counterfactual_helper_no_telemetry_when_disabled():
    """Test that CounterfactualHelper does NOT emit Shapley telemetry when emit_events=False."""
    from esper.karn.counterfactual_helper import CounterfactualHelper
    from esper.leyline import TelemetryEventType

    events = []

    # Mock the hub to capture events
    with patch("esper.karn.counterfactual.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        # Create helper with telemetry disabled
        helper = CounterfactualHelper(
            strategy="full_factorial",
            shapley_samples=20,
            emit_events=False,
        )

        # Mock evaluate function
        def evaluate_fn(alpha_settings):
            acc = 0.70
            for slot_id, alpha in alpha_settings.items():
                if alpha > 0.5:
                    acc += 0.03
            return 0.3, min(acc, 1.0)

        # Compute contributions
        slot_ids = ["r0c0", "r0c1"]
        results = helper.compute_contributions(slot_ids, evaluate_fn, epoch=5)

        # Verify results were computed
        assert len(results) == 2

    # Check that NO Shapley telemetry was emitted
    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and e.data.get("kind") == "shapley_computed"]
    assert len(shapley_events) == 0
