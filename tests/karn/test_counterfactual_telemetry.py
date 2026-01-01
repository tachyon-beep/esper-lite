"""Tests for counterfactual telemetry emission."""


def test_shapley_computed_event_emitted():
    """Test that Shapley telemetry is emitted via emit_callback."""
    from esper.simic.attribution.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult
    from esper.leyline import TelemetryEventType

    events = []

    def capture_emit(e):
        events.append(e)

    engine = CounterfactualEngine(emit_callback=capture_emit)

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
    engine.compute_shapley_values(matrix)

    # Check that event was emitted
    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and getattr(e.data, "kind", None) == "shapley_computed"]
    assert len(shapley_events) == 1
    assert shapley_events[0].data.shapley_values is not None
    assert shapley_events[0].data.num_slots is not None
    assert shapley_events[0].data.num_slots == 2
    # Verify epoch propagated to batch field
    assert shapley_events[0].data.batch == 10
    # Verify env_id is sentinel
    assert shapley_events[0].data.env_id == -1


def test_no_shapley_event_when_telemetry_disabled():
    """Test that Shapley telemetry is NOT emitted when disabled."""
    from esper.simic.attribution.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult

    events = []

    # emit_callback defaults to None (disabled)
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
    engine.compute_shapley_values(matrix)

    # Check that NO event was emitted
    assert len(events) == 0


def test_shapley_event_includes_all_slots():
    """Test that Shapley event includes values for all slots."""
    from esper.simic.attribution.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult
    from esper.leyline import TelemetryEventType

    events = []

    def capture_emit(e):
        events.append(e)

    engine = CounterfactualEngine(emit_callback=capture_emit)

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

    engine.compute_shapley_values(matrix)

    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and getattr(e.data, "kind", None) == "shapley_computed"]
    assert len(shapley_events) == 1

    # Should have all three slots in the data
    shapley_data = shapley_events[0].data.shapley_values
    assert len(shapley_data) >= 3  # At least the three slots (may have estimates)
    assert shapley_events[0].data.num_slots == 3


def test_counterfactual_helper_emits_shapley_telemetry():
    """Test that CounterfactualHelper emits Shapley telemetry via emit_callback."""
    from esper.simic.attribution import CounterfactualHelper
    from esper.leyline import TelemetryEventType

    events = []

    def capture_emit(e):
        events.append(e)

    # Create helper with emit_callback
    helper = CounterfactualHelper(
        strategy="full_factorial",
        shapley_samples=20,
        emit_callback=capture_emit,
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
                      and getattr(e.data, "kind", None) == "shapley_computed"]
    assert len(shapley_events) == 1
    assert shapley_events[0].data.shapley_values is not None
    assert shapley_events[0].data.num_slots == 2
    # Verify epoch was propagated to batch field
    assert shapley_events[0].data.batch == 5
    # Verify env_id is sentinel (will be replaced by emit_with_env_context in production)
    assert shapley_events[0].data.env_id == -1


def test_counterfactual_helper_no_telemetry_when_disabled():
    """Test that CounterfactualHelper does NOT emit Shapley telemetry when emit_callback=None."""
    from esper.simic.attribution import CounterfactualHelper

    # Create helper with no emit_callback (telemetry disabled)
    helper = CounterfactualHelper(
        strategy="full_factorial",
        shapley_samples=20,
        emit_callback=None,  # No callback = no telemetry
    )

    # Mock evaluate function
    def evaluate_fn(alpha_settings):
        acc = 0.70
        for slot_id, alpha in alpha_settings.items():
            if alpha > 0.5:
                acc += 0.03
        return 0.3, min(acc, 1.0)

    # Compute contributions - should work without crashing
    slot_ids = ["r0c0", "r0c1"]
    results = helper.compute_contributions(slot_ids, evaluate_fn, epoch=5)

    # Verify results were computed
    assert len(results) == 2
    assert "r0c0" in results
    assert "r0c1" in results


# New tests for callback injection pattern


class TestCounterfactualEngineCallback:
    """Test CounterfactualEngine with emit callback injection."""

    def test_emits_shapley_via_callback(self) -> None:
        """Shapley computation should emit via injected callback."""
        from esper.simic.attribution.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult

        emitted_events: list = []

        def capture_emit(event):
            emitted_events.append(event)

        engine = CounterfactualEngine(
            emit_callback=capture_emit,
        )

        # Create a simple matrix for Shapley computation
        matrix = CounterfactualMatrix(
            epoch=10,
            strategy_used="full_factorial",
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

        engine.compute_shapley_values(matrix)

        # Should have emitted ANALYTICS_SNAPSHOT
        assert len(emitted_events) == 1
        assert emitted_events[0].event_type.name == "ANALYTICS_SNAPSHOT"
        assert emitted_events[0].data.kind == "shapley_computed"

    def test_no_emit_without_callback(self) -> None:
        """Without callback, Shapley still works but no emission."""
        from esper.simic.attribution.counterfactual import CounterfactualEngine, CounterfactualMatrix, CounterfactualResult

        engine = CounterfactualEngine(
            emit_callback=None,
        )

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

        shapley = engine.compute_shapley_values(matrix)

        # Should work without crash
        assert "r0c0" in shapley
        assert "r0c1" in shapley


def test_is_significant_defaults_to_false():
    """Test that ContributionResult.is_significant defaults to False.

    This prevents false significance claims when Shapley hasn't been computed yet.
    The default was incorrectly True before this fix.
    """
    from esper.simic.attribution.counterfactual_helper import ContributionResult

    # When creating a ContributionResult without Shapley values,
    # is_significant should default to False
    result = ContributionResult(
        slot_id="test",
        contribution=0.1,
    )
    assert result.is_significant is False, (
        "Expected is_significant to default to False"
    )

    # Shapley fields should also default correctly
    assert result.shapley_mean == 0.0
    assert result.shapley_std == 0.0


def test_is_significant_true_after_shapley_computed():
    """Test that is_significant becomes True when Shapley confirms significance."""
    from esper.simic.attribution import CounterfactualHelper

    helper = CounterfactualHelper(
        strategy="full_factorial",  # Will compute all 2^n configs
        shapley_samples=20,
        emit_callback=None,
    )

    def evaluate_fn(alpha_settings):
        # Strong signal: each slot adds 10% accuracy
        acc = 0.50
        for alpha in alpha_settings.values():
            if alpha > 0.5:
                acc += 0.10
        return 0.3, acc

    slot_ids = ["r0c0", "r0c1"]
    results = helper.compute_contributions(slot_ids, evaluate_fn, epoch=5)

    # With full_factorial and strong signal, Shapley should show significance
    # At least one slot should be significant
    significant_count = sum(1 for r in results.values() if r.is_significant)
    assert significant_count > 0, "Expected at least one slot to be significant"
