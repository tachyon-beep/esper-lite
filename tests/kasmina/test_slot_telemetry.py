def test_lifecycle_events_include_alpha_and_epochs():
    from esper.kasmina.slot import SeedSlot, SeedState
    from esper.leyline import SeedStage, TelemetryEventType

    emitted = []
    slot = SeedSlot(
        slot_id="r0c1",
        channels=64,
        device="cpu",
        task_config=None,
        on_telemetry=emitted.append,
    )
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1", stage=SeedStage.TRAINING)
    slot.state.alpha = 0.3
    slot.telemetry_inner_epoch = 5
    slot.telemetry_global_epoch = 12

    slot._emit_telemetry(TelemetryEventType.SEED_STAGE_CHANGED, data={})

    assert emitted
    event = emitted[0]
    assert event.data["alpha"] == 0.3
    assert event.data["inner_epoch"] == 5
    assert event.data["global_epoch"] == 12


def test_lifecycle_events_include_health_fields_when_available():
    from esper.kasmina.slot import SeedSlot, SeedState
    from esper.leyline import SeedStage, TelemetryEventType

    emitted = []
    slot = SeedSlot(
        slot_id="r0c1",
        channels=64,
        device="cpu",
        task_config=None,
        on_telemetry=emitted.append,
    )
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c1", stage=SeedStage.TRAINING)
    slot.state.metrics.seed_gradient_norm_ratio = 0.42
    slot.state.sync_telemetry(
        gradient_norm=1.0,
        gradient_health=0.9,
        has_vanishing=True,
        has_exploding=False,
        epoch=1,
        max_epochs=25,
    )

    slot._emit_telemetry(TelemetryEventType.SEED_STAGE_CHANGED, data={"from": "A", "to": "B"})
    payload = emitted[-1].data

    assert payload["seed_gradient_norm_ratio"] == 0.42
    assert payload["gradient_health"] == 0.9
    assert payload["has_vanishing"] is True
    assert payload["has_exploding"] is False


def test_sync_telemetry_training_stage_accuracy_delta_zero():
    """TRAINING stage seeds should always have accuracy_delta=0.0.

    Seeds in TRAINING have alpha=0 and cannot affect host output,
    so their causal contribution is always zero regardless of
    host accuracy changes during their existence.
    """
    from esper.kasmina.slot import SeedState
    from esper.leyline import SeedStage

    state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c0", stage=SeedStage.TRAINING)

    # Simulate accuracy improvement that would normally show positive delta
    state.metrics.record_accuracy(80.0)  # Initial
    state.metrics.record_accuracy(85.0)  # Improved by 5%

    # Verify metrics tracked improvement
    assert state.metrics.improvement_since_stage_start == 5.0

    # But sync_telemetry should set accuracy_delta to 0 for TRAINING
    state.sync_telemetry(
        gradient_norm=1.0,
        gradient_health=0.9,
        has_vanishing=False,
        has_exploding=False,
        epoch=5,
        max_epochs=25,
    )

    assert state.telemetry.accuracy_delta == 0.0, (
        "TRAINING seeds have alpha=0 and cannot affect output, "
        "so accuracy_delta must be 0.0"
    )


def test_sync_telemetry_germinated_stage_accuracy_delta_zero():
    """GERMINATED stage seeds should also have accuracy_delta=0.0."""
    from esper.kasmina.slot import SeedState
    from esper.leyline import SeedStage

    state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c0", stage=SeedStage.GERMINATED)

    state.metrics.record_accuracy(80.0)
    state.metrics.record_accuracy(82.0)

    state.sync_telemetry(
        gradient_norm=1.0,
        gradient_health=0.9,
        has_vanishing=False,
        has_exploding=False,
        epoch=1,
        max_epochs=25,
    )

    assert state.telemetry.accuracy_delta == 0.0


def test_sync_telemetry_blending_stage_uses_improvement():
    """BLENDING stage seeds should use improvement_since_stage_start.

    Seeds in BLENDING have alpha>0 and contribute to host output,
    so their accuracy_delta reflects stage-relative improvement.
    """
    from esper.kasmina.slot import SeedState
    from esper.leyline import SeedStage

    state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c0", stage=SeedStage.BLENDING)

    state.metrics.record_accuracy(80.0)  # Start of stage
    state.metrics.record_accuracy(83.5)  # Improved by 3.5%

    state.sync_telemetry(
        gradient_norm=1.0,
        gradient_health=0.9,
        has_vanishing=False,
        has_exploding=False,
        epoch=10,
        max_epochs=25,
    )

    # BLENDING seeds use actual improvement
    assert state.telemetry.accuracy_delta == 3.5


def test_sync_telemetry_probationary_stage_uses_improvement():
    """PROBATIONARY stage seeds should use improvement_since_stage_start."""
    from esper.kasmina.slot import SeedState
    from esper.leyline import SeedStage

    state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="r0c0", stage=SeedStage.PROBATIONARY)

    state.metrics.record_accuracy(85.0)
    state.metrics.record_accuracy(86.0)

    state.sync_telemetry(
        gradient_norm=1.0,
        gradient_health=0.9,
        has_vanishing=False,
        has_exploding=False,
        epoch=15,
        max_epochs=25,
    )

    assert state.telemetry.accuracy_delta == 1.0
