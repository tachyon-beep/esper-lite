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
