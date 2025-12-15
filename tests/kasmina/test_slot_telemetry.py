def test_lifecycle_events_include_alpha_and_epochs():
    from esper.kasmina.slot import SeedSlot, SeedState
    from esper.leyline import SeedStage, TelemetryEventType

    emitted = []
    slot = SeedSlot(
        slot_id="mid",
        channels=64,
        device="cpu",
        task_config=None,
        on_telemetry=emitted.append,
    )
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="mid", stage=SeedStage.TRAINING)
    slot.state.alpha = 0.3
    slot.telemetry_inner_epoch = 5
    slot.telemetry_global_epoch = 12

    slot._emit_telemetry(TelemetryEventType.SEED_STAGE_CHANGED, data={})

    assert emitted
    event = emitted[0]
    assert event.data["alpha"] == 0.3
    assert event.data["inner_epoch"] == 5
    assert event.data["global_epoch"] == 12

