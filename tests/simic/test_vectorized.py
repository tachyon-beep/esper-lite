from unittest.mock import Mock, patch


def test_lifecycle_only_keeps_slot_telemetry():
    from esper.simic import vectorized

    slot = Mock()
    slot.fast_mode = False
    slot.on_telemetry = None
    slot.telemetry_lifecycle_only = False

    env_state = Mock()
    env_state.model = Mock(seed_slots={"mid": slot})
    env_state.telemetry_cb = Mock()

    vectorized._apply_slot_telemetry(
        env_state,
        ops_telemetry_enabled=False,
        lifecycle_only=True,
    )

    assert slot.on_telemetry is env_state.telemetry_cb
    assert slot.fast_mode is True
    assert slot.telemetry_lifecycle_only is True


def test_emit_with_env_context_includes_device():
    from esper.leyline import TelemetryEvent, TelemetryEventType
    from esper.simic.vectorized import _emit_with_env_context

    hub = Mock()
    event = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data={})
    _emit_with_env_context(hub, env_idx=2, device="cpu", event=event)

    emitted = hub.emit.call_args[0][0]
    assert emitted.data["env_id"] == 2
    assert emitted.data["device"] == "cpu"


def test_last_action_event_emitted():
    from esper.leyline.factored_actions import FactoredAction
    from esper.simic import vectorized

    with patch("esper.simic.vectorized.get_hub") as get_hub:
        hub = Mock()
        get_hub.return_value = hub

        factored_action = FactoredAction.from_indices(
            slot_idx=1,
            blueprint_idx=1,
            blend_idx=1,
            op_idx=1,
        )
        vectorized._emit_last_action(
            env_id=0,
            epoch=3,
            factored_action=factored_action,
            masked={"op": False, "slot": False, "blueprint": False, "blend": True},
            success=True,
        )

        emitted = hub.emit.call_args[0][0]
        assert emitted.data["slot_id"] == "mid"
        assert emitted.data["blend_masked"] is True


def test_ppo_update_event_includes_vitals():
    from esper.simic import vectorized

    hub = Mock()
    metrics = {"policy_loss": 0.1}
    vectorized._emit_ppo_update_event(
        hub=hub,
        metrics=metrics,
        episodes_completed=5,
        batch_idx=0,
        epoch=3,
        optimizer=Mock(param_groups=[{"lr": 0.0003}]),
        grad_norm=1.23,
        update_time_ms=12.5,
    )
    data = hub.emit.call_args[0][0].data
    assert data["lr"] == 0.0003
    assert data["grad_norm"] == 1.23
    assert data["update_time_ms"] == 12.5


def test_action_distribution_snapshot():
    from esper.simic import vectorized

    hub = Mock()
    vectorized._emit_action_distribution(
        hub=hub,
        batch_idx=1,
        episodes_completed=4,
        action_counts={"WAIT": 3, "GERMINATE": 1},
        success_counts={"WAIT": 3, "GERMINATE": 1},
    )
    data = hub.emit.call_args[0][0].data
    assert data["action_counts"]["WAIT"] == 3


def test_counterfactual_unavailable_event():
    from esper.simic import vectorized

    hub = Mock()
    vectorized._emit_cf_unavailable(
        hub,
        env_id=0,
        slot_id="mid",
        reason="missing_baseline",
    )
    data = hub.emit.call_args[0][0].data
    assert data["available"] is False
    assert data["reason"] == "missing_baseline"


def test_throughput_metrics_emitted():
    from esper.simic import vectorized

    hub = Mock()
    vectorized._emit_throughput(
        hub=hub,
        env_id=0,
        batch_idx=1,
        episodes_completed=4,
        step_time_ms=5.0,
        dataloader_wait_ms=2.0,
    )
    data = hub.emit.call_args[0][0].data
    assert data["step_time_ms"] == 5.0
    assert data["dataloader_wait_ms"] == 2.0


def test_reward_summary_emitted():
    from esper.simic import vectorized

    hub = Mock()
    vectorized._emit_reward_summary(
        hub=hub,
        env_id=0,
        batch_idx=1,
        summary={"bounded_attribution": 0.4, "compute_rent": -0.1, "total_reward": 0.3},
    )
    data = hub.emit.call_args[0][0].data
    assert data["summary"]["total_reward"] == 0.3


def test_mask_hit_rates_emitted():
    from esper.simic import vectorized

    hub = Mock()
    vectorized._emit_mask_hit_rates(
        hub=hub,
        batch_idx=1,
        episodes_completed=4,
        mask_hits={"op": 10},
        mask_total={"op": 12},
    )
    data = hub.emit.call_args[0][0].data
    assert data["mask_hits"]["op"] == 10
    assert data["mask_total"]["op"] == 12
