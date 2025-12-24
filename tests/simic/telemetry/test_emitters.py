"""Tests for simic telemetry emitters."""

from unittest.mock import MagicMock

from esper.simic.telemetry.emitters import emit_ppo_update_event


def test_emit_ppo_update_event_propagates_group_id():
    """emit_ppo_update_event should propagate group_id to TelemetryEvent."""
    hub = MagicMock()

    emit_ppo_update_event(
        hub=hub,
        metrics={"policy_loss": 0.1, "value_loss": 0.2, "entropy": 1.5},
        episodes_completed=10,
        batch_idx=5,
        epoch=100,
        optimizer=None,
        grad_norm=1.0,
        update_time_ms=50.0,
        group_id="B",  # New parameter
    )

    # Verify hub.emit was called
    hub.emit.assert_called_once()
    event = hub.emit.call_args[0][0]

    # Verify group_id was propagated
    assert event.group_id == "B"
