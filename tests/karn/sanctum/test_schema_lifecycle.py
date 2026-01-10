"""Tests for SeedLifecycleEvent dataclass."""

from esper.karn.sanctum.schema import SeedLifecycleEvent


def test_lifecycle_event_creation():
    """SeedLifecycleEvent should capture all transition data."""
    event = SeedLifecycleEvent(
        epoch=12,
        action="GERMINATE(conv_heavy)",
        from_stage="DORMANT",
        to_stage="GERMINATED",
        blueprint_id="conv_heavy",
        slot_id="r0c0",
        alpha=None,
        accuracy_delta=None,
    )
    assert event.epoch == 12
    assert event.action == "GERMINATE(conv_heavy)"
    assert event.from_stage == "DORMANT"
    assert event.to_stage == "GERMINATED"
    assert event.blueprint_id == "conv_heavy"
    assert event.slot_id == "r0c0"
    assert event.alpha is None
    assert event.accuracy_delta is None


def test_lifecycle_event_with_alpha():
    """SeedLifecycleEvent should capture alpha for blending transitions."""
    event = SeedLifecycleEvent(
        epoch=31,
        action="ADVANCE",
        from_stage="TRAINING",
        to_stage="BLENDING",
        blueprint_id="conv_heavy",
        slot_id="r0c0",
        alpha=0.15,
        accuracy_delta=None,
    )
    assert event.alpha == 0.15


def test_lifecycle_event_with_accuracy_delta():
    """SeedLifecycleEvent should capture accuracy_delta for fossilize."""
    event = SeedLifecycleEvent(
        epoch=58,
        action="FOSSILIZE",
        from_stage="HOLDING",
        to_stage="FOSSILIZED",
        blueprint_id="conv_heavy",
        slot_id="r0c0",
        alpha=1.0,
        accuracy_delta=2.3,
    )
    assert event.accuracy_delta == 2.3
