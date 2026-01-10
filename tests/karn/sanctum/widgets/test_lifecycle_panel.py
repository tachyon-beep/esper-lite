"""Tests for LifecyclePanel widget."""

from rich.panel import Panel

from esper.karn.sanctum.schema import SeedLifecycleEvent
from esper.karn.sanctum.widgets.lifecycle_panel import LifecyclePanel


def test_lifecycle_panel_renders_events():
    """LifecyclePanel should render lifecycle events."""
    events = [
        SeedLifecycleEvent(
            epoch=5,
            action="GERMINATE(conv_heavy)",
            from_stage="DORMANT",
            to_stage="GERMINATED",
            blueprint_id="conv_heavy",
            slot_id="r0c0",
            alpha=None,
            accuracy_delta=None,
        ),
        SeedLifecycleEvent(
            epoch=10,
            action="[auto]",
            from_stage="GERMINATED",
            to_stage="TRAINING",
            blueprint_id="conv_heavy",
            slot_id="r0c0",
            alpha=None,
            accuracy_delta=None,
        ),
    ]
    panel = LifecyclePanel(events=events, slot_filter="r0c0")

    # Widget should be creatable
    assert panel is not None
    assert panel._events == events
    assert panel._slot_filter == "r0c0"


def test_lifecycle_panel_filters_by_slot():
    """LifecyclePanel should filter events by slot."""
    events = [
        SeedLifecycleEvent(
            epoch=5,
            action="GERMINATE(conv_heavy)",
            from_stage="DORMANT",
            to_stage="GERMINATED",
            blueprint_id="conv_heavy",
            slot_id="r0c0",
            alpha=None,
            accuracy_delta=None,
        ),
        SeedLifecycleEvent(
            epoch=8,
            action="GERMINATE(attention)",
            from_stage="DORMANT",
            to_stage="GERMINATED",
            blueprint_id="attention",
            slot_id="r0c1",
            alpha=None,
            accuracy_delta=None,
        ),
    ]
    panel = LifecyclePanel(events=events, slot_filter="r0c0")
    filtered = panel._get_filtered_events()

    assert len(filtered) == 1
    assert filtered[0].slot_id == "r0c0"


def test_lifecycle_panel_all_slots():
    """LifecyclePanel with slot_filter=None shows all events."""
    events = [
        SeedLifecycleEvent(
            epoch=5,
            action="GERMINATE(conv_heavy)",
            from_stage="DORMANT",
            to_stage="GERMINATED",
            blueprint_id="conv_heavy",
            slot_id="r0c0",
            alpha=None,
            accuracy_delta=None,
        ),
        SeedLifecycleEvent(
            epoch=8,
            action="GERMINATE(attention)",
            from_stage="DORMANT",
            to_stage="GERMINATED",
            blueprint_id="attention",
            slot_id="r0c1",
            alpha=None,
            accuracy_delta=None,
        ),
    ]
    panel = LifecyclePanel(events=events, slot_filter=None)
    filtered = panel._get_filtered_events()

    assert len(filtered) == 2


def test_lifecycle_panel_render_output():
    """LifecyclePanel render() should produce a Panel with event lines."""
    events = [
        SeedLifecycleEvent(
            epoch=5,
            action="GERMINATE(conv_heavy)",
            from_stage="DORMANT",
            to_stage="GERMINATED",
            blueprint_id="conv_heavy",
            slot_id="r0c0",
            alpha=None,
            accuracy_delta=None,
        ),
    ]
    panel = LifecyclePanel(events=events, slot_filter="r0c0")
    result = panel.render()

    # Verify it returns a Panel
    assert isinstance(result, Panel)
    # Verify title includes filter label
    assert "r0c0" in result.title
