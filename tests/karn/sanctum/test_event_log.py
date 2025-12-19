"""Tests for EventLog widget.

Tests cover:
- EventLog: Event log display with color-coded entries
"""
from io import StringIO

from rich.console import Console

from esper.karn.sanctum.schema import SanctumSnapshot


def render_to_text(renderable) -> str:
    """Helper to render a Rich renderable to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


# ============================================================================
# EventLog Tests
# ============================================================================


def test_event_log_creation():
    """Test widget creation."""
    from esper.karn.sanctum.widgets.event_log import EventLog

    widget = EventLog()
    assert widget is not None
    assert widget._max_events == 20
    assert widget.border_title == "EVENTS"


def test_event_log_no_events():
    """Test render with no events shows waiting message."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from rich.text import Text

    widget = EventLog()
    widget._snapshot = SanctumSnapshot(event_log=[])
    result = widget.render()

    # Should return "Waiting for events..." text
    assert isinstance(result, Text)
    assert "Waiting for events" in result.plain


def test_event_log_with_events():
    """Test render with events shows formatted log."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry
    from rich.console import Group

    widget = EventLog(max_events=10)
    widget._snapshot = SanctumSnapshot(
        event_log=[
            EventLogEntry(
                timestamp="10:15:30",
                event_type="TRAINING_STARTED",
                env_id=None,
                message="Training started"
            ),
            EventLogEntry(
                timestamp="10:15:31",
                event_type="REWARD_COMPUTED",
                env_id=0,
                message="WAIT r=+0.500"
            ),
            EventLogEntry(
                timestamp="10:15:32",
                event_type="SEED_GERMINATED",
                env_id=1,
                message="A1 germinated (dense_m)"
            ),
        ]
    )
    result = widget.render()

    # Should return Group of Text lines
    assert isinstance(result, Group)

    # Render to text and check content
    rendered = render_to_text(result)
    assert "10:15:30" in rendered
    assert "10:15:31" in rendered
    assert "10:15:32" in rendered
    assert "WAIT r=+0.500" in rendered
    assert "A1 germinated" in rendered


def test_event_log_max_events_limit():
    """Test that only max_events are shown."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry

    events = [
        EventLogEntry(
            timestamp=f"10:00:{i:02d}",
            event_type="EPOCH_COMPLETED",
            env_id=0,
            message=f"Event {i}"
        )
        for i in range(30)  # 30 events
    ]

    widget = EventLog(max_events=10)  # Only show 10
    widget._snapshot = SanctumSnapshot(event_log=events)
    result = widget.render()

    rendered = render_to_text(result)
    # Should only have last 10 events (20-29)
    assert "Event 20" in rendered
    assert "Event 29" in rendered
    assert "Event 19" not in rendered  # Should be cut off


def test_event_log_color_mapping():
    """Test event types get correct colors."""
    from esper.karn.sanctum.widgets.event_log import _EVENT_COLORS

    # Verify color mapping exists for expected event types
    assert "TRAINING_STARTED" in _EVENT_COLORS
    assert "EPOCH_COMPLETED" in _EVENT_COLORS
    assert "PPO_UPDATE_COMPLETED" in _EVENT_COLORS
    assert "REWARD_COMPUTED" in _EVENT_COLORS
    assert "SEED_GERMINATED" in _EVENT_COLORS
    assert "SEED_STAGE_CHANGED" in _EVENT_COLORS
    assert "SEED_FOSSILIZED" in _EVENT_COLORS
    assert "SEED_CULLED" in _EVENT_COLORS
    assert "BATCH_COMPLETED" in _EVENT_COLORS


def test_event_log_episode_grouping():
    """EventLog should group events by episode with separators."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry
    from rich.console import Group

    widget = EventLog()
    widget._snapshot = SanctumSnapshot(
        event_log=[
            EventLogEntry(
                timestamp="12:33:12",
                event_type="BATCH_COMPLETED",
                env_id=None,
                message="Episode 4 complete",
                episode=4,
                relative_time="(1m)",
            ),
            EventLogEntry(
                timestamp="12:34:51",
                event_type="REWARD_COMPUTED",
                env_id=0,
                message="WAIT r=+0.12",
                episode=5,
                relative_time="(7s)",
            ),
            EventLogEntry(
                timestamp="12:34:56",
                event_type="SEED_GERMINATED",
                env_id=0,
                message="seed_0a3f germinated",
                episode=5,
                relative_time="(2s)",
            ),
        ]
    )

    result = widget.render()
    assert isinstance(result, Group)

    # Render to text and check for episode separator
    rendered = render_to_text(result)
    assert "Episode 5" in rendered
    assert "seed_0a3f germinated" in rendered
    assert "WAIT r=+0.12" in rendered
    assert "Episode 4 complete" in rendered
    assert "(2s)" in rendered
    assert "(7s)" in rendered
    assert "(1m)" in rendered


def test_event_log_color_coding():
    """EventLog should color-code events by type."""
    from esper.karn.sanctum.widgets.event_log import EventLog

    widget = EventLog()

    # Test color mapping
    assert widget._get_event_color("SEED_GERMINATED") == "bright_yellow"
    assert widget._get_event_color("SEED_FOSSILIZED") == "bright_green"
    assert widget._get_event_color("SEED_CULLED") == "bright_red"
    assert widget._get_event_color("REWARD_COMPUTED") == "bright_cyan"
    assert widget._get_event_color("BATCH_COMPLETED") == "bright_blue"
    assert widget._get_event_color("UNKNOWN_EVENT") == "white"  # Default


def test_event_log_emoji_mapping():
    """EventLog should return correct emoji for event types."""
    from esper.karn.sanctum.widgets.event_log import EventLog

    widget = EventLog()

    # Test emoji mapping
    assert widget._get_event_emoji("SEED_GERMINATED") == "🌱"
    assert widget._get_event_emoji("SEED_FOSSILIZED") == "✅"
    assert widget._get_event_emoji("SEED_CULLED") == "⚠️"
    assert widget._get_event_emoji("REWARD_COMPUTED") == "📊"
    assert widget._get_event_emoji("BATCH_COMPLETED") == "🏆"
    assert widget._get_event_emoji("UNKNOWN_EVENT") == ""  # No emoji for unknown
