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
    assert widget._max_lines == 30  # Default max lines
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
    """Test render with events shows formatted log.

    NOTE: The refactored EventLog is append-only and groups events by second.
    It waits for a second to COMPLETE before showing events, so we can't test
    live rendering without mocking datetime. We test the line data population instead.
    """
    from esper.karn.sanctum.widgets.event_log import EventLog, _EVENT_COLORS
    from esper.karn.sanctum.schema import EventLogEntry
    from unittest.mock import patch
    from datetime import datetime, timezone

    widget = EventLog(max_lines=10)

    # Mock datetime to make all test timestamps "complete" (in the past)
    mock_now = datetime(2024, 1, 1, 10, 15, 40, tzinfo=timezone.utc)

    snapshot = SanctumSnapshot(
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

    with patch("esper.karn.sanctum.widgets.event_log.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        widget.update_snapshot(snapshot)

    # Should have line data for the 3 seconds (each event type per second = 1 line each)
    assert len(widget._line_data) == 3

    # Verify the event types are in the line data
    all_text = " ".join(left.plain for left, _ in widget._line_data)
    assert "TRAINING_STARTED" in all_text or "TRAINING" in all_text
    assert "REWARD" in all_text
    assert "GERMINATED" in all_text


def test_event_log_max_lines_limit():
    """Test that only max_lines are kept in line data."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry
    from unittest.mock import patch
    from datetime import datetime, timezone

    # Create 30 events at different seconds
    events = [
        EventLogEntry(
            timestamp=f"10:00:{i:02d}",
            event_type="EPOCH_COMPLETED",
            env_id=0,
            message=f"Event {i}"
        )
        for i in range(30)  # 30 events at 30 different seconds
    ]

    widget = EventLog(max_lines=10)  # Only keep 10 lines

    # Mock datetime to make all timestamps "complete"
    mock_now = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)

    with patch("esper.karn.sanctum.widgets.event_log.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        widget.update_snapshot(SanctumSnapshot(event_log=events))

    # Should have trimmed to max_lines
    assert len(widget._line_data) == 10


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
    assert "SEED_PRUNED" in _EVENT_COLORS
    assert "BATCH_EPOCH_COMPLETED" in _EVENT_COLORS


def test_event_log_groups_by_second():
    """EventLog groups events by timestamp second, showing counts for duplicates."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry
    from unittest.mock import patch
    from datetime import datetime, timezone

    widget = EventLog()

    # Multiple events in the same second should be counted
    snapshot = SanctumSnapshot(
        event_log=[
            EventLogEntry(
                timestamp="12:33:12",
                event_type="REWARD_COMPUTED",
                env_id=0,
                message="WAIT r=+0.12",
            ),
            EventLogEntry(
                timestamp="12:33:12",
                event_type="REWARD_COMPUTED",
                env_id=1,
                message="WAIT r=+0.15",
            ),
            EventLogEntry(
                timestamp="12:33:12",
                event_type="REWARD_COMPUTED",
                env_id=2,
                message="WAIT r=+0.18",
            ),
            EventLogEntry(
                timestamp="12:33:13",
                event_type="SEED_GERMINATED",
                env_id=0,
                message="seed germinated",
            ),
        ]
    )

    # Mock datetime to make all timestamps "complete"
    mock_now = datetime(2024, 1, 1, 12, 33, 20, tzinfo=timezone.utc)

    with patch("esper.karn.sanctum.widgets.event_log.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        widget.update_snapshot(snapshot)

    # Should have 2 lines: one for REWARD_COMPUTED ×3, one for SEED_GERMINATED
    assert len(widget._line_data) == 2

    # Check first line has count indicator
    first_line_text = widget._line_data[0][0].plain
    assert "×3" in first_line_text  # 3 REWARD_COMPUTED events

    # Check env IDs are captured
    first_line_envs = widget._line_data[0][1]
    assert "0" in first_line_envs
    assert "1" in first_line_envs
    assert "2" in first_line_envs


def test_event_log_color_coding():
    """EventLog uses module-level _EVENT_COLORS for color mapping."""
    from esper.karn.sanctum.widgets.event_log import _EVENT_COLORS

    # Test color mapping directly from module constant
    assert _EVENT_COLORS["SEED_GERMINATED"] == "bright_yellow"
    assert _EVENT_COLORS["SEED_FOSSILIZED"] == "bright_green"
    assert _EVENT_COLORS["SEED_PRUNED"] == "bright_red"
    assert _EVENT_COLORS["REWARD_COMPUTED"] == "bright_cyan"
    assert _EVENT_COLORS["BATCH_EPOCH_COMPLETED"] == "bright_blue"

    # Unknown events default to "white" via .get() in implementation
    assert _EVENT_COLORS.get("UNKNOWN_EVENT", "white") == "white"


def test_event_log_uses_colors_in_rendering():
    """EventLog applies colors from _EVENT_COLORS when rendering lines."""
    from esper.karn.sanctum.widgets.event_log import EventLog, _EVENT_COLORS
    from esper.karn.sanctum.schema import EventLogEntry
    from unittest.mock import patch
    from datetime import datetime, timezone

    widget = EventLog()

    snapshot = SanctumSnapshot(
        event_log=[
            EventLogEntry(
                timestamp="12:00:00",
                event_type="SEED_GERMINATED",
                env_id=0,
                message="seed germinated",
            ),
        ]
    )

    mock_now = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    with patch("esper.karn.sanctum.widgets.event_log.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        widget.update_snapshot(snapshot)

    # Line data should contain styled text
    assert len(widget._line_data) == 1
    left_text, _ = widget._line_data[0]

    # The text should contain GERMINATED (after SEED_ is stripped)
    assert "GERMINATED" in left_text.plain
