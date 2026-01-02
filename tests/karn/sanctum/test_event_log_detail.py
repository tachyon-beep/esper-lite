"""Tests for EventLogDetail modal.

Tests cover:
- EventLogDetail: Modal for viewing raw event log entries
- GLOBAL_EVENT_ENV_DISPLAY constant for events without env_id
"""
from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table

from esper.karn.sanctum.schema import EventLogEntry
from esper.karn.sanctum.widgets.event_log_detail import (
    EventLogDetail,
    GLOBAL_EVENT_ENV_DISPLAY,
)


def render_to_text(renderable) -> str:
    """Helper to render a Rich renderable to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


class TestEventLogDetailRendering:
    """Test EventLogDetail renders all fields correctly."""

    def test_modal_creation(self):
        """Modal should be creatable with event list."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="PPO_UPDATE",
                env_id=None,
                episode=1,
                message="Test message",
                metadata={},
            )
        ]
        modal = EventLogDetail(events)
        assert modal is not None
        assert modal._events == events

    def test_renders_header_with_event_count(self):
        """Header should show event count."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="PPO_UPDATE",
                env_id=None,
                episode=1,
                message="Test",
                metadata={},
            )
            for _ in range(5)
        ]
        modal = EventLogDetail(events)
        header = modal._render_header()
        header_text = header.plain

        assert "RAW EVENT LOG" in header_text
        assert "5 events" in header_text

    def test_renders_events_table(self):
        """Events should render as a table."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="GERMINATE",
                env_id=0,
                episode=10,
                message="Seed created",
                metadata={"slot": "r0c0", "blueprint": "conv3x3"},
            )
        ]
        modal = EventLogDetail(events)
        table = modal._render_events()

        assert isinstance(table, Table)
        # Table has 6 columns: Time, Type, Env, Ep, Message, Details
        assert len(table.columns) == 6

    def test_env_id_none_shows_global_constant(self):
        """Events without env_id should show GLOBAL_EVENT_ENV_DISPLAY."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="PPO_UPDATE",
                env_id=None,  # Global event
                episode=1,
                message="PPO step",
                metadata={},
            )
        ]
        modal = EventLogDetail(events)
        table = modal._render_events()
        output = render_to_text(table)

        # Should show "--" for global events
        assert GLOBAL_EVENT_ENV_DISPLAY in output

    def test_env_id_present_shows_formatted_id(self):
        """Events with env_id should show formatted ID."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="GERMINATE",
                env_id=5,
                episode=1,
                message="Seed created",
                metadata={},
            )
        ]
        modal = EventLogDetail(events)
        table = modal._render_events()
        output = render_to_text(table)

        # Should show "05" for env_id=5
        assert "05" in output

    def test_events_in_reverse_chronological_order(self):
        """Events should be displayed newest first."""
        events = [
            EventLogEntry(
                timestamp="12:00:00",
                event_type="FIRST",
                env_id=0,
                episode=1,
                message="First",
                metadata={},
            ),
            EventLogEntry(
                timestamp="12:00:01",
                event_type="SECOND",
                env_id=0,
                episode=1,
                message="Second",
                metadata={},
            ),
            EventLogEntry(
                timestamp="12:00:02",
                event_type="THIRD",
                env_id=0,
                episode=1,
                message="Third",
                metadata={},
            ),
        ]
        modal = EventLogDetail(events)
        table = modal._render_events()
        output = render_to_text(table)

        # THIRD should appear before FIRST in output (reverse order)
        third_pos = output.find("THIRD")
        first_pos = output.find("FIRST")
        assert third_pos < first_pos, "Newest event should appear first"

    def test_metadata_rendered_as_key_value_pairs(self):
        """Event metadata should render as key=value pairs."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="GERMINATE",
                env_id=0,
                episode=1,
                message="Test",
                metadata={"slot": "r0c0", "blueprint": "conv3x3"},
            )
        ]
        modal = EventLogDetail(events)
        table = modal._render_events()
        output = render_to_text(table)

        assert "slot=r0c0" in output
        assert "blueprint=conv3x3" in output


class TestGlobalEventEnvDisplay:
    """Test the GLOBAL_EVENT_ENV_DISPLAY constant."""

    def test_constant_value(self):
        """GLOBAL_EVENT_ENV_DISPLAY should be '--'."""
        assert GLOBAL_EVENT_ENV_DISPLAY == "--"

    def test_constant_used_for_none_env_id(self):
        """Constant should be used when env_id is None."""
        events = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="BATCH_END",
                env_id=None,
                episode=1,
                message="Batch complete",
                metadata={},
            )
        ]
        modal = EventLogDetail(events)
        table = modal._render_events()
        output = render_to_text(table)

        # The constant value should appear in output
        assert GLOBAL_EVENT_ENV_DISPLAY in output
