"""EventLogDetail modal - Raw event log viewer.

Shows unprocessed event entries with all fields visible.
No rollup, no formatting compression - just raw data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EventLogEntry


class EventLogDetail(ModalScreen[None]):
    """Modal showing raw event log entries.

    Displays all event fields without rollup or formatting compression.
    Useful for debugging and detailed inspection.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    def on_click(self) -> None:
        """Dismiss modal on click anywhere."""
        self.dismiss()

    DEFAULT_CSS = """
    EventLogDetail {
        align: center middle;
        background: $surface-darken-1 80%;
    }

    EventLogDetail > #event-detail-container {
        width: 90%;
        height: 85%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    EventLogDetail > #event-detail-container > #event-detail-scroll {
        height: 1fr;
    }

    EventLogDetail #event-detail-header {
        height: auto;
        margin-bottom: 1;
    }
    """

    def __init__(self, events: list["EventLogEntry"]) -> None:
        """Initialize with event list.

        Args:
            events: List of EventLogEntry to display.
        """
        super().__init__()
        self._events = events

    def compose(self):
        """Compose the modal layout."""
        with Container(id="event-detail-container"):
            yield Static(self._render_header(), id="event-detail-header")
            with VerticalScroll(id="event-detail-scroll"):
                yield Static(self._render_events(), id="event-detail-content")

    def _render_header(self) -> Text:
        """Render the modal header."""
        header = Text()
        header.append("RAW EVENT LOG", style="bold cyan")
        header.append(f"  ({len(self._events)} events)", style="dim")
        header.append("\n")
        header.append("Press ", style="dim")
        header.append("ESC", style="cyan")
        header.append(" or ", style="dim")
        header.append("q", style="cyan")
        header.append(" to close", style="dim")
        return header

    def _render_events(self) -> Table:
        """Render all events in a table with full field visibility."""
        table = Table(
            expand=True,
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
        )

        # All EventLogEntry fields including metadata
        table.add_column("Time", style="dim", width=10)
        table.add_column("Type", style="bright_white", width=22)
        table.add_column("Env", style="bright_blue", width=4, justify="right")
        table.add_column("Ep", style="dim", width=4, justify="right")
        table.add_column("Message", style="white", width=16)
        table.add_column("Details", style="cyan", ratio=1)

        # Show events in reverse chronological order (newest first)
        for entry in reversed(self._events):
            env_str = f"{entry.env_id:02d}" if entry.env_id is not None else "--"
            # Format metadata as key=value pairs
            details = " ".join(
                f"{k}={v}" for k, v in entry.metadata.items()
            ) if entry.metadata else ""
            table.add_row(
                entry.timestamp,
                entry.event_type,
                env_str,
                str(entry.episode),
                entry.message,
                details,
            )

        return table
