"""EventLog widget - Recent event feed with color coding.

Direct port of the event log section from tui.py.
Shows recent telemetry events with color coding by event type.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EventLogEntry, SanctumSnapshot


# Event type color mapping (matches TUIOutput)
_EVENT_COLORS: dict[str, str] = {
    "TRAINING_STARTED": "bright_green",
    "EPOCH_COMPLETED": "bright_blue",
    "PPO_UPDATE_COMPLETED": "bright_magenta",
    "REWARD_COMPUTED": "bright_cyan",
    "SEED_GERMINATED": "bright_yellow",
    "SEED_STAGE_CHANGED": "bright_white",
    "SEED_FOSSILIZED": "bright_green",
    "SEED_CULLED": "bright_red",
    "BATCH_COMPLETED": "bright_blue",
}


class EventLog(Static):
    """Event log widget showing recent telemetry events.

    Displays recent events with:
    - Timestamp (HH:MM:SS)
    - Event type (color coded)
    - Env ID (if applicable)
    - Message

    Events are color coded by type and scroll automatically
    to show the most recent events at the bottom.
    """

    def __init__(self, max_events: int = 20, **kwargs) -> None:
        """Initialize EventLog widget.

        Args:
            max_events: Maximum events to display (default: 20)
        """
        super().__init__(**kwargs)
        self._max_events = max_events
        self._snapshot: SanctumSnapshot | None = None

    def compose(self):
        """Compose the widget layout."""
        yield Static("Event Log", classes="panel-title")

    def on_mount(self) -> None:
        """Called when widget mounts."""
        self.update("[dim]Waiting for events...[/dim]")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        self._snapshot = snapshot

        # Render event log
        if not snapshot.event_log:
            self.update("[dim]No events yet[/dim]")
            return

        # Take the last N events
        events = list(snapshot.event_log)[-self._max_events :]

        # Build event lines with color coding
        lines = []
        for entry in events:
            color = _EVENT_COLORS.get(entry.event_type, "white")

            # Format: [timestamp] [ENV:id] message
            text = Text()
            text.append(f"[{entry.timestamp}] ", style="dim")

            if entry.env_id is not None:
                text.append(f"ENV:{entry.env_id:02d} ", style="bright_blue")

            text.append(entry.message, style=color)
            lines.append(text)

        # Render as a group
        self.update(Group(*lines))
