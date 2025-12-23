"""EventLog widget - Normal scrolling log with consecutive rollup.

Like a regular log:
- New entries at bottom, scrolls up
- Each event type on its own line
- Consecutive identical events rolled up: "REWARD ×6"
- Click for full unrolled history
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EventLogEntry, SanctumSnapshot


# Event type color mapping
_EVENT_COLORS: dict[str, str] = {
    # Seed lifecycle (green family)
    "SEED_GERMINATED": "bright_yellow",
    "SEED_STAGE_CHANGED": "bright_white",
    "SEED_FOSSILIZED": "bright_green",
    "SEED_PRUNED": "bright_red",
    # Tamiyo actions (cyan)
    "REWARD_COMPUTED": "bright_cyan",
    # Training events (blue)
    "TRAINING_STARTED": "bright_green",
    "EPOCH_COMPLETED": "bright_blue",
    "PPO_UPDATE_COMPLETED": "bright_magenta",
    "BATCH_EPOCH_COMPLETED": "bright_blue",
}

# Event type emoji mapping (disabled - causes terminal rendering artifacts)
_EVENT_EMOJI: dict[str, str] = {
    # "SEED_GERMINATED": "🌱",
    # "SEED_FOSSILIZED": "✅",
    # "SEED_PRUNED": "⚠️",
    # "REWARD_COMPUTED": "📊",
    # "BATCH_EPOCH_COMPLETED": "🏆",
}


class EventLog(Static):
    """Normal scrolling log with consecutive rollup.

    Like a regular log file:
    - New entries appear at bottom
    - Each event type on its own line
    - Consecutive identical events rolled up: "REWARD ×6"
    - Click anywhere for full unrolled history
    """

    class DetailRequested(Message):
        """Posted when user clicks to view raw event log."""

        def __init__(self, events: list["EventLogEntry"]) -> None:
            super().__init__()
            self.events = events

    def __init__(self, max_events: int = 30, **kwargs) -> None:
        """Initialize EventLog widget.

        Args:
            max_events: Maximum events to display (default: 30)
        """
        super().__init__(**kwargs)
        self._max_events = max_events
        self._snapshot: SanctumSnapshot | None = None
        self.border_title = "EVENTS"

    def _get_event_color(self, event_type: str) -> str:
        """Get color for event type."""
        return _EVENT_COLORS.get(event_type, "white")

    def _get_event_emoji(self, event_type: str) -> str:
        """Get emoji for event type."""
        return _EVENT_EMOJI.get(event_type, "")

    def render(self):
        """Render scrolling log with consecutive event rollup.

        - Normal scrolling log (new entries at bottom)
        - Consecutive identical events rolled up: "REWARD ×6"
        - Each event type on its own line
        - Click for full unrolled history
        """
        if self._snapshot is None or not self._snapshot.event_log:
            return Text("Waiting for events...", style="dim")

        events = list(self._snapshot.event_log)

        if not events:
            return Text("No events (click for history)", style="dim")

        # Roll up CONSECUTIVE identical events (same event_type)
        rolled: list[tuple[EventLogEntry, int]] = []
        for entry in events:
            if rolled and rolled[-1][0].event_type == entry.event_type:
                # Same as previous - increment count, keep latest entry
                _, count = rolled[-1]
                rolled[-1] = (entry, count + 1)
            else:
                rolled.append((entry, 1))

        # Take last N entries, render oldest to newest (new at bottom)
        rolled = rolled[-self._max_events:]

        lines = []
        last_minute = None

        for entry, count in rolled:
            text = Text()

            # Timestamp: "MM:SS " or "  :SS "
            parts = entry.timestamp.split(":")
            if len(parts) == 3:
                current_minute = parts[1]
                if current_minute != last_minute:
                    text.append(f"{parts[1]}:{parts[2]} ", style="dim")
                    last_minute = current_minute
                else:
                    text.append(f"  :{parts[2]} ", style="dim")
            else:
                text.append(f"{entry.timestamp:>5} ", style="dim")

            # Event type (shortened)
            color = self._get_event_color(entry.event_type)
            label = entry.event_type.replace("SEED_", "").replace("_COMPLETED", "").replace("_COMPUTED", "")
            text.append(label, style=color)

            # Count if > 1
            if count > 1:
                text.append(f" ×{count}", style="dim")

            lines.append(text)

        return Group(*lines)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        self._snapshot = snapshot
        self.refresh()  # Trigger re-render

    def on_click(self) -> None:
        """Handle click to open raw event detail modal."""
        if self._snapshot is None or not self._snapshot.event_log:
            return
        # Pass all events (not just rendered ones) for full visibility
        self.post_message(self.DetailRequested(list(self._snapshot.event_log)))
