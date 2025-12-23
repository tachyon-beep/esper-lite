"""EventLog widget - Recent event feed with color coding and global rollup.

Enhanced design:
- Only shows last 7 seconds of events (click for full history)
- Color-coded by event type (green=lifecycle, cyan=tamiyo, yellow=warning, red=error)
- Compact timestamps (:SS, MM:SS on minute change)
- ALL identical messages rolled up globally with ×N suffix
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
    """Event log widget showing recent telemetry events.

    Enhanced features:
    - Only shows last 7 seconds of events (reduces noise)
    - Compact timestamps (:SS, MM:SS on minute change)
    - Color coding by event type
    - ALL identical messages rolled up globally (×N suffix)
    - Click anywhere to open full event history modal
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
        """Render the event log with global message rollup.

        Format optimized for fast-scrolling real-time events:
        - Only shows events from last 7 seconds (click for full history)
        - Compact time: :SS (shows MM:SS on minute change)
        - Compact env: just the number (00, 01, etc.)
        - ALL identical messages rolled up with ×N suffix (not just consecutive)
        - Ordered by most recent occurrence
        """
        from datetime import datetime

        if self._snapshot is None or not self._snapshot.event_log:
            return Text("Waiting for events...", style="dim")

        # Filter to events from last 7 seconds only
        now = datetime.now()
        max_age_seconds = 7

        recent_events = []
        for entry in self._snapshot.event_log:
            # Parse timestamp "HH:MM:SS" and compare with current time
            try:
                parts = entry.timestamp.split(":")
                if len(parts) == 3:
                    event_time = now.replace(
                        hour=int(parts[0]),
                        minute=int(parts[1]),
                        second=int(parts[2]),
                        microsecond=0,
                    )
                    # Handle midnight wraparound (event time > now means yesterday)
                    if event_time > now:
                        continue  # Skip events from "tomorrow" (actually yesterday)
                    age = (now - event_time).total_seconds()
                    if age <= max_age_seconds:
                        recent_events.append(entry)
            except (ValueError, IndexError):
                # If we can't parse, include the event
                recent_events.append(entry)

        if not recent_events:
            return Text("No recent events (click for history)", style="dim")

        events = recent_events[-self._max_events:]

        # Global rollup: count all occurrences of each unique message
        # Track most recent entry and count for each message
        message_counts: dict[str, tuple[EventLogEntry, int]] = {}
        for entry in events:
            if entry.message in message_counts:
                # Keep the most recent entry (latest in list), increment count
                _, count = message_counts[entry.message]
                message_counts[entry.message] = (entry, count + 1)
            else:
                message_counts[entry.message] = (entry, 1)

        # Sort by most recent timestamp (descending), then render
        rolled_events = sorted(
            message_counts.values(),
            key=lambda x: x[0].timestamp,
            reverse=True,
        )

        # Render rolled-up events (newest first)
        lines = []
        last_minute = None

        for entry, count in rolled_events:
            color = self._get_event_color(entry.event_type)
            text = Text()

            # Compact timestamp: show MM:SS on minute change, otherwise just :SS
            parts = entry.timestamp.split(":")
            if len(parts) == 3:
                current_minute = parts[1]
                if current_minute != last_minute:
                    text.append(f"{parts[1]}:{parts[2]} ", style="dim")
                    last_minute = current_minute
                else:
                    text.append(f":{parts[2]} ", style="dim")
            else:
                text.append(f"{entry.timestamp} ", style="dim")

            # Compact env ID: only show if single occurrence
            if count == 1 and entry.env_id is not None:
                text.append(f"{entry.env_id:02d} ", style="bright_blue")

            text.append(entry.message, style=color)

            # Add rollup count if > 1
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
