"""EventLog widget - Scrolling log that waits for each second to complete.

- Buffers events until the second is complete
- Then shows each event type on its own line with count
- New completed seconds appear at bottom
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
    """Scrolling log that waits for each second to complete.

    - Events from current second are buffered (not shown yet)
    - Once a second completes, all its events are shown
    - Each event type on its own line with count: "REWARD ×6"
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
        """Render scrolling log - wait for each second to complete before showing.

        - Group events by second
        - Only show COMPLETED seconds (not current second still accumulating)
        - Each event type on its own line with count
        - New completed seconds appear at bottom
        """
        from collections import defaultdict
        from datetime import datetime, timezone

        if self._snapshot is None or not self._snapshot.event_log:
            return Text("Waiting for events...", style="dim")

        events = list(self._snapshot.event_log)
        if not events:
            return Text("No events (click for history)", style="dim")

        # Get current second (we won't show events from this second yet)
        now = datetime.now(timezone.utc)
        current_second = now.strftime("%H:%M:%S")

        # Group events by timestamp second, then by event_type
        # Structure: {timestamp: {event_type: count}}
        second_groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for entry in events:
            # Skip events from current second (still accumulating)
            if entry.timestamp == current_second:
                continue
            second_groups[entry.timestamp][entry.event_type] += 1

        if not second_groups:
            return Text("Accumulating... (click for history)", style="dim")

        # Sort seconds chronologically, take last N
        sorted_seconds = sorted(second_groups.keys())[-self._max_events:]

        lines = []
        last_minute = None

        for timestamp in sorted_seconds:
            type_counts = second_groups[timestamp]

            # Each event type gets its own line
            for event_type in sorted(type_counts.keys()):
                count = type_counts[event_type]
                text = Text()

                # Timestamp: "MM:SS " or "  :SS "
                parts = timestamp.split(":")
                if len(parts) == 3:
                    current_minute = parts[1]
                    if current_minute != last_minute:
                        text.append(f"{parts[1]}:{parts[2]} ", style="dim")
                        last_minute = current_minute
                    else:
                        text.append(f"  :{parts[2]} ", style="dim")
                else:
                    text.append(f"{timestamp:>5} ", style="dim")

                # Event type (shortened)
                color = self._get_event_color(event_type)
                label = event_type.replace("SEED_", "").replace("_COMPLETED", "").replace("_COMPUTED", "")
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
