"""EventLog widget - Scrolling event feed grouped by second.

Enhanced design:
- Scrolls chronologically (newest at top)
- Each second = one line with all event types rolled up
- Format: "12:45 REWARD×4 GERMINATED×2 STAGE_CHANGED"
- Click for full unrolled history in modal
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
    """Scrolling event log grouped by second.

    Each second gets one line showing all event types that occurred:
    - "12:45 REWARD×4 GERMINATED×2" (multiple types, counts if >1)
    - Scrolls chronologically, newest at top
    - Click anywhere to open full unrolled history modal
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
        """Render the event log as scrolling list grouped by second.

        Format: Each second gets one line with all event types rolled up.
        - Scrolls chronologically (newest at top)
        - Events within same second grouped: "12:45 Reward ×4 Germinated ×2"
        - Click for full unrolled history
        """
        from collections import defaultdict
        from datetime import datetime, timezone

        if self._snapshot is None or not self._snapshot.event_log:
            return Text("Waiting for events...", style="dim")

        # Filter to events from last 60 seconds
        now = datetime.now(timezone.utc)
        max_age_seconds = 60

        recent_events = []
        for entry in self._snapshot.event_log:
            try:
                parts = entry.timestamp.split(":")
                if len(parts) == 3:
                    event_time = now.replace(
                        hour=int(parts[0]),
                        minute=int(parts[1]),
                        second=int(parts[2]),
                        microsecond=0,
                    )
                    if event_time > now:
                        continue
                    age = (now - event_time).total_seconds()
                    if age <= max_age_seconds:
                        recent_events.append(entry)
            except (ValueError, IndexError):
                recent_events.append(entry)

        if not recent_events:
            return Text("No recent events (click for history)", style="dim")

        # Group by timestamp (second), then count event types within each second
        # Structure: {timestamp: {event_type: count}}
        second_groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for entry in recent_events:
            second_groups[entry.timestamp][entry.event_type] += 1

        # Sort by timestamp descending (newest first)
        sorted_seconds = sorted(second_groups.keys(), reverse=True)

        # Render each second as one line
        lines = []
        last_minute = None

        for timestamp in sorted_seconds[-self._max_events:]:
            type_counts = second_groups[timestamp]
            text = Text()

            # Timestamp with minute change detection
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

            # All event types for this second, space-separated
            first = True
            for event_type, count in sorted(type_counts.items()):
                if not first:
                    text.append(" ", style="dim")
                first = False

                color = self._get_event_color(event_type)
                # Short label for event type
                label = event_type.replace("SEED_", "").replace("_COMPLETED", "").replace("_COMPUTED", "")
                text.append(label, style=color)
                if count > 1:
                    text.append(f"×{count}", style="dim")

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
