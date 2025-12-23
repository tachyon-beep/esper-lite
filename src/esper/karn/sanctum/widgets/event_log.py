"""EventLog widget - Recent event feed with color coding and episode grouping.

Enhanced design:
- Full-width rows using all horizontal space
- Color-coded by event type (green=lifecycle, cyan=tamiyo, yellow=warning, red=error)
- Timestamp + relative time "(2s ago)"
- Episode grouping with visual separators
- Emoji prefixes for quick scanning
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
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
    - Full-width rows with timestamp + relative time
    - Color coding by event type
    - Episode grouping with separators
    - Emoji prefixes for quick scanning
    """

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
        """Render the event log with episode grouping.

        Format optimized for fast-scrolling real-time events:
        - Compact time: :SS (shows MM:SS on minute change)
        - Compact env: just the number (00, 01, etc.)
        - No relative time (events scroll too fast for it to matter)
        """
        if self._snapshot is None or not self._snapshot.event_log:
            return Text("Waiting for events...", style="dim")

        events = list(self._snapshot.event_log)[-self._max_events:]
        lines = []
        last_episode = None
        last_minute = None

        for entry in events:
            # Episode separator (short, won't stretch container)
            if entry.episode != last_episode and last_episode is not None:
                separator = Text(f"─── Episode {entry.episode} ───", style="dim")
                lines.append(separator)
            last_episode = entry.episode

            # Event line
            color = self._get_event_color(entry.event_type)

            text = Text()

            # Compact timestamp: show MM:SS on minute change, otherwise just :SS
            # entry.timestamp is "HH:MM:SS"
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

            # Compact env ID: just the number
            if entry.env_id is not None:
                text.append(f"{entry.env_id:02d} ", style="bright_blue")

            text.append(entry.message, style=color)

            lines.append(text)

        return Group(*lines)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        self._snapshot = snapshot
        self.refresh()  # Trigger re-render
