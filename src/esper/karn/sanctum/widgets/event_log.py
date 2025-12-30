"""EventLog widget - Append-only scrolling log.

Architecture:
- Maintains internal list of line data (append-only, never recalculated)
- Tracks which seconds have been processed
- On update: checks for NEW completed seconds, aggregates them, appends line data
- On render: formats line data using actual widget width for right-justification

This is a LOG, not a reactive view. Line data stays put once added.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EventLogEntry, SanctumSnapshot


# Event type color mapping
_EVENT_COLORS: dict[str, str] = {
    # Seed lifecycle
    "SEED_GERMINATED": "bright_yellow",
    "SEED_STAGE_CHANGED": "bright_white",
    "SEED_FOSSILIZED": "bright_green",
    "SEED_PRUNED": "bright_red",
    # Tamiyo actions
    "REWARD_COMPUTED": "bright_cyan",
    # Training events
    "TRAINING_STARTED": "bright_green",
    "EPOCH_COMPLETED": "bright_blue",
    "PPO_UPDATE_COMPLETED": "bright_magenta",
    "BATCH_EPOCH_COMPLETED": "bright_blue",
}

# Max envs to show before truncating with "+"
_MAX_ENVS_SHOWN = 3


class EventLog(Static):
    """Append-only scrolling event log.

    - Waits for each second to COMPLETE before showing its events
    - Each event type within a second gets its own line with count
    - Shows contributing env IDs right-justified to panel edge
    - Lines are appended at the bottom; existing lines never change
    - Click anywhere to open raw event detail modal
    """

    class DetailRequested(Message):
        """Posted when user clicks to view raw event log."""

        def __init__(self, events: list["EventLogEntry"]) -> None:
            super().__init__()
            self.events = events

    def __init__(self, max_lines: int = 36, **kwargs: Any) -> None:
        """Initialize EventLog widget.

        Args:
            max_lines: Maximum lines to display (oldest trimmed).
        """
        super().__init__(**kwargs)
        self._max_lines = max_lines
        self.border_title = "EVENTS [↵]"

        # === APPEND-ONLY STATE ===
        # Line data: list of (timestamp, event_label, color, count, env_str) tuples
        # Timestamp formatting is done at render time, not append time
        self._line_data: list[tuple[str, str, str, int, str]] = []
        # Seconds we've already processed (never process twice)
        self._processed_seconds: set[str] = set()
        # Keep reference to snapshot for click handler
        self._snapshot: SanctumSnapshot | None = None
        # Track trimmed lines for scroll indicator
        self._trimmed_count: int = 0

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Process snapshot and append any newly completed seconds.

        This is where the work happens - NOT in render().
        """
        self._snapshot = snapshot

        if not snapshot.event_log:
            return

        # What second is it NOW? (UTC)
        now = datetime.now(timezone.utc)
        current_second = now.strftime("%H:%M:%S")

        # Group events by timestamp, excluding current second (still accumulating)
        # Structure: {timestamp: {event_type: set[env_id]}}
        second_groups: dict[str, dict[str, set[int | None]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for entry in snapshot.event_log:
            ts = entry.timestamp
            # Skip current second (not complete yet)
            if ts == current_second:
                continue
            # Skip already-processed seconds
            if ts in self._processed_seconds:
                continue
            second_groups[ts][entry.event_type].add(entry.env_id)

        if not second_groups:
            return

        # Process new completed seconds in chronological order
        for timestamp in sorted(second_groups.keys()):
            type_envs = second_groups[timestamp]
            self._append_second(timestamp, type_envs)
            self._processed_seconds.add(timestamp)

        # Trim if we exceed max lines
        if len(self._line_data) > self._max_lines:
            excess = len(self._line_data) - self._max_lines
            self._trimmed_count += excess
            self._line_data = self._line_data[-self._max_lines:]

        # Update border title with scroll indicator
        if self._trimmed_count > 0:
            self.border_title = f"EVENTS [↵] ↑{self._trimmed_count}"
        else:
            self.border_title = "EVENTS [↵]"

        # Trigger re-render
        self.refresh()

    def _append_second(
        self, timestamp: str, type_envs: dict[str, set[int | None]]
    ) -> None:
        """Append line data for a completed second.

        Each event type gets its own line with count and contributing envs.
        Stores raw data - formatting happens at render time.
        """
        for event_type in sorted(type_envs.keys()):
            env_ids = type_envs[event_type]
            count = len(env_ids)

            # Event type label (shortened for display)
            color = _EVENT_COLORS.get(event_type, "white")
            label = (
                event_type
                .replace("SEED_", "")
                .replace("_COMPLETED", "")
                .replace("_COMPUTED", "")
            )

            # Build env list string
            env_str = self._format_env_list(env_ids)

            # Store raw data tuple: (timestamp, label, color, count, env_str)
            self._line_data.append((timestamp, label, color, count, env_str))

    def _format_env_list(self, env_ids: set[int | None]) -> str:
        """Format env IDs for display, e.g., '0 1 2 3 +'.

        Returns empty string for global events (all None).
        """
        # Filter out None (global events)
        real_ids = sorted(eid for eid in env_ids if eid is not None)
        if not real_ids:
            return ""

        if len(real_ids) <= _MAX_ENVS_SHOWN:
            return " ".join(str(eid) for eid in real_ids)
        else:
            shown = " ".join(str(eid) for eid in real_ids[:_MAX_ENVS_SHOWN])
            return f"{shown} +"

    def render(self) -> RenderableType:
        """Render the log with proper right-justification.

        Uses actual widget width to align env IDs to right edge.
        Timestamp rules (all stored as HH:MM:SS):
        - Top line (i==0): shows full HH:MM:SS
        - First event of each new minute: shows full HH:MM:SS
        - All other lines: shows only :SS (with padding for alignment)
        """
        if not self._line_data:
            return Text("Waiting for events...", style="dim")

        # Get actual widget width (subtract 2 for border padding)
        width = self.size.width - 2 if self.size.width > 4 else 30

        lines = []
        last_minute: str | None = None

        for i, (timestamp, label, color, count, env_str) in enumerate(self._line_data):
            parts = timestamp.split(":")
            if len(parts) != 3:
                continue

            hour_minute = f"{parts[0]}:{parts[1]}"
            second = parts[2]

            line = Text()

            # Determine if this line shows full timestamp or abbreviated
            # Full: top line (i==0) OR first event of a new minute
            show_full = (i == 0) or (hour_minute != last_minute)

            if show_full:
                # Full format: " HH:MM:SS " = 10 chars (1 space indent + timestamp + space)
                line.append(f" {timestamp} ", style="bold cyan")
            else:
                # Short format: "      :SS " = 9 chars (6 spaces + :SS + space)
                line.append(f"      :{second} ", style="dim")

            # Always update last_minute for tracking
            last_minute = hour_minute

            # Event label
            line.append(label, style=color)

            # Count if > 1
            if count > 1:
                line.append(f" ×{count}", style="dim")

            # Right-justify env IDs
            if env_str:
                left_len = len(line.plain)
                right_len = len(env_str)
                padding = max(1, width - left_len - right_len)
                line.append(" " * padding)
                line.append(env_str, style="dim")

            lines.append(line)

        return Group(*lines)

    def on_click(self) -> None:
        """Handle click to open raw event detail modal."""
        if self._snapshot is None or not self._snapshot.event_log:
            return
        self.post_message(self.DetailRequested(list(self._snapshot.event_log)))
