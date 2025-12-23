"""EventLog widget - Append-only scrolling log.

Architecture:
- Maintains internal list of rendered lines (append-only, never recalculated)
- Tracks which seconds have been processed
- On update: checks for NEW completed seconds, aggregates them, appends lines
- On render: just displays the lines list (no recalculation)

This is a LOG, not a reactive view. Lines stay put once added.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Group
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

# Max envs to show before truncating with "(+ N)"
_MAX_ENVS_SHOWN = 3


class EventLog(Static):
    """Append-only scrolling event log.

    - Waits for each second to COMPLETE before showing its events
    - Each event type within a second gets its own line with count
    - Shows contributing env IDs right-justified
    - Lines are appended at the bottom; existing lines never change
    - Click anywhere to open raw event detail modal
    """

    class DetailRequested(Message):
        """Posted when user clicks to view raw event log."""

        def __init__(self, events: list["EventLogEntry"]) -> None:
            super().__init__()
            self.events = events

    def __init__(self, max_lines: int = 30, **kwargs) -> None:
        """Initialize EventLog widget.

        Args:
            max_lines: Maximum lines to display (oldest trimmed).
        """
        super().__init__(**kwargs)
        self._max_lines = max_lines
        self.border_title = "EVENTS"

        # === APPEND-ONLY STATE ===
        # The rendered lines - this is the source of truth for display
        self._lines: list[Text] = []
        # Seconds we've already processed (never process twice)
        self._processed_seconds: set[str] = set()
        # Track last minute shown (for abbreviated timestamps)
        self._last_minute: str | None = None
        # Keep reference to snapshot for click handler
        self._snapshot: SanctumSnapshot | None = None

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
        # env_id can be None for global events (PPO, BATCH)
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
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines:]

        # Trigger re-render (just displays self._lines)
        self.refresh()

    def _append_second(
        self, timestamp: str, type_envs: dict[str, set[int | None]]
    ) -> None:
        """Append lines for a completed second.

        Each event type gets its own line with count and contributing envs.
        """
        parts = timestamp.split(":")
        if len(parts) != 3:
            return

        current_minute = parts[1]

        for event_type in sorted(type_envs.keys()):
            env_ids = type_envs[event_type]
            count = len(env_ids)
            text = Text()

            # Timestamp: "MM:SS " or "  :SS " (abbreviated if same minute)
            if current_minute != self._last_minute:
                text.append(f"{parts[1]}:{parts[2]} ", style="dim")
                self._last_minute = current_minute
            else:
                text.append(f"  :{parts[2]} ", style="dim")

            # Event type (shortened for display)
            color = _EVENT_COLORS.get(event_type, "white")
            label = (
                event_type
                .replace("SEED_", "")
                .replace("_COMPLETED", "")
                .replace("_COMPUTED", "")
            )
            text.append(label, style=color)

            # Count if > 1
            if count > 1:
                text.append(f" ×{count}", style="dim")

            # Right-justified env list (only if we have real env_ids)
            env_suffix = self._format_env_list(env_ids)
            if env_suffix:
                # Pad to push envs right (estimate ~35 char width for panel)
                current_len = len(text.plain)
                padding = max(1, 35 - current_len - len(env_suffix))
                text.append(" " * padding)
                text.append(env_suffix, style="dim")

            self._lines.append(text)

    def _format_env_list(self, env_ids: set[int | None]) -> str:
        """Format env IDs for display, e.g., '0 1 2 (+3)'.

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
            extra = len(real_ids) - _MAX_ENVS_SHOWN
            return f"{shown} (+{extra})"

    def render(self):
        """Render the log - just display self._lines.

        NO recalculation. NO grouping. Just display what we have.
        """
        if not self._lines:
            return Text("Waiting for events...", style="dim")

        return Group(*self._lines)

    def on_click(self) -> None:
        """Handle click to open raw event detail modal."""
        if self._snapshot is None or not self._snapshot.event_log:
            return
        self.post_message(self.DetailRequested(list(self._snapshot.event_log)))
