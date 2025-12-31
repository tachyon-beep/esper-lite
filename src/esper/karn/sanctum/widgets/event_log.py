"""EventLog widget - Append-only scrolling log with rich metadata.

Architecture:
- Maintains internal list of line data (append-only for individual events)
- Tracks which events have been processed by ID
- Shows rich inline metadata for actionable events (GERMINATED, FOSSILIZED, etc.)
- Aggregates high-frequency events (EPOCH_COMPLETED, REWARD_COMPUTED)
- Updates aggregated lines live as new events arrive
- On render: formats timestamps based on visible line order for proper clock flow

This is a LOG, not a reactive view. Line data stays put once added.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EventLogEntry, SanctumSnapshot


# Event type color mapping
_EVENT_COLORS: dict[str, str] = {
    # Seed lifecycle - actionable events
    "SEED_GERMINATED": "bright_yellow",
    "SEED_STAGE_CHANGED": "bright_white",
    "SEED_FOSSILIZED": "bright_green",
    "SEED_PRUNED": "bright_red",
    # PPO events
    "PPO_UPDATE_COMPLETED": "bright_magenta",
    # High-frequency (aggregated)
    "REWARD_COMPUTED": "dim",
    "TRAINING_STARTED": "bright_green",
    "EPOCH_COMPLETED": "bright_blue",
    "BATCH_EPOCH_COMPLETED": "bright_blue",
}

# Events to show individually with full metadata (not aggregated)
_INDIVIDUAL_EVENTS = {
    "SEED_GERMINATED",
    "SEED_STAGE_CHANGED",
    "SEED_FOSSILIZED",
    "SEED_PRUNED",
    "PPO_UPDATE_COMPLETED",
    "BATCH_EPOCH_COMPLETED",
    "TRAINING_STARTED",
}

# Compact stage name mapping
_STAGE_SHORT = {
    "DORMANT": "DORM",
    "GERMINATED": "GERM",
    "TRAINING": "TRAIN",
    "HOLDING": "HOLD",
    "BLENDING": "BLEND",
    "FOSSILIZED": "FOSS",
    "PRUNED": "PRUNE",
    "EMBARGOED": "EMBG",
    "RESETTING": "RESET",
}

# Max envs to show before truncating with "+"
_MAX_ENVS_SHOWN = 3


@dataclass
class _LineData:
    """Raw line data for render-time formatting."""
    timestamp: str  # HH:MM:SS
    content: Text   # Pre-formatted content (without timestamp)
    env_str: str = ""  # Right-justified env info (single ID or "0 1 2 +")
    event_type: str = ""  # Used for updating aggregated lines
    is_aggregate: bool = False


class EventLog(Static):
    """Append-only scrolling event log with rich metadata display.

    - Shows individual entries for actionable events (seed lifecycle, PPO)
    - Aggregates high-frequency events (EPOCH_COMPLETED, REWARD_COMPUTED)
    - Displays key metadata inline (slot, blueprint, stage transition, etc.)
    - Timestamp abbreviation computed at render time for proper clock flow
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
        self.border_title = "EVENTS [click for detail]"

        # === APPEND-ONLY STATE ===
        # Line data: list of _LineData for render-time formatting
        self._line_data: list[_LineData] = []
        # Track processed event IDs to avoid duplicates
        self._processed_ids: set[str] = set()
        # Mapping for in-place updates of aggregated lines
        self._aggregate_line_index: dict[tuple[str, str], int] = {}
        # Keep reference to snapshot for click handler
        self._snapshot: SanctumSnapshot | None = None
        # Track trimmed lines for scroll indicator
        self._trimmed_count: int = 0

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Process snapshot and append new events.

        Individual events are shown immediately with full metadata.
        High-frequency events are aggregated per-second and updated live.
        """
        self._snapshot = snapshot

        if not snapshot.event_log:
            return

        # Separate individual vs aggregated events
        individual_events: list["EventLogEntry"] = []
        aggregate_events: dict[tuple[str, str], set[int | None]] = defaultdict(set)

        for entry in snapshot.event_log:
            # Create a unique ID for deduplication
            event_id = f"{entry.timestamp}:{entry.event_type}:{entry.env_id}:{hash(str(entry.metadata))}"

            if entry.event_type in _INDIVIDUAL_EVENTS:
                if event_id not in self._processed_ids:
                    individual_events.append(entry)
                    self._processed_ids.add(event_id)
            else:
                # Aggregate by timestamp and event type (updated live)
                aggregate_events[(entry.timestamp, entry.event_type)].add(entry.env_id)

        # Process individual events (show immediately with metadata)
        for entry in sorted(individual_events, key=lambda e: e.timestamp):
            line_data = self._format_individual_event(entry)
            if line_data:
                self._line_data.append(line_data)

        # Process aggregated events (show counts per second, updated live)
        for (timestamp, event_type) in sorted(aggregate_events.keys()):
            env_ids = aggregate_events[(timestamp, event_type)]
            line_data = self._format_aggregate_event(timestamp, event_type, env_ids)
            if not line_data:
                continue

            key = (timestamp, event_type)
            existing_idx = self._aggregate_line_index.get(key)
            if existing_idx is None:
                self._line_data.append(line_data)
            else:
                self._line_data[existing_idx] = line_data

        # Sort by timestamp to maintain clock flow
        self._line_data.sort(key=lambda ld: ld.timestamp)

        # Rebuild aggregate line index after sort
        self._aggregate_line_index.clear()
        for idx, ld in enumerate(self._line_data):
            if ld.is_aggregate:
                self._aggregate_line_index[(ld.timestamp, ld.event_type)] = idx

        # Trim if we exceed max lines
        if len(self._line_data) > self._max_lines:
            excess = len(self._line_data) - self._max_lines
            self._trimmed_count += excess
            self._line_data = self._line_data[-self._max_lines:]
            # Rebuild again after trimming (indices changed)
            self._aggregate_line_index.clear()
            for idx, ld in enumerate(self._line_data):
                if ld.is_aggregate:
                    self._aggregate_line_index[(ld.timestamp, ld.event_type)] = idx

        # Update border title with scroll indicator
        if self._trimmed_count > 0:
            self.border_title = f"EVENTS [click] ↑{self._trimmed_count}"
        else:
            self.border_title = "EVENTS [click for detail]"

        # Trigger re-render
        self.refresh()

    def _format_individual_event(self, entry: "EventLogEntry") -> _LineData | None:
        """Format an individual event with rich inline metadata.

        Returns _LineData with timestamp and content separated for render-time formatting.
        """
        content = Text()
        event_type = entry.event_type
        metadata = entry.metadata
        color = _EVENT_COLORS.get(event_type, "white")

        if event_type == "SEED_GERMINATED":
            slot = metadata.get("slot_id", "?")
            blueprint = metadata.get("blueprint", "?")
            # Truncate blueprint to 10 chars
            if len(str(blueprint)) > 10:
                blueprint = str(blueprint)[:9] + "…"
            content.append(f"{slot} ", style="cyan")
            content.append("GERM ", style=color)
            content.append(str(blueprint), style="white")

        elif event_type == "SEED_STAGE_CHANGED":
            slot = metadata.get("slot_id", "?")
            from_stage = _STAGE_SHORT.get(str(metadata.get("from", "?")), "?")
            to_stage = _STAGE_SHORT.get(str(metadata.get("to", "?")), "?")
            content.append(f"{slot} ", style="cyan")
            content.append(f"{from_stage}→{to_stage}", style=color)

        elif event_type == "SEED_FOSSILIZED":
            slot = metadata.get("slot_id", "?")
            improvement = metadata.get("improvement", 0)
            content.append(f"{slot} ", style="cyan")
            content.append("FOSS ", style=color)
            if isinstance(improvement, (int, float)):
                content.append(f"+{improvement:.1f}%", style="green")
            else:
                content.append(str(improvement), style="green")

        elif event_type == "SEED_PRUNED":
            slot = metadata.get("slot_id", "?")
            reason = metadata.get("reason", "")
            # Truncate reason to 8 chars
            if len(str(reason)) > 8:
                reason = str(reason)[:7] + "…"
            content.append(f"{slot} ", style="cyan")
            content.append("PRUNE ", style=color)
            if reason:
                content.append(str(reason), style="dim")

        elif event_type == "PPO_UPDATE_COMPLETED":
            entropy = metadata.get("entropy")
            clip = metadata.get("clip_fraction")
            content.append("PPO ", style=color)
            if entropy is not None:
                content.append("ent:", style="dim")
                content.append(f"{float(entropy):.2f}", style="cyan")
            if clip is not None:
                content.append(" clip:", style="dim")
                clip_val = float(clip)
                content.append(f"{clip_val*100:.0f}%", style="yellow" if clip_val > 0.2 else "green")

        elif event_type == "BATCH_EPOCH_COMPLETED":
            batch = metadata.get("batch", "?")
            episodes = metadata.get("episodes", 0)
            content.append("BATCH ", style=color)
            content.append(f"#{batch}", style="cyan")
            if episodes:
                content.append(f" +{episodes}eps", style="dim")

        elif event_type == "TRAINING_STARTED":
            content.append("▶ TRAINING STARTED", style=color)

        else:
            # Fallback for unknown types
            label = event_type.replace("SEED_", "").replace("_COMPLETED", "")
            content.append(label, style=color)

        return _LineData(
            timestamp=entry.timestamp,
            content=content,
            env_str=str(entry.env_id) if entry.env_id is not None else "",
            event_type=entry.event_type,
            is_aggregate=False,
        )

    def _format_aggregate_event(
        self, timestamp: str, event_type: str, env_ids: set[int | None]
    ) -> _LineData | None:
        """Format an aggregated event with count."""
        content = Text()
        color = _EVENT_COLORS.get(event_type, "dim")
        label = (
            event_type
            .replace("SEED_", "")
            .replace("_COMPLETED", "")
            .replace("_COMPUTED", "")
        )

        count = len(env_ids)
        content.append(label, style=color)
        if count > 1:
            content.append(f" ×{count}", style="dim")

        # Format env IDs for the line data
        real_ids = sorted(eid for eid in env_ids if eid is not None)
        env_str: str | None = None
        if real_ids:
            if len(real_ids) <= _MAX_ENVS_SHOWN:
                env_str = " ".join(str(eid) for eid in real_ids)
            else:
                env_str = " ".join(str(eid) for eid in real_ids[:_MAX_ENVS_SHOWN]) + " +"

        return _LineData(
            timestamp=timestamp,
            content=content,
            env_str=env_str if env_str else "",
            event_type=event_type,
            is_aggregate=True,
        )

    def render(self) -> RenderableType:
        """Render the log with proper timestamp abbreviation.

        Uses actual widget width to align env IDs to right edge.
        Timestamp rules (computed at render time for proper clock flow):
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

        for i, ld in enumerate(self._line_data):
            parts = ld.timestamp.split(":")
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
                line.append(f" {ld.timestamp} ", style="bold cyan")
            else:
                # Short format: "      :SS " = 10 chars (6 spaces + :SS + space)
                line.append(f"      :{second} ", style="dim")

            # Always update last_minute for tracking
            last_minute = hour_minute

            # Append the pre-formatted content
            line.append_text(ld.content)

            # Right-justify env info if present
            if ld.env_str:
                left_len = len(line.plain)
                right_len = len(ld.env_str)
                padding = max(1, width - left_len - right_len)
                line.append(" " * padding)
                line.append(ld.env_str, style="dim")

            lines.append(line)

        return Group(*lines)

    def on_click(self) -> None:
        """Handle click to open raw event detail modal."""
        if self._snapshot is None or not self._snapshot.event_log:
            return
        self.post_message(self.DetailRequested(list(self._snapshot.event_log)))
