"""EventLog widget - Append-only scrolling log with rich metadata.

Architecture:
- Maintains internal list of line data (append-only for individual events)
- Tracks which events have been processed by ID
- Queues individual events and drip-feeds them at observed events/sec
- Shows rich inline metadata for actionable events (GERMINATED, FOSSILIZED, etc.)
- Aggregates high-frequency events (EPOCH_COMPLETED, REWARD_COMPUTED)
- Updates aggregated lines live as new events arrive
- On render: formats timestamps based on visible line order for proper clock flow

This is a LOG, not a reactive view. Line data stays put once added.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
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
    "SEED_GATE_EVALUATED": "bright_white",
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
    "SEED_GATE_EVALUATED",
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
_DRIP_TICK_SECONDS = 0.1


@dataclass
class _LineData:
    """Raw line data for render-time formatting."""
    seq: int
    timestamp: str  # HH:MM:SS
    content: Text   # Pre-formatted content (without timestamp)
    env_str: str = ""  # Right-justified env info (single ID or "0 1 2 +")
    event_type: str = ""  # Used for updating aggregated lines
    is_aggregate: bool = False


@dataclass(frozen=True)
class _QueuedEvent:
    """EventLogEntry with a scheduled release time for smooth scrolling."""

    queued_ts: float
    ready_ts: float
    entry: "EventLogEntry"


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

    def __init__(
        self,
        max_lines: int = 41,
        buffer_seconds: float = 1.0,
        max_delay_seconds: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Initialize EventLog widget.

        Args:
            max_lines: Maximum lines to display (oldest trimmed).
            buffer_seconds: Delay before displaying individual events for smooth scrolling.
            max_delay_seconds: Maximum wall-clock delay for individual events.
        """
        super().__init__(**kwargs)
        if buffer_seconds > max_delay_seconds:
            raise ValueError("buffer_seconds must be <= max_delay_seconds")
        self._max_lines = max_lines
        self._buffer_seconds = buffer_seconds
        self._max_delay_seconds = max_delay_seconds
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

        # === DRIP-FEED STATE ===
        self._pending_individual: deque[_QueuedEvent] = deque()
        self._drip_budget: float = 0.0
        self._last_drip_ts: float = 0.0

        # Throughput (events/sec) derived from total_events_received over a short window.
        self._eps_samples: deque[tuple[float, int]] = deque()
        self._events_per_second: float = 0.0

        # Stable ordering within same second
        self._next_seq: int = 0

    def on_mount(self) -> None:
        self._last_drip_ts = time.monotonic()
        self.set_interval(_DRIP_TICK_SECONDS, self._drip_tick)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Process snapshot and append new events.

        Individual events are queued and drip-fed with full metadata.
        High-frequency events are aggregated per-second and updated live.
        """
        self._snapshot = snapshot

        self._update_event_rate(snapshot.total_events_received)
        self._update_border_title()

        if not snapshot.event_log:
            self.refresh()
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

        # Queue individual events for drip-feed insertion.
        now = time.monotonic()
        for entry in sorted(individual_events, key=lambda e: e.timestamp):
            self._pending_individual.append(
                _QueuedEvent(
                    queued_ts=now,
                    ready_ts=now + self._buffer_seconds,
                    entry=entry,
                )
            )

        # Process aggregated events (show counts per second, updated live)
        for (timestamp, event_type) in sorted(aggregate_events.keys()):
            env_ids = aggregate_events[(timestamp, event_type)]
            key = (timestamp, event_type)
            existing_idx = self._aggregate_line_index.get(key)
            if existing_idx is None:
                seq = self._allocate_seq()
            else:
                seq = self._line_data[existing_idx].seq

            line_data = self._format_aggregate_event(timestamp, event_type, env_ids, seq=seq)
            if not line_data:
                continue

            if existing_idx is None:
                self._line_data.append(line_data)
            else:
                self._line_data[existing_idx] = line_data

        self._sort_and_trim_lines()
        self._update_border_title()
        self.refresh()

    def _allocate_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def _update_event_rate(self, total_events_received: int) -> None:
        now = time.monotonic()
        self._eps_samples.append((now, total_events_received))

        WINDOW_S = 5.0
        while self._eps_samples and (now - self._eps_samples[0][0]) > WINDOW_S:
            self._eps_samples.popleft()

        if len(self._eps_samples) < 2:
            self._events_per_second = 0.0
            return

        t0, c0 = self._eps_samples[0]
        t1, c1 = self._eps_samples[-1]
        dt = t1 - t0
        self._events_per_second = (c1 - c0) / dt if dt > 0 else 0.0

    def _update_border_title(self) -> None:
        if self._snapshot is None:
            return

        total = self._snapshot.total_events_received
        total_str = f"{total:,}"

        eps = self._events_per_second
        eps_str = f"{eps:.1f}/s" if eps < 100 else f"{eps:.0f}/s"

        if self._trimmed_count > 0:
            self.border_title = f"EVENTS {total_str} ({eps_str}) [click] ↑{self._trimmed_count}"
        else:
            self.border_title = f"EVENTS {total_str} ({eps_str}) [click for detail]"

    def _sort_and_trim_lines(self) -> None:
        # Sort by timestamp for clock flow; seq preserves stable ordering within a second.
        self._line_data.sort(key=lambda ld: (ld.timestamp, ld.seq))

        self._aggregate_line_index.clear()
        for idx, ld in enumerate(self._line_data):
            if ld.is_aggregate:
                self._aggregate_line_index[(ld.timestamp, ld.event_type)] = idx

        if len(self._line_data) <= self._max_lines:
            return

        excess = len(self._line_data) - self._max_lines
        self._trimmed_count += excess
        self._line_data = self._line_data[-self._max_lines:]

        self._aggregate_line_index.clear()
        for idx, ld in enumerate(self._line_data):
            if ld.is_aggregate:
                self._aggregate_line_index[(ld.timestamp, ld.event_type)] = idx

    def _drip_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_drip_ts
        self._last_drip_ts = now
        if dt <= 0:
            return

        if not self._pending_individual:
            self._drip_budget = 0.0
            return

        # Jitter buffer: hold events briefly so bursts can be drip-fed smoothly.
        if self._pending_individual[0].ready_ts > now:
            self._drip_budget = 0.0
            return

        oldest_age = now - self._pending_individual[0].queued_ts
        catch_up_threshold = max(self._max_delay_seconds - _DRIP_TICK_SECONDS, 0.0)
        if oldest_age >= catch_up_threshold:
            emitted = 0
            MAX_EMIT_PER_TICK = 500
            while self._pending_individual and emitted < MAX_EMIT_PER_TICK:
                queued = self._pending_individual[0]
                if queued.ready_ts > now:
                    break
                oldest_age = now - queued.queued_ts
                if oldest_age < self._buffer_seconds:
                    break
                queued = self._pending_individual.popleft()
                line_data = self._format_individual_event(queued.entry, seq=self._allocate_seq())
                if line_data is not None:
                    self._line_data.append(line_data)
                emitted += 1

            if emitted > 0:
                self._drip_budget = 0.0
                self._sort_and_trim_lines()
                self._update_border_title()
                self.refresh()
            return

        rate = self._events_per_second
        if rate <= 0:
            rate = 1.0

        self._drip_budget += rate * dt
        # Prevent runaway bursts if the UI thread stalls or the observed rate spikes.
        MAX_EMIT_PER_TICK = 3
        MAX_BUDGET = float(MAX_EMIT_PER_TICK) * 5.0
        if self._drip_budget > MAX_BUDGET:
            self._drip_budget = MAX_BUDGET

        to_emit = min(int(self._drip_budget), MAX_EMIT_PER_TICK)
        if to_emit <= 0:
            return

        emitted = 0
        while emitted < to_emit and self._pending_individual:
            queued = self._pending_individual[0]
            if queued.ready_ts > now:
                break
            queued = self._pending_individual.popleft()
            line_data = self._format_individual_event(queued.entry, seq=self._allocate_seq())
            if line_data is not None:
                self._line_data.append(line_data)
            emitted += 1

        self._drip_budget -= emitted
        self._sort_and_trim_lines()
        self._update_border_title()
        self.refresh()

    def _format_individual_event(self, entry: "EventLogEntry", *, seq: int) -> _LineData | None:
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

        elif event_type == "SEED_GATE_EVALUATED":
            slot = metadata.get("slot_id", "?")
            gate = metadata.get("gate", "?")
            target = str(metadata.get("target_stage", "?"))
            target_short = _STAGE_SHORT.get(target, target)
            result = str(metadata.get("result", "?"))
            failed_checks = metadata.get("failed_checks")

            content.append(f"{slot} ", style="cyan")
            content.append(f"{gate} ", style="dim")
            if result == "PASS":
                content.append("✓", style="green")
            elif result == "FAIL":
                content.append("✗", style="red")
            else:
                content.append(result, style="yellow")
            content.append(f" →{target_short}", style=color)
            if isinstance(failed_checks, int) and failed_checks > 0:
                content.append(f" ({failed_checks} failed)", style="dim")

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
            seq=seq,
            timestamp=entry.timestamp,
            content=content,
            env_str=str(entry.env_id) if entry.env_id is not None else "",
            event_type=entry.event_type,
            is_aggregate=False,
        )

    def _format_aggregate_event(
        self, timestamp: str, event_type: str, env_ids: set[int | None], *, seq: int
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
            seq=seq,
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
