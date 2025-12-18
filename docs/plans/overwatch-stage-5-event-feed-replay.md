# Overwatch Stage 5: Event Feed + Replay Controls

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the Event Feed widget for timestamped lifecycle events and implement full replay controls for navigating recorded training sessions.

**Architecture:** EventFeed is a scrollable widget showing FeedEvent items with type badges. ReplayController is a state machine managing playback (play/pause/step/speed). The app wires replay controls to keybindings and displays a replay status bar showing position and progress.

**Tech Stack:** Python 3.11, Textual widgets (ScrollableContainer, Static), Rich markup for badges, asyncio timers for playback.

**Prerequisites:**
- Stage 4 complete (Detail Panels)
- Branch: `feat/overwatch-textual-ui`

---

## Task 1: Create EventBadge Helper and EventFeed Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/event_feed.py`
- Create: `tests/karn/overwatch/test_event_feed.py`

**Step 1: Write failing tests for EventFeed**

```python
# tests/karn/overwatch/test_event_feed.py
"""Tests for EventFeed widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import FeedEvent


class TestEventBadge:
    """Tests for event badge rendering."""

    def test_event_badge_gate(self) -> None:
        """GATE events get cyan badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("GATE")
        assert "GATE" in badge
        assert "cyan" in badge

    def test_event_badge_stage(self) -> None:
        """STAGE events get blue badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("STAGE")
        assert "STAGE" in badge
        assert "blue" in badge

    def test_event_badge_ppo(self) -> None:
        """PPO events get magenta badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("PPO")
        assert "PPO" in badge
        assert "magenta" in badge

    def test_event_badge_germ(self) -> None:
        """GERM events get green badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("GERM")
        assert "GERM" in badge
        assert "green" in badge

    def test_event_badge_cull(self) -> None:
        """CULL events get red badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("CULL")
        assert "CULL" in badge
        assert "red" in badge

    def test_event_badge_unknown(self) -> None:
        """Unknown events get white badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("UNKNOWN")
        assert "UNKNOWN" in badge
        assert "white" in badge


class TestEventFeed:
    """Tests for EventFeed widget."""

    def test_event_feed_imports(self) -> None:
        """EventFeed can be imported."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        assert EventFeed is not None

    def test_event_feed_renders_events(self) -> None:
        """EventFeed renders list of events."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Env 0: Gate G1 passed"),
            FeedEvent("12:00:02", "GERM", 1, "Env 1: Seed germinated in r0c1"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        assert "12:00:01" in content
        assert "Gate G1 passed" in content
        assert "12:00:02" in content
        assert "Seed germinated" in content

    def test_event_feed_shows_badges(self) -> None:
        """EventFeed shows event type badges."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Gate passed"),
            FeedEvent("12:00:02", "PPO", None, "Policy updated"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        assert "GATE" in content
        assert "PPO" in content

    def test_event_feed_shows_env_id(self) -> None:
        """EventFeed shows env ID when present."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 3, "Gate passed"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        assert "E3" in content or "Env 3" in content or "[3]" in content

    def test_event_feed_empty_state(self) -> None:
        """EventFeed shows empty state when no events."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        feed = EventFeed()

        content = feed.render_events()
        assert "No events" in content or "Waiting" in content

    def test_event_feed_compact_mode(self) -> None:
        """EventFeed has compact mode (fewer visible lines)."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        feed = EventFeed()
        assert feed.expanded is False  # Starts compact

        feed.toggle_expanded()
        assert feed.expanded is True

        feed.toggle_expanded()
        assert feed.expanded is False

    def test_event_feed_filters_by_type(self) -> None:
        """EventFeed can filter events by type."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Gate passed"),
            FeedEvent("12:00:02", "PPO", None, "Policy updated"),
            FeedEvent("12:00:03", "GATE", 1, "Gate failed"),
        ]
        feed = EventFeed()
        feed.update_events(events)
        feed.set_filter("GATE")

        content = feed.render_events()
        assert "Gate passed" in content
        assert "Gate failed" in content
        assert "Policy updated" not in content

    def test_event_feed_clear_filter(self) -> None:
        """EventFeed can clear filter to show all events."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Gate passed"),
            FeedEvent("12:00:02", "PPO", None, "Policy updated"),
        ]
        feed = EventFeed()
        feed.update_events(events)
        feed.set_filter("GATE")
        feed.clear_filter()

        content = feed.render_events()
        assert "Gate passed" in content
        assert "Policy updated" in content

    def test_event_feed_newest_first(self) -> None:
        """EventFeed shows newest events at bottom (log style)."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "First event"),
            FeedEvent("12:00:02", "GATE", 0, "Second event"),
            FeedEvent("12:00:03", "GATE", 0, "Third event"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        # Newest at bottom means Third appears after First
        first_pos = content.find("First event")
        third_pos = content.find("Third event")
        assert first_pos < third_pos
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_event_feed.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement EventFeed widget**

```python
# src/esper/karn/overwatch/widgets/event_feed.py
"""Event Feed Widget.

Displays timestamped lifecycle events with type badges:
- GATE: Gate evaluations (cyan)
- STAGE: Stage transitions (blue)
- PPO: Policy updates (magenta)
- GERM: Seed germinations (green)
- CULL: Seed cullings (red)

Supports filtering by event type and compact/expanded modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import FeedEvent


# Event type to color mapping
EVENT_COLORS: dict[str, str] = {
    "GATE": "cyan",
    "STAGE": "blue",
    "PPO": "magenta",
    "GERM": "green",
    "CULL": "red",
    "BLEND": "yellow",
    "WARN": "yellow",
    "ERROR": "red",
}


def event_badge(event_type: str) -> str:
    """Render event type as colored badge.

    Args:
        event_type: Event type string (GATE, PPO, etc.)

    Returns:
        Rich markup string for badge like "[cyan][GATE][/cyan]"
    """
    color = EVENT_COLORS.get(event_type, "white")
    return f"[{color}][{event_type}][/{color}]"


class EventFeed(ScrollableContainer):
    """Scrollable feed of timestamped lifecycle events.

    Features:
    - Type badges with color coding
    - Compact (4 lines) / expanded (8 lines) modes
    - Filter by event type
    - Newest events at bottom (log style)

    Usage:
        feed = EventFeed()
        feed.update_events(snapshot.event_feed)
        feed.set_filter("GATE")  # Only show gate events
        feed.toggle_expanded()   # Toggle size
    """

    DEFAULT_CSS = """
    EventFeed {
        height: 6;
        border: solid $primary;
        padding: 0 1;
    }

    EventFeed.expanded {
        height: 12;
    }

    EventFeed .event-line {
        height: 1;
    }

    EventFeed .timestamp {
        color: $text-muted;
    }

    EventFeed .env-id {
        color: $secondary;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the event feed."""
        super().__init__(**kwargs)
        self._events: list[FeedEvent] = []
        self._filter_type: str | None = None
        self._expanded = False

    @property
    def expanded(self) -> bool:
        """Whether feed is in expanded mode."""
        return self._expanded

    def toggle_expanded(self) -> None:
        """Toggle between compact and expanded mode."""
        self._expanded = not self._expanded
        if self._expanded:
            self.add_class("expanded")
        else:
            self.remove_class("expanded")

    def set_filter(self, event_type: str) -> None:
        """Filter events to show only specified type.

        Args:
            event_type: Event type to filter (e.g., "GATE")
        """
        self._filter_type = event_type
        self._refresh_content()

    def clear_filter(self) -> None:
        """Clear filter to show all events."""
        self._filter_type = None
        self._refresh_content()

    def update_events(self, events: list[FeedEvent]) -> None:
        """Update the event list.

        Args:
            events: List of FeedEvent objects
        """
        self._events = events
        self._refresh_content()

    def _get_filtered_events(self) -> list[FeedEvent]:
        """Get events after applying filter."""
        if self._filter_type is None:
            return self._events
        return [e for e in self._events if e.event_type == self._filter_type]

    def render_events(self) -> str:
        """Render all events as text.

        Returns:
            Multi-line string with all event lines
        """
        events = self._get_filtered_events()

        if not events:
            filter_text = f" (filter: {self._filter_type})" if self._filter_type else ""
            return f"[dim]No events{filter_text}[/dim]"

        lines = []
        for event in events:
            line = self._render_event_line(event)
            lines.append(line)

        return "\n".join(lines)

    def _render_event_line(self, event: FeedEvent) -> str:
        """Render a single event line.

        Args:
            event: FeedEvent to render

        Returns:
            Formatted line like "12:00:01 [GATE] [E3] Gate G1 passed"
        """
        parts = []

        # Timestamp (dimmed)
        parts.append(f"[dim]{event.timestamp}[/dim]")

        # Event badge
        parts.append(event_badge(event.event_type))

        # Env ID if present
        if event.env_id is not None:
            parts.append(f"[cyan][E{event.env_id}][/cyan]")

        # Message
        parts.append(event.message)

        return " ".join(parts)

    def compose(self) -> ComposeResult:
        """Compose the feed layout."""
        yield Static(self.render_events(), id="event-content")

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#event-content", Static).update(self.render_events())
        except Exception:
            pass  # Widget not mounted yet
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_event_feed.py -v`

Expected: All 14 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/event_feed.py tests/karn/overwatch/test_event_feed.py
git commit -m "feat(overwatch): add EventFeed widget with badges and filtering"
```

---

## Task 2: Create ReplayController State Machine

**Files:**
- Create: `src/esper/karn/overwatch/replay_controller.py`
- Create: `tests/karn/overwatch/test_replay_controller.py`

**Step 1: Write failing tests for ReplayController**

```python
# tests/karn/overwatch/test_replay_controller.py
"""Tests for ReplayController state machine."""

from __future__ import annotations

from pathlib import Path


class TestReplayController:
    """Tests for ReplayController."""

    def test_replay_controller_imports(self) -> None:
        """ReplayController can be imported."""
        from esper.karn.overwatch.replay_controller import ReplayController

        assert ReplayController is not None

    def test_replay_controller_load_file(self, tmp_path: Path) -> None:
        """ReplayController loads snapshots from file."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        # Create test file
        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        assert controller.total_frames == 5
        assert controller.current_index == 0

    def test_replay_controller_current_snapshot(self, tmp_path: Path) -> None:
        """ReplayController returns current snapshot."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
                episode=42,
            )
            writer.write(snap)

        controller = ReplayController(path)
        current = controller.current_snapshot
        assert current is not None
        assert current.episode == 42

    def test_replay_controller_step_forward(self, tmp_path: Path) -> None:
        """ReplayController steps forward through snapshots."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        assert controller.current_snapshot.episode == 0

        controller.step_forward()
        assert controller.current_snapshot.episode == 1

        controller.step_forward()
        assert controller.current_snapshot.episode == 2

    def test_replay_controller_step_backward(self, tmp_path: Path) -> None:
        """ReplayController steps backward through snapshots."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        controller.step_forward()
        controller.step_forward()
        assert controller.current_index == 2

        controller.step_backward()
        assert controller.current_index == 1

        controller.step_backward()
        assert controller.current_index == 0

    def test_replay_controller_bounds_checking(self, tmp_path: Path) -> None:
        """ReplayController respects bounds."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)

        # Can't go below 0
        controller.step_backward()
        assert controller.current_index == 0

        # Can't go above max
        controller.step_forward()
        controller.step_forward()
        controller.step_forward()  # Try to exceed
        assert controller.current_index == 2

    def test_replay_controller_play_pause(self, tmp_path: Path) -> None:
        """ReplayController toggles play/pause state."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        controller = ReplayController(path)
        assert controller.playing is False

        controller.toggle_play()
        assert controller.playing is True

        controller.toggle_play()
        assert controller.playing is False

    def test_replay_controller_speed(self, tmp_path: Path) -> None:
        """ReplayController adjusts playback speed."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        controller = ReplayController(path)
        assert controller.speed == 1.0

        controller.increase_speed()
        assert controller.speed == 2.0

        controller.increase_speed()
        assert controller.speed == 4.0

        controller.decrease_speed()
        assert controller.speed == 2.0

        controller.decrease_speed()
        assert controller.speed == 1.0

        controller.decrease_speed()
        assert controller.speed == 0.5

    def test_replay_controller_speed_bounds(self, tmp_path: Path) -> None:
        """ReplayController speed has min/max bounds."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        controller = ReplayController(path)

        # Min speed is 0.25
        for _ in range(10):
            controller.decrease_speed()
        assert controller.speed == 0.25

        # Max speed is 8.0
        for _ in range(10):
            controller.increase_speed()
        assert controller.speed == 8.0

    def test_replay_controller_progress(self, tmp_path: Path) -> None:
        """ReplayController reports progress percentage."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                )
                writer.write(snap)

        controller = ReplayController(path)
        assert controller.progress == 0.0

        controller.step_forward()
        controller.step_forward()
        assert controller.progress == 0.5  # 2 of 4 steps (index 2 of 0-4)

        controller.step_forward()
        controller.step_forward()
        assert controller.progress == 1.0

    def test_replay_controller_seek(self, tmp_path: Path) -> None:
        """ReplayController can seek to specific index."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(10):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:{i:02d}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        controller.seek(5)
        assert controller.current_index == 5
        assert controller.current_snapshot.episode == 5

    def test_replay_controller_status_text(self, tmp_path: Path) -> None:
        """ReplayController generates status text."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                )
                writer.write(snap)

        controller = ReplayController(path)
        status = controller.status_text

        assert "REPLAY" in status or "Paused" in status
        assert "1x" in status or "1.0" in status
        assert "1/5" in status or "0%" in status
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay_controller.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement ReplayController**

```python
# src/esper/karn/overwatch/replay_controller.py
"""Replay Controller for Overwatch TUI.

State machine for controlling replay playback:
- Play/pause toggle
- Step forward/backward
- Speed adjustment (0.25x to 8x)
- Progress tracking and seeking

Usage:
    controller = ReplayController(Path("training.jsonl"))
    controller.toggle_play()
    controller.step_forward()
    snapshot = controller.current_snapshot
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


# Available speed multipliers
SPEED_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


class ReplayController:
    """State machine for replay playback control.

    Manages:
    - Loading snapshots from JSONL file
    - Current position (frame index)
    - Play/pause state
    - Playback speed
    - Navigation (step, seek)

    The controller does NOT handle timing - it just provides
    state and navigation. The app's timer calls step_forward()
    based on speed when playing.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize replay controller.

        Args:
            path: Path to JSONL replay file
        """
        self._path = Path(path)
        self._snapshots: list[TuiSnapshot] = []
        self._current_index: int = 0
        self._playing: bool = False
        self._speed_index: int = 2  # 1.0x

        self._load_snapshots()

    def _load_snapshots(self) -> None:
        """Load all snapshots from file into memory."""
        from esper.karn.overwatch.replay import SnapshotReader

        reader = SnapshotReader(self._path)
        self._snapshots = list(reader)

    @property
    def total_frames(self) -> int:
        """Total number of frames in replay."""
        return len(self._snapshots)

    @property
    def current_index(self) -> int:
        """Current frame index (0-based)."""
        return self._current_index

    @property
    def current_snapshot(self) -> TuiSnapshot | None:
        """Current snapshot at playback position."""
        if not self._snapshots:
            return None
        return self._snapshots[self._current_index]

    @property
    def playing(self) -> bool:
        """Whether replay is currently playing."""
        return self._playing

    @property
    def speed(self) -> float:
        """Current playback speed multiplier."""
        return SPEED_LEVELS[self._speed_index]

    @property
    def progress(self) -> float:
        """Progress through replay (0.0 to 1.0)."""
        if self.total_frames <= 1:
            return 0.0
        return self._current_index / (self.total_frames - 1)

    @property
    def status_text(self) -> str:
        """Human-readable status string.

        Returns:
            Status like "[▶ REPLAY 2x] 3/10 30%"
        """
        icon = "▶" if self._playing else "⏸"
        speed_str = f"{self.speed}x" if self.speed != 1.0 else "1x"
        frame_str = f"{self._current_index + 1}/{self.total_frames}"
        pct_str = f"{int(self.progress * 100)}%"

        return f"[{icon} REPLAY {speed_str}] {frame_str} {pct_str}"

    def toggle_play(self) -> None:
        """Toggle play/pause state."""
        self._playing = not self._playing

    def pause(self) -> None:
        """Pause playback."""
        self._playing = False

    def play(self) -> None:
        """Start playback."""
        self._playing = True

    def step_forward(self) -> bool:
        """Advance to next frame.

        Returns:
            True if advanced, False if at end
        """
        if self._current_index < self.total_frames - 1:
            self._current_index += 1
            return True
        return False

    def step_backward(self) -> bool:
        """Go back to previous frame.

        Returns:
            True if moved back, False if at start
        """
        if self._current_index > 0:
            self._current_index -= 1
            return True
        return False

    def seek(self, index: int) -> None:
        """Seek to specific frame index.

        Args:
            index: Target frame index (clamped to valid range)
        """
        self._current_index = max(0, min(index, self.total_frames - 1))

    def increase_speed(self) -> None:
        """Increase playback speed to next level."""
        if self._speed_index < len(SPEED_LEVELS) - 1:
            self._speed_index += 1

    def decrease_speed(self) -> None:
        """Decrease playback speed to previous level."""
        if self._speed_index > 0:
            self._speed_index -= 1

    def tick_interval_ms(self) -> float:
        """Get timer interval for current speed.

        Returns:
            Milliseconds between frames at current speed.
            Base rate is 1000ms (1 frame per second).
        """
        return 1000.0 / self.speed
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay_controller.py -v`

Expected: All 13 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/replay_controller.py tests/karn/overwatch/test_replay_controller.py
git commit -m "feat(overwatch): add ReplayController state machine"
```

---

## Task 3: Create ReplayStatusBar Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/replay_status.py`
- Create: `tests/karn/overwatch/test_replay_status.py`

**Step 1: Write failing tests for ReplayStatusBar**

```python
# tests/karn/overwatch/test_replay_status.py
"""Tests for ReplayStatusBar widget."""

from __future__ import annotations


class TestReplayStatusBar:
    """Tests for ReplayStatusBar widget."""

    def test_replay_status_imports(self) -> None:
        """ReplayStatusBar can be imported."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        assert ReplayStatusBar is not None

    def test_replay_status_shows_mode(self) -> None:
        """ReplayStatusBar shows replay mode indicator."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "REPLAY" in content
        assert "▶" in content

    def test_replay_status_shows_paused(self) -> None:
        """ReplayStatusBar shows paused indicator."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=False, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "⏸" in content or "PAUSED" in content

    def test_replay_status_shows_speed(self) -> None:
        """ReplayStatusBar shows playback speed."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=2.0, current=5, total=10)

        content = bar.render_bar()
        assert "2" in content or "2x" in content

    def test_replay_status_shows_progress_bar(self) -> None:
        """ReplayStatusBar shows visual progress bar."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "█" in content or "▓" in content or "░" in content

    def test_replay_status_shows_frame_count(self) -> None:
        """ReplayStatusBar shows current/total frame count."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "5" in content or "6" in content  # 0-indexed or 1-indexed
        assert "10" in content

    def test_replay_status_shows_timestamp(self) -> None:
        """ReplayStatusBar shows snapshot timestamp."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(
            playing=True,
            speed=1.0,
            current=5,
            total=10,
            timestamp="12:00:05",
        )

        content = bar.render_bar()
        assert "12:00:05" in content

    def test_replay_status_hidden_when_not_replay(self) -> None:
        """ReplayStatusBar can be hidden for live mode."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.set_visible(False)

        assert bar.is_visible is False
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay_status.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement ReplayStatusBar widget**

```python
# src/esper/karn/overwatch/widgets/replay_status.py
"""Replay Status Bar Widget.

Displays replay playback status:
- Play/pause indicator
- Speed multiplier
- Progress bar
- Frame counter
- Timestamp from current snapshot
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static


def progress_bar(progress: float, width: int = 20) -> str:
    """Generate a progress bar.

    Args:
        progress: Progress 0.0 to 1.0
        width: Bar width in characters

    Returns:
        Progress bar like "████████░░░░░░░░░░░░"
    """
    filled = int(progress * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


class ReplayStatusBar(Static):
    """Status bar showing replay playback state.

    Displays:
    - Mode icon (▶ playing, ⏸ paused)
    - Speed multiplier
    - Visual progress bar
    - Frame counter (current/total)
    - Timestamp from snapshot

    Usage:
        bar = ReplayStatusBar()
        bar.update_status(
            playing=True,
            speed=2.0,
            current=5,
            total=100,
            timestamp="12:00:05"
        )
    """

    DEFAULT_CSS = """
    ReplayStatusBar {
        height: 1;
        background: $primary-darken-2;
        padding: 0 1;
    }

    ReplayStatusBar.hidden {
        display: none;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the replay status bar."""
        super().__init__("", **kwargs)
        self._playing = False
        self._speed = 1.0
        self._current = 0
        self._total = 0
        self._timestamp = ""
        self._visible = True

    @property
    def is_visible(self) -> bool:
        """Whether the bar is visible."""
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Set visibility.

        Args:
            visible: True to show, False to hide
        """
        self._visible = visible
        if visible:
            self.remove_class("hidden")
        else:
            self.add_class("hidden")

    def update_status(
        self,
        playing: bool,
        speed: float,
        current: int,
        total: int,
        timestamp: str = "",
    ) -> None:
        """Update replay status.

        Args:
            playing: Whether playback is active
            speed: Playback speed multiplier
            current: Current frame index (0-based)
            total: Total frame count
            timestamp: Timestamp from current snapshot
        """
        self._playing = playing
        self._speed = speed
        self._current = current
        self._total = total
        self._timestamp = timestamp
        self.update(self.render_bar())

    def render_bar(self) -> str:
        """Render the status bar content.

        Returns:
            Formatted status string
        """
        # Mode icon
        icon = "[green]▶[/green]" if self._playing else "[yellow]⏸[/yellow]"

        # Speed
        speed_str = f"{self._speed}x" if self._speed != 1.0 else "1x"

        # Progress
        progress = self._current / max(1, self._total - 1) if self._total > 1 else 0.0
        bar = progress_bar(progress, width=15)
        pct = int(progress * 100)

        # Frame counter (1-indexed for display)
        frame_str = f"{self._current + 1}/{self._total}"

        # Timestamp
        time_str = f" {self._timestamp}" if self._timestamp else ""

        return f"{icon} [bold]REPLAY[/bold] {speed_str} [{bar}] {frame_str} {pct}%{time_str}"
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay_status.py -v`

Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/replay_status.py tests/karn/overwatch/test_replay_status.py
git commit -m "feat(overwatch): add ReplayStatusBar widget"
```

---

## Task 4: Update Widget Package Exports

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/__init__.py`
- Modify: `tests/karn/overwatch/test_widgets.py`

**Step 1: Add test for new widget exports**

Append to `tests/karn/overwatch/test_widgets.py`:

```python
class TestStage5WidgetExports:
    """Tests for Stage 5 widget exports."""

    def test_event_feed_importable(self) -> None:
        """EventFeed is importable from package."""
        from esper.karn.overwatch.widgets import EventFeed

        assert EventFeed is not None

    def test_replay_status_bar_importable(self) -> None:
        """ReplayStatusBar is importable from package."""
        from esper.karn.overwatch.widgets import ReplayStatusBar

        assert ReplayStatusBar is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestStage5WidgetExports -v`

Expected: FAIL with `ImportError`

**Step 3: Update widgets __init__.py**

Add to `src/esper/karn/overwatch/widgets/__init__.py`:

```python
"""Overwatch TUI Widgets.

Custom Textual widgets for the Overwatch monitoring interface.
"""

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.slot_chip import SlotChip
from esper.karn.overwatch.widgets.env_row import EnvRow
from esper.karn.overwatch.widgets.flight_board import FlightBoard
from esper.karn.overwatch.widgets.run_header import RunHeader
from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip
from esper.karn.overwatch.widgets.context_panel import ContextPanel
from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel
from esper.karn.overwatch.widgets.detail_panel import DetailPanel
from esper.karn.overwatch.widgets.event_feed import EventFeed
from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

__all__ = [
    "HelpOverlay",
    "SlotChip",
    "EnvRow",
    "FlightBoard",
    "RunHeader",
    "TamiyoStrip",
    "ContextPanel",
    "TamiyoDetailPanel",
    "DetailPanel",
    "EventFeed",
    "ReplayStatusBar",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestStage5WidgetExports -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/__init__.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): export Stage 5 widgets (EventFeed, ReplayStatusBar)"
```

---

## Task 5: Wire EventFeed and Replay Controls into App

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`

**Step 0: Understand the delta from existing app.py**

The current `app.py` has methods that will be **removed** and **replaced**:

**Removed methods:**
- `_load_first_snapshot()` - replaced by `_init_replay()`
- `_render_event_feed_content()` - no longer needed (EventFeed widget handles rendering)

**Modified methods:**
- `compose()` - Replace `Static` event feed with `EventFeed` widget, add `ReplayStatusBar` after Header
- `on_mount()` - Replace `_load_first_snapshot()` call with `_init_replay()`, add else branch to hide ReplayStatusBar in live mode
- `_update_all_widgets()` - Change event feed line from `self.query_one("#event-feed", Static).update(...)` to `self.query_one(EventFeed).update_events(self._snapshot.event_feed)`

**New instance variables in `__init__`:**
- `self._replay_controller = None`
- `self._playback_timer = None`

**New methods:**
- `_init_replay()` - Initialize ReplayController and load first snapshot
- `_update_replay_status()` - Update ReplayStatusBar with current state
- `_start_playback()` / `_stop_playback()` - Timer management
- `_playback_tick()` - Called by timer to advance frames
- `action_toggle_play()`, `action_step_forward()`, `action_step_backward()`, `action_speed_up()`, `action_speed_down()`, `action_toggle_feed()` - Replay control actions

**Step 1: Update app.py with replay infrastructure**

This is a significant update. The key changes:
1. Add imports for EventFeed, ReplayStatusBar, ReplayController
2. Add replay keybindings: Space, period, comma, less, greater, f
3. Replace static event feed with EventFeed widget
4. Add ReplayStatusBar below header
5. Initialize ReplayController when replay_path provided
6. Add timer for playback
7. Add action methods for all replay controls
8. Update `_update_all_widgets()` to include event feed

```python
# src/esper/karn/overwatch/app.py
"""Overwatch Textual Application.

Main application class for the Overwatch TUI monitoring interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header, Static

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.flight_board import FlightBoard
from esper.karn.overwatch.widgets.run_header import RunHeader
from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip
from esper.karn.overwatch.widgets.detail_panel import DetailPanel
from esper.karn.overwatch.widgets.event_feed import EventFeed
from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class OverwatchApp(App):
    """Overwatch TUI for monitoring Esper training runs.

    Provides real-time visibility into:
    - Training environments (Flight Board)
    - Seed lifecycle and health
    - Tamiyo agent decisions
    - System resources

    Usage:
        app = OverwatchApp()
        app.run()

        # Or with replay file:
        app = OverwatchApp(replay_path="training.jsonl")
        app.run()
    """

    TITLE = "Esper Overwatch"
    SUB_TITLE = "Training Monitor"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "toggle_help", "Help", show=True),
        Binding("escape", "dismiss", "Dismiss", show=False),
        Binding("c", "show_context", "Context", show=True),
        Binding("t", "show_tamiyo", "Tamiyo", show=True),
        # Replay controls
        Binding("space", "toggle_play", "Play/Pause", show=True),
        Binding("period", "step_forward", "Step →", show=False),
        Binding("comma", "step_backward", "← Step", show=False),
        Binding("shift+period", "speed_up", "Faster", show=False),
        Binding("shift+comma", "speed_down", "Slower", show=False),
        Binding("f", "toggle_feed", "Feed", show=True),
    ]

    def __init__(
        self,
        replay_path: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Overwatch app.

        Args:
            replay_path: Optional path to JSONL replay file
            **kwargs: Additional args passed to App
        """
        super().__init__(**kwargs)
        self._replay_path = Path(replay_path) if replay_path else None
        self._snapshot: TuiSnapshot | None = None
        self._help_visible = False
        self._replay_controller = None
        self._playback_timer = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        # Replay status bar (hidden in live mode)
        yield ReplayStatusBar(id="replay-status")

        # Run header (run identity, connection status)
        # NOTE: Keep id="header" for backwards compatibility with existing integration tests
        yield RunHeader(id="header")

        # Tamiyo Strip (PPO vitals, action summary)
        yield TamiyoStrip(id="tamiyo-strip")

        # Main area with flight board and detail panel
        with Container(id="main-area"):
            yield FlightBoard(id="flight-board")
            yield DetailPanel(id="detail-panel")

        # Event feed (scrollable)
        yield EventFeed(id="event-feed")

        # Help overlay (hidden by default)
        yield HelpOverlay(id="help-overlay", classes="hidden")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Initialize replay controller if replay file provided
        if self._replay_path:
            self._init_replay()
        else:
            # Hide replay status bar in live mode
            self.query_one(ReplayStatusBar).set_visible(False)

        # Set focus to flight board for navigation
        self.query_one(FlightBoard).focus()

    def _init_replay(self) -> None:
        """Initialize replay mode."""
        from esper.karn.overwatch.replay_controller import ReplayController

        if not self._replay_path or not self._replay_path.exists():
            self.notify(f"Replay file not found: {self._replay_path}", severity="error")
            return

        self._replay_controller = ReplayController(self._replay_path)

        if self._replay_controller.total_frames == 0:
            self.notify("No snapshots found in replay file", severity="warning")
            return

        # Load first snapshot
        self._snapshot = self._replay_controller.current_snapshot
        self._update_all_widgets()
        self._update_replay_status()

        self.notify(f"Loaded {self._replay_controller.total_frames} snapshots")

    def _update_replay_status(self) -> None:
        """Update replay status bar."""
        if not self._replay_controller:
            return

        timestamp = ""
        if self._snapshot:
            # Extract time from captured_at
            captured = self._snapshot.captured_at
            if "T" in captured:
                timestamp = captured.split("T")[1][:8]  # HH:MM:SS

        self.query_one(ReplayStatusBar).update_status(
            playing=self._replay_controller.playing,
            speed=self._replay_controller.speed,
            current=self._replay_controller.current_index,
            total=self._replay_controller.total_frames,
            timestamp=timestamp,
        )

    def _update_all_widgets(self) -> None:
        """Update all widgets with current snapshot."""
        if self._snapshot is None:
            return

        # Update run header
        self.query_one(RunHeader).update_snapshot(self._snapshot)

        # Update tamiyo strip
        self.query_one(TamiyoStrip).update_snapshot(self._snapshot)

        # Update flight board
        self.query_one(FlightBoard).update_snapshot(self._snapshot)

        # Update detail panel with tamiyo data
        detail_panel = self.query_one(DetailPanel)
        detail_panel.update_tamiyo(self._snapshot.tamiyo)

        # Update context panel with initial env selection
        board = self.query_one(FlightBoard)
        if board.selected_env_id is not None:
            for env in self._snapshot.flight_board:
                if env.env_id == board.selected_env_id:
                    detail_panel.update_env(env)
                    break

        # Update event feed
        self.query_one(EventFeed).update_events(self._snapshot.event_feed)

    def action_toggle_help(self) -> None:
        """Toggle the help overlay visibility."""
        help_overlay = self.query_one("#help-overlay")
        help_overlay.toggle_class("hidden")
        self._help_visible = not self._help_visible

    def action_dismiss(self) -> None:
        """Dismiss overlays or collapse expanded elements."""
        if self._help_visible:
            self.action_toggle_help()

    def action_show_context(self) -> None:
        """Toggle context panel view."""
        self.query_one(DetailPanel).toggle_mode("context")

    def action_show_tamiyo(self) -> None:
        """Toggle tamiyo detail panel view."""
        self.query_one(DetailPanel).toggle_mode("tamiyo")

    def action_toggle_play(self) -> None:
        """Toggle replay play/pause."""
        if not self._replay_controller:
            return

        self._replay_controller.toggle_play()

        if self._replay_controller.playing:
            self._start_playback()
        else:
            self._stop_playback()

        self._update_replay_status()

    def action_step_forward(self) -> None:
        """Step forward one frame."""
        if not self._replay_controller:
            return

        self._replay_controller.pause()
        self._stop_playback()

        if self._replay_controller.step_forward():
            self._snapshot = self._replay_controller.current_snapshot
            self._update_all_widgets()

        self._update_replay_status()

    def action_step_backward(self) -> None:
        """Step backward one frame."""
        if not self._replay_controller:
            return

        self._replay_controller.pause()
        self._stop_playback()

        if self._replay_controller.step_backward():
            self._snapshot = self._replay_controller.current_snapshot
            self._update_all_widgets()

        self._update_replay_status()

    def action_speed_up(self) -> None:
        """Increase playback speed."""
        if not self._replay_controller:
            return

        self._replay_controller.increase_speed()
        self._update_replay_status()

        # Restart timer with new speed if playing
        if self._replay_controller.playing:
            self._stop_playback()
            self._start_playback()

    def action_speed_down(self) -> None:
        """Decrease playback speed."""
        if not self._replay_controller:
            return

        self._replay_controller.decrease_speed()
        self._update_replay_status()

        # Restart timer with new speed if playing
        if self._replay_controller.playing:
            self._stop_playback()
            self._start_playback()

    def action_toggle_feed(self) -> None:
        """Toggle event feed expanded/compact."""
        self.query_one(EventFeed).toggle_expanded()

    def _start_playback(self) -> None:
        """Start playback timer."""
        if self._playback_timer:
            self._playback_timer.stop()

        interval = self._replay_controller.tick_interval_ms() / 1000.0
        self._playback_timer = self.set_interval(interval, self._playback_tick)

    def _stop_playback(self) -> None:
        """Stop playback timer."""
        if self._playback_timer:
            self._playback_timer.stop()
            self._playback_timer = None

    def _playback_tick(self) -> None:
        """Called on each playback tick."""
        if not self._replay_controller or not self._replay_controller.playing:
            self._stop_playback()
            return

        if self._replay_controller.step_forward():
            self._snapshot = self._replay_controller.current_snapshot
            self._update_all_widgets()
            self._update_replay_status()
        else:
            # Reached end
            self._replay_controller.pause()
            self._stop_playback()
            self._update_replay_status()
            self.notify("Replay complete")

    def on_flight_board_env_selected(self, message: FlightBoard.EnvSelected) -> None:
        """Handle env selection in flight board."""
        self._update_detail_panel_env(message.env_id)

    def on_flight_board_env_expanded(self, message: FlightBoard.EnvExpanded) -> None:
        """Handle env expansion in flight board."""
        pass  # Could update detail panel

    def _update_detail_panel_env(self, env_id: int | None) -> None:
        """Update detail panel with selected env info."""
        if env_id is None or self._snapshot is None:
            self.query_one(DetailPanel).update_env(None)
            return

        # Find the env
        env = None
        for e in self._snapshot.flight_board:
            if e.env_id == env_id:
                env = e
                break

        self.query_one(DetailPanel).update_env(env)
```

**Step 2: Run app tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/app.py
git commit -m "feat(overwatch): wire EventFeed and replay controls into app"
```

---

## Task 6: Integration Tests for Event Feed and Replay

**Files:**
- Modify: `tests/karn/overwatch/test_integration.py`

**Step 1: Add integration tests**

Append to `tests/karn/overwatch/test_integration.py`:

```python
class TestEventFeedIntegration:
    """Integration tests for EventFeed functionality."""

    @pytest.fixture
    def events_replay(self, tmp_path: Path) -> Path:
        """Create replay with event feed data."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            FeedEvent,
        )

        path = tmp_path / "events.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T14:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
                flight_board=[
                    EnvSummary(env_id=0, device_id=0, status="OK"),
                ],
                event_feed=[
                    FeedEvent("14:00:01", "GATE", 0, "Gate G1 passed"),
                    FeedEvent("14:00:02", "PPO", None, "Policy updated: KL=0.015"),
                    FeedEvent("14:00:03", "GERM", 1, "Seed germinated in r0c1"),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_event_feed_displays_events(self, events_replay: Path) -> None:
        """EventFeed displays events from snapshot."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        app = OverwatchApp(replay_path=events_replay)

        async with app.run_test() as pilot:
            feed = app.query_one(EventFeed)
            content = feed.render_events()

            assert "Gate G1 passed" in content
            assert "Policy updated" in content
            assert "Seed germinated" in content

    @pytest.mark.asyncio
    async def test_f_key_toggles_feed_size(self, events_replay: Path) -> None:
        """f key toggles event feed between compact and expanded."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        app = OverwatchApp(replay_path=events_replay)

        async with app.run_test() as pilot:
            feed = app.query_one(EventFeed)
            assert feed.expanded is False

            await pilot.press("f")
            assert feed.expanded is True

            await pilot.press("f")
            assert feed.expanded is False


class TestReplayControlsIntegration:
    """Integration tests for replay controls."""

    @pytest.fixture
    def multi_snapshot_replay(self, tmp_path: Path) -> Path:
        """Create replay with multiple snapshots for navigation testing."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
        )

        path = tmp_path / "multi.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T14:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                    flight_board=[
                        EnvSummary(env_id=0, device_id=0, status="OK"),
                    ],
                )
                writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_replay_status_bar_visible(self, multi_snapshot_replay: Path) -> None:
        """Replay status bar is visible in replay mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        app = OverwatchApp(replay_path=multi_snapshot_replay)

        async with app.run_test() as pilot:
            bar = app.query_one(ReplayStatusBar)
            assert bar.is_visible is True

    @pytest.mark.asyncio
    async def test_step_forward_with_period(self, multi_snapshot_replay: Path) -> None:
        """Period key steps forward through replay."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=multi_snapshot_replay)

        async with app.run_test() as pilot:
            assert app._snapshot.episode == 0

            await pilot.press("period")
            assert app._snapshot.episode == 1

            await pilot.press("period")
            assert app._snapshot.episode == 2

    @pytest.mark.asyncio
    async def test_step_backward_with_comma(self, multi_snapshot_replay: Path) -> None:
        """Comma key steps backward through replay."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=multi_snapshot_replay)

        async with app.run_test() as pilot:
            # Step forward first
            await pilot.press("period")
            await pilot.press("period")
            assert app._snapshot.episode == 2

            await pilot.press("comma")
            assert app._snapshot.episode == 1

    @pytest.mark.asyncio
    async def test_space_toggles_play(self, multi_snapshot_replay: Path) -> None:
        """Space key toggles play/pause."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=multi_snapshot_replay)

        async with app.run_test() as pilot:
            assert app._replay_controller.playing is False

            await pilot.press("space")
            assert app._replay_controller.playing is True

            await pilot.press("space")
            assert app._replay_controller.playing is False

    @pytest.mark.asyncio
    async def test_speed_controls(self, multi_snapshot_replay: Path) -> None:
        """< and > keys adjust playback speed."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=multi_snapshot_replay)

        async with app.run_test() as pilot:
            assert app._replay_controller.speed == 1.0

            await pilot.press("shift+period")
            assert app._replay_controller.speed == 2.0

            await pilot.press("shift+comma")
            assert app._replay_controller.speed == 1.0
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_integration.py -v -k "EventFeed or ReplayControls"`

Expected: All 7 new tests PASS

**Step 3: Commit**

```bash
git add tests/karn/overwatch/test_integration.py
git commit -m "test(overwatch): add integration tests for EventFeed and replay controls"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all Overwatch tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v`

Expected: All tests PASS (190+ tests)

**Step 2: Run linting**

Run: `uv run ruff check src/esper/karn/overwatch/`

Expected: No errors

**Step 3: Final commit if needed**

```bash
git status
# If any uncommitted files:
git add -A
git commit -m "chore(overwatch): Stage 5 complete - Event Feed + Replay"
```

---

## Verification Checklist

- [ ] EventFeed displays timestamped events
- [ ] Event type badges have correct colors (GATE=cyan, PPO=magenta, etc.)
- [ ] `f` key toggles feed between compact (6 lines) and expanded (12 lines)
- [ ] EventFeed filters events by type
- [ ] ReplayStatusBar shows play/pause icon
- [ ] ReplayStatusBar shows speed multiplier
- [ ] ReplayStatusBar shows progress bar and frame counter
- [ ] ReplayController loads all snapshots from file
- [ ] `Space` toggles play/pause
- [ ] `.` steps forward one frame
- [ ] `,` steps backward one frame
- [ ] `>` increases playback speed
- [ ] `<` decreases playback speed
- [ ] Playback auto-advances at correct speed
- [ ] Playback stops at end of file
- [ ] All tests pass (190+)
- [ ] Linting passes

---

## Files Created/Modified

```
src/esper/karn/overwatch/
├── app.py                      # Modified: add replay controls, event feed
├── replay_controller.py        # NEW: playback state machine
└── widgets/
    ├── __init__.py             # Modified: export new widgets
    ├── event_feed.py           # NEW: scrollable event log
    └── replay_status.py        # NEW: playback status bar

tests/karn/overwatch/
├── test_event_feed.py          # NEW: EventFeed tests
├── test_replay_controller.py   # NEW: ReplayController tests
├── test_replay_status.py       # NEW: ReplayStatusBar tests
├── test_widgets.py             # Modified: add export tests
└── test_integration.py         # Modified: add integration tests
```

---

## Next Stage

After Stage 5 is merged, proceed to **Stage 6: Live Telemetry Integration** which will:
- Add TelemetryAggregator to build snapshots from live events
- Add TelemetryListener to receive events from training
- Wire live updates into the app with set_interval polling
- Add `--overwatch` flag to `esper ppo` command
