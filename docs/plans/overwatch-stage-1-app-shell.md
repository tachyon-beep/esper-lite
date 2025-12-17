# Overwatch Stage 1: Minimal Textual App Shell

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get a Textual app that launches, shows placeholder layout, loads snapshots from JSONL, and exits cleanly.

**Architecture:** Textual `App` with CSS-based grid layout. Five regions (header, tamiyo strip, flight board, detail panel, event feed) rendered as placeholder `Static` widgets. Snapshot data flows from JSONL via `SnapshotReader` (Stage 0) to app state.

**Tech Stack:** Python 3.11, Textual >=0.47.0, argparse CLI, Stage 0 schema classes.

**Prerequisites:**
- Stage 0 must be complete (schema.py, replay.py, test fixtures)
- Branch: `feat/overwatch-textual-ui`

---

## Task 1: Add Textual Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add textual to optional dependencies**

Add a new optional dependency group `overwatch` with textual:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
]
dashboard = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "websockets>=14.0",
]
overwatch = [
    "textual>=0.47.0",
]
```

**Step 2: Install the new dependency**

Run: `uv sync --extra overwatch`

Expected: Textual and dependencies installed successfully

**Step 3: Verify textual imports**

Run: `uv run python -c "from textual.app import App; print('Textual OK')"`

Expected: `Textual OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add textual for overwatch TUI"
```

---

## Task 2: Create Widgets Package Structure

**Files:**
- Create: `src/esper/karn/overwatch/widgets/__init__.py`

**Step 1: Create widgets package**

```python
# src/esper/karn/overwatch/widgets/__init__.py
"""Overwatch TUI Widgets.

Custom Textual widgets for the Overwatch monitoring interface.
"""

__all__: list[str] = []
```

**Step 2: Verify package imports**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.overwatch import widgets; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/widgets/__init__.py
git commit -m "chore(overwatch): create widgets package"
```

---

## Task 3: Create HelpOverlay Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/help.py`
- Create: `tests/karn/overwatch/test_widgets.py`

**Step 1: Write failing test for HelpOverlay**

```python
# tests/karn/overwatch/test_widgets.py
"""Tests for Overwatch TUI widgets."""

from __future__ import annotations

import pytest


class TestHelpOverlay:
    """Tests for HelpOverlay widget."""

    def test_help_overlay_imports(self) -> None:
        """HelpOverlay can be imported."""
        from esper.karn.overwatch.widgets.help import HelpOverlay

        assert HelpOverlay is not None

    def test_help_overlay_has_content(self) -> None:
        """HelpOverlay contains help content."""
        from esper.karn.overwatch.widgets.help import HelpOverlay

        widget = HelpOverlay()
        # Widget should have some renderable content
        assert hasattr(widget, "compose")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestHelpOverlay -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement HelpOverlay**

```python
# src/esper/karn/overwatch/widgets/help.py
"""Help Overlay Widget.

Displays keyboard shortcuts and navigation help.
Press ? to show, Esc to dismiss.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static


HELP_TEXT = """\
[bold cyan]Overwatch Keyboard Shortcuts[/bold cyan]

[bold]Navigation[/bold]
  [cyan]j / ↓[/cyan]    Move down in flight board
  [cyan]k / ↑[/cyan]    Move up in flight board
  [cyan]Enter[/cyan]    Expand selected environment
  [cyan]Esc[/cyan]      Collapse / Close overlay

[bold]Panels[/bold]
  [cyan]c[/cyan]        Show context panel (why flagged)
  [cyan]t[/cyan]        Show Tamiyo detail panel
  [cyan]f[/cyan]        Toggle event feed size

[bold]Replay[/bold]
  [cyan]Space[/cyan]    Play / Pause replay
  [cyan].[/cyan]        Step forward one snapshot
  [cyan],[/cyan]        Step backward one snapshot
  [cyan]< / >[/cyan]    Decrease / Increase playback speed

[bold]General[/bold]
  [cyan]?[/cyan]        Toggle this help overlay
  [cyan]q[/cyan]        Quit Overwatch

[dim]Press Esc to close this help[/dim]
"""


class HelpOverlay(Container):
    """Modal overlay showing keyboard shortcuts.

    Usage:
        # In app compose():
        yield HelpOverlay(id="help-overlay", classes="hidden")

        # Toggle visibility:
        self.query_one("#help-overlay").toggle_class("hidden")
    """

    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    HelpOverlay.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the help content."""
        yield Static(HELP_TEXT, markup=True)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestHelpOverlay -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/help.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): add HelpOverlay widget"
```

---

## Task 4: Create Textual CSS Stylesheet

**Files:**
- Create: `src/esper/karn/overwatch/styles.tcss`

**Step 1: Create the stylesheet with layout grid**

```css
/* src/esper/karn/overwatch/styles.tcss */

/* Overwatch TUI Styles
 *
 * Layout regions:
 * - Header (2 rows): Run identity, connection, health indicators
 * - TamiyoStrip (2 rows): PPO vitals, action summary
 * - MainArea (flexible): FlightBoard + DetailPanel side-by-side
 * - EventFeed (4 rows collapsed, 8 expanded)
 */

/* === Root Layout === */

Screen {
    layout: grid;
    grid-size: 1;
    grid-rows: 2 2 1fr 4;
}

/* === Region Placeholders === */

#header {
    background: $surface;
    border-bottom: solid $primary-darken-2;
    padding: 0 1;
}

#tamiyo-strip {
    background: $surface;
    color: $text;
    border-bottom: solid $secondary;
    padding: 0 1;
}

#main-area {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 2fr 1fr;
}

#flight-board {
    background: $surface;
    border-right: solid $primary-darken-3;
    padding: 0 1;
}

#detail-panel {
    background: $surface;
    padding: 0 1;
}

#event-feed {
    background: $surface;
    border-top: solid $primary-darken-2;
    padding: 0 1;
}

/* === Status Colors === */

.status-ok {
    color: $success;
}

.status-warn {
    color: $warning;
}

.status-crit {
    color: $error;
}

/* === Tamiyo Theme (Magenta) === */

.tamiyo {
    color: #c678dd;
}

.tamiyo-muted {
    color: #8b5a9e;
}

/* === Focus Indicators === */

.focused {
    background: $primary-darken-1;
}

/* === Hidden State === */

.hidden {
    display: none;
}
```

**Step 2: Verify file created**

Run: `ls -la src/esper/karn/overwatch/styles.tcss`

Expected: File exists

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/styles.tcss
git commit -m "feat(overwatch): add Textual CSS stylesheet"
```

---

## Task 5: Create OverwatchApp Class (Minimal)

**Files:**
- Create: `src/esper/karn/overwatch/app.py`
- Create: `tests/karn/overwatch/test_app.py`

**Step 1: Write failing test for OverwatchApp**

```python
# tests/karn/overwatch/test_app.py
"""Tests for OverwatchApp."""

from __future__ import annotations

import pytest


class TestOverwatchApp:
    """Tests for OverwatchApp class."""

    def test_app_imports(self) -> None:
        """OverwatchApp can be imported."""
        from esper.karn.overwatch.app import OverwatchApp

        assert OverwatchApp is not None

    def test_app_has_compose(self) -> None:
        """OverwatchApp has compose method."""
        from esper.karn.overwatch.app import OverwatchApp

        app = OverwatchApp()
        assert hasattr(app, "compose")

    def test_app_has_bindings(self) -> None:
        """OverwatchApp has keyboard bindings."""
        from esper.karn.overwatch.app import OverwatchApp

        app = OverwatchApp()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "q" in binding_keys
        assert "question_mark" in binding_keys
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py::TestOverwatchApp -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement minimal OverwatchApp**

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

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        # Header region (run identity, connection status)
        yield Static(
            self._render_header_content(),
            id="header",
        )

        # Tamiyo Strip (PPO vitals, action summary)
        yield Static(
            self._render_tamiyo_content(),
            id="tamiyo-strip",
        )

        # Main area with flight board and detail panel
        with Container(id="main-area"):
            yield Static(
                self._render_flight_board_content(),
                id="flight-board",
            )
            yield Static(
                self._render_detail_panel_content(),
                id="detail-panel",
            )

        # Event feed
        yield Static(
            self._render_event_feed_content(),
            id="event-feed",
        )

        # Help overlay (hidden by default)
        yield HelpOverlay(id="help-overlay", classes="hidden")

        yield Footer()

    def _render_header_content(self) -> str:
        """Render header placeholder content."""
        if self._snapshot:
            ts = self._snapshot.captured_at
            run_id = self._snapshot.run_id or "unknown"
            task = self._snapshot.task_name or "unknown"
            return f"[HEADER] Run: {run_id} | Task: {task} | Snapshot: {ts}"
        return "[HEADER] Waiting for data..."

    def _render_tamiyo_content(self) -> str:
        """Render Tamiyo Strip placeholder content."""
        if self._snapshot and self._snapshot.tamiyo:
            kl = self._snapshot.tamiyo.kl_divergence
            ent = self._snapshot.tamiyo.entropy
            return f"[TAMIYO] KL: {kl:.3f} | Entropy: {ent:.2f}"
        return "[TAMIYO STRIP] Waiting for policy data..."

    def _render_flight_board_content(self) -> str:
        """Render Flight Board placeholder content."""
        if self._snapshot and self._snapshot.flight_board:
            n = len(self._snapshot.flight_board)
            return f"[FLIGHT BOARD] {n} environments loaded"
        return "[FLIGHT BOARD] No environments"

    def _render_detail_panel_content(self) -> str:
        """Render Detail Panel placeholder content."""
        return "[DETAIL PANEL] Select an environment"

    def _render_event_feed_content(self) -> str:
        """Render Event Feed placeholder content."""
        if self._snapshot and self._snapshot.event_feed:
            n = len(self._snapshot.event_feed)
            return f"[EVENT FEED] {n} events"
        return "[EVENT FEED] No events"

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Load initial snapshot if replay file provided
        if self._replay_path:
            self._load_first_snapshot()

    def _load_first_snapshot(self) -> None:
        """Load the first snapshot from replay file."""
        from esper.karn.overwatch.replay import SnapshotReader

        if not self._replay_path or not self._replay_path.exists():
            self.notify(f"Replay file not found: {self._replay_path}", severity="error")
            return

        reader = SnapshotReader(self._replay_path)
        for snapshot in reader:
            self._snapshot = snapshot
            break  # Take first snapshot only

        if self._snapshot:
            self.notify(f"Loaded snapshot from {self._snapshot.captured_at}")
            # Refresh all placeholders
            self._refresh_placeholders()
        else:
            self.notify("No snapshots found in replay file", severity="warning")

    def _refresh_placeholders(self) -> None:
        """Refresh all placeholder widgets with current snapshot."""
        self.query_one("#header", Static).update(self._render_header_content())
        self.query_one("#tamiyo-strip", Static).update(self._render_tamiyo_content())
        self.query_one("#flight-board", Static).update(self._render_flight_board_content())
        self.query_one("#detail-panel", Static).update(self._render_detail_panel_content())
        self.query_one("#event-feed", Static).update(self._render_event_feed_content())

    def action_toggle_help(self) -> None:
        """Toggle the help overlay visibility."""
        help_overlay = self.query_one("#help-overlay")
        help_overlay.toggle_class("hidden")
        self._help_visible = not self._help_visible

    def action_dismiss(self) -> None:
        """Dismiss overlays or collapse expanded elements."""
        if self._help_visible:
            self.action_toggle_help()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py::TestOverwatchApp -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/app.py tests/karn/overwatch/test_app.py
git commit -m "feat(overwatch): add OverwatchApp with placeholder layout"
```

---

## Task 6: Update Package Exports for App

**Files:**
- Modify: `src/esper/karn/overwatch/__init__.py`
- Modify: `src/esper/karn/overwatch/widgets/__init__.py`

**Step 1: Write failing test for imports**

```python
# tests/karn/overwatch/test_app.py (append to file)

class TestPackageExports:
    """Tests for package-level exports."""

    def test_app_importable_from_package(self) -> None:
        """OverwatchApp importable from overwatch package."""
        from esper.karn.overwatch import OverwatchApp

        assert OverwatchApp is not None

    def test_help_overlay_importable_from_widgets(self) -> None:
        """HelpOverlay importable from widgets package."""
        from esper.karn.overwatch.widgets import HelpOverlay

        assert HelpOverlay is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py::TestPackageExports -v`

Expected: FAIL with `ImportError`

**Step 3: Update widgets __init__.py**

```python
# src/esper/karn/overwatch/widgets/__init__.py
"""Overwatch TUI Widgets.

Custom Textual widgets for the Overwatch monitoring interface.
"""

from esper.karn.overwatch.widgets.help import HelpOverlay

__all__ = [
    "HelpOverlay",
]
```

**Step 4: Update overwatch __init__.py**

```python
# src/esper/karn/overwatch/__init__.py
"""Overwatch - Textual TUI for Esper training monitoring.

Provides real-time visibility into training environments, seed lifecycle,
and Tamiyo decision-making.
"""

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    EnvSummary,
    SlotChipState,
    TamiyoState,
    ConnectionStatus,
    DeviceVitals,
    FeedEvent,
)

from esper.karn.overwatch.replay import (
    SnapshotWriter,
    SnapshotReader,
)

from esper.karn.overwatch.app import OverwatchApp

__all__ = [
    # Schema
    "TuiSnapshot",
    "EnvSummary",
    "SlotChipState",
    "TamiyoState",
    "ConnectionStatus",
    "DeviceVitals",
    "FeedEvent",
    # Replay
    "SnapshotWriter",
    "SnapshotReader",
    # App
    "OverwatchApp",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py::TestPackageExports -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/overwatch/__init__.py src/esper/karn/overwatch/widgets/__init__.py tests/karn/overwatch/test_app.py
git commit -m "feat(overwatch): export OverwatchApp and HelpOverlay"
```

---

## Task 7: Create CLI Entry Point Script

**Files:**
- Create: `src/esper/scripts/overwatch.py`

**Step 1: Write failing test for CLI**

```python
# tests/karn/overwatch/test_cli.py
"""Tests for Overwatch CLI entry point."""

from __future__ import annotations

import pytest


class TestOverwatchCLI:
    """Tests for overwatch CLI."""

    def test_cli_module_imports(self) -> None:
        """CLI module can be imported."""
        from esper.scripts import overwatch

        assert hasattr(overwatch, "main")
        assert hasattr(overwatch, "build_parser")

    def test_parser_has_replay_arg(self) -> None:
        """Parser accepts --replay argument."""
        from esper.scripts.overwatch import build_parser

        parser = build_parser()
        args = parser.parse_args(["--replay", "test.jsonl"])

        assert args.replay == "test.jsonl"

    def test_parser_replay_required_message(self) -> None:
        """Parser shows helpful message when --replay missing."""
        from esper.scripts.overwatch import build_parser

        parser = build_parser()
        # In Stage 1, --replay is required (live mode comes in Stage 6)
        with pytest.raises(SystemExit):
            parser.parse_args([])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_cli.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement CLI script**

```python
#!/usr/bin/env python3
# src/esper/scripts/overwatch.py
"""Overwatch TUI Entry Point.

Launch the Overwatch training monitor.

Usage:
    # Replay mode (Stage 1+)
    uv run python -m esper.scripts.overwatch --replay training.jsonl

    # Live mode (Stage 6)
    uv run python -m esper.scripts.overwatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="overwatch",
        description="Esper Overwatch - Training Monitor TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View a replay file
    python -m esper.scripts.overwatch --replay training.jsonl

    # Live monitoring (requires training to be running)
    python -m esper.scripts.overwatch  # Coming in Stage 6

Keyboard shortcuts:
    q       Quit
    ?       Toggle help overlay
    j/k     Navigate flight board
    Enter   Expand environment
    Esc     Collapse / Dismiss
""",
    )

    parser.add_argument(
        "--replay",
        type=str,
        required=True,  # Required in Stage 1, optional in Stage 6
        metavar="FILE",
        help="Path to JSONL replay file (from SnapshotWriter)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay playback speed multiplier (default: 1.0)",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate replay file exists
    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"Error: Replay file not found: {replay_path}", file=sys.stderr)
        return 1

    if not replay_path.suffix == ".jsonl":
        print(f"Warning: Expected .jsonl file, got: {replay_path.suffix}", file=sys.stderr)

    # Import app here to avoid slow import on --help
    try:
        from esper.karn.overwatch import OverwatchApp
    except ImportError as e:
        print(f"Error: Failed to import Overwatch: {e}", file=sys.stderr)
        print("Hint: Install with `uv sync --extra overwatch`", file=sys.stderr)
        return 1

    # Launch the app
    app = OverwatchApp(replay_path=replay_path)
    app.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_cli.py -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/esper/scripts/overwatch.py tests/karn/overwatch/test_cli.py
git commit -m "feat(overwatch): add CLI entry point script"
```

---

## Task 8: Integration Test - App Launch with Fixture

**Files:**
- Create: `tests/karn/overwatch/test_integration.py`

**Step 1: Write integration test using Textual pilot**

```python
# tests/karn/overwatch/test_integration.py
"""Integration tests for Overwatch TUI."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestOverwatchIntegration:
    """Integration tests for the full app."""

    @pytest.fixture
    def sample_replay_path(self, tmp_path: Path) -> Path:
        """Create a sample replay file for testing."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "test_replay.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    kl_divergence=0.019,
                    entropy=1.24,
                    action_counts={"GERMINATE": 10, "BLEND": 20},
                ),
                run_id="test-run-001",
                task_name="cifar10",
                episode=47,
                flight_board=[
                    EnvSummary(
                        env_id=0,
                        device_id=0,
                        status="OK",
                        slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
                    ),
                    EnvSummary(
                        env_id=1,
                        device_id=1,
                        status="WARN",
                        anomaly_score=0.65,
                    ),
                ],
                envs_ok=1,
                envs_warn=1,
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_app_launches_with_replay(self, sample_replay_path: Path) -> None:
        """App launches and loads replay file."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            # App should have loaded the snapshot
            assert app._snapshot is not None
            assert app._snapshot.run_id == "test-run-001"

            # Header should show snapshot info
            header = app.query_one("#header")
            assert "test-run-001" in header.renderable

    @pytest.mark.asyncio
    async def test_app_help_toggle(self, sample_replay_path: Path) -> None:
        """? key toggles help overlay."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            # Help should be hidden initially
            help_overlay = app.query_one("#help-overlay")
            assert "hidden" in help_overlay.classes

            # Press ? to show help
            await pilot.press("question_mark")
            assert "hidden" not in help_overlay.classes

            # Press Esc to hide help
            await pilot.press("escape")
            assert "hidden" in help_overlay.classes

    @pytest.mark.asyncio
    async def test_app_quit(self, sample_replay_path: Path) -> None:
        """q key quits the app."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should be exiting
            assert app._exit


class TestAppWithoutReplay:
    """Tests for app behavior without replay file."""

    @pytest.mark.asyncio
    async def test_app_launches_without_data(self) -> None:
        """App launches even without replay file."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp()

        async with app.run_test() as pilot:
            # Should show placeholder content
            header = app.query_one("#header")
            assert "Waiting" in header.renderable
```

**Step 2: Add pytest-asyncio dependency for async tests**

Add to dev dependencies in pyproject.toml:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
]
```

**Step 3: Run uv sync to install**

Run: `uv sync --extra dev --extra overwatch`

**Step 4: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_integration.py -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add tests/karn/overwatch/test_integration.py pyproject.toml uv.lock
git commit -m "test(overwatch): add integration tests with Textual pilot"
```

---

## Task 9: Manual Verification with Test Fixture

**Step 1: Run app with healthy_run fixture**

Run: `PYTHONPATH=src uv run python -m esper.scripts.overwatch --replay tests/karn/overwatch/fixtures/healthy_run.jsonl`

Expected:
- TUI launches successfully
- Header shows snapshot timestamp
- Five regions visible (header, tamiyo strip, flight board, detail panel, event feed)
- `?` shows help overlay
- `Esc` dismisses help
- `q` quits cleanly

**Step 2: Run app with anomaly fixture**

Run: `PYTHONPATH=src uv run python -m esper.scripts.overwatch --replay tests/karn/overwatch/fixtures/anomaly_detected.jsonl`

Expected:
- TUI launches successfully
- Header shows different snapshot data
- Placeholder text reflects loaded data

**Step 3: Document any issues**

If any issues are found, note them for the next task.

---

## Task 10: Run Full Test Suite and Final Commit

**Step 1: Run all Overwatch tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v`

Expected: All tests PASS (~40+ tests including Stage 0)

**Step 2: Run linting**

Run: `uv run ruff check src/esper/karn/overwatch/ src/esper/scripts/overwatch.py`

Expected: No errors (fix any that appear)

**Step 3: Run type checking**

Run: `uv run mypy src/esper/karn/overwatch/ src/esper/scripts/overwatch.py --ignore-missing-imports`

Expected: No errors (or only minor issues)

**Step 4: Final summary commit**

```bash
git add -A
git status
```

If any uncommitted changes:
```bash
git commit -m "chore(overwatch): Stage 1 complete - minimal Textual app shell"
```

---

## Verification Checklist

- [ ] `uv run python -m esper.scripts.overwatch --replay FILE` launches TUI
- [ ] Layout shows 5 regions (header, tamiyo strip, flight board, detail panel, feed)
- [ ] Snapshot timestamp displayed in header
- [ ] Snapshot data (task name, run_id) displayed
- [ ] `q` exits cleanly
- [ ] `?` shows help overlay
- [ ] `Esc` dismisses help overlay
- [ ] All tests pass (~40+ including Stage 0)
- [ ] Linting passes
- [ ] Can be merged to main independently

---

## Files Created/Modified

```
src/esper/karn/overwatch/
├── __init__.py          # Updated: export OverwatchApp
├── schema.py            # (Stage 0)
├── replay.py            # (Stage 0)
├── app.py               # NEW: OverwatchApp
├── styles.tcss          # NEW: Textual CSS
└── widgets/
    ├── __init__.py      # NEW: export HelpOverlay
    └── help.py          # NEW: HelpOverlay widget

src/esper/scripts/
└── overwatch.py         # NEW: CLI entry point

tests/karn/overwatch/
├── __init__.py
├── conftest.py          # (Stage 0)
├── test_schema.py       # (Stage 0)
├── test_replay.py       # (Stage 0)
├── test_widgets.py      # NEW: widget tests
├── test_app.py          # NEW: app tests
├── test_cli.py          # NEW: CLI tests
├── test_integration.py  # NEW: integration tests
└── fixtures/            # (Stage 0)

pyproject.toml           # Modified: add textual, pytest-asyncio
```

---

## Next Stage

After Stage 1 is merged, proceed to **Stage 2: Flight Board** which will:
- Replace placeholder with real `FlightBoard` widget
- Implement `EnvRow` and `SlotChip` widgets
- Add navigation (j/k, Enter/Esc)
- Implement anomaly sorting with hysteresis
