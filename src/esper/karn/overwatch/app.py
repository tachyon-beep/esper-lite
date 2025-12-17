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
