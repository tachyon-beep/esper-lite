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

        # Run header (run identity, connection status)
        # NOTE: Keep id="header" for backwards compatibility with existing integration tests
        yield RunHeader(id="header")

        # Tamiyo Strip (PPO vitals, action summary)
        yield TamiyoStrip(id="tamiyo-strip")

        # Main area with flight board and detail panel
        with Container(id="main-area"):
            # Real FlightBoard widget
            yield FlightBoard(id="flight-board")

            yield DetailPanel(id="detail-panel")

        # Event feed
        yield Static(
            self._render_event_feed_content(),
            id="event-feed",
        )

        # Help overlay (hidden by default)
        yield HelpOverlay(id="help-overlay", classes="hidden")

        yield Footer()

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

        # Set focus to flight board for navigation
        self.query_one(FlightBoard).focus()

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
            self._update_all_widgets()
        else:
            self.notify("No snapshots found in replay file", severity="warning")

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

        # Update event feed placeholder
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

    def action_show_context(self) -> None:
        """Toggle context panel view."""
        self.query_one(DetailPanel).toggle_mode("context")

    def action_show_tamiyo(self) -> None:
        """Toggle tamiyo detail panel view."""
        self.query_one(DetailPanel).toggle_mode("tamiyo")

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
