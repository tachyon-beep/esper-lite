"""Overwatch Textual Application.

Main application class for the Overwatch TUI monitoring interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.flight_board import FlightBoard
from esper.karn.overwatch.widgets.run_header import RunHeader
from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip
from esper.karn.overwatch.widgets.detail_panel import DetailPanel
from esper.karn.overwatch.widgets.event_feed import EventFeed
from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot
    from esper.karn.overwatch.backend import OverwatchBackend


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
        backend: "OverwatchBackend | None" = None,
        poll_interval_ms: int = 250,
        **kwargs,
    ) -> None:
        """Initialize the Overwatch app.

        Args:
            replay_path: Optional path to JSONL replay file
            backend: Optional OverwatchBackend for live mode
            poll_interval_ms: Polling interval for live updates (default: 250ms)
            **kwargs: Additional args passed to App
        """
        super().__init__(**kwargs)
        self._replay_path = Path(replay_path) if replay_path else None
        self._backend = backend
        self._poll_interval_ms = poll_interval_ms
        self._snapshot: TuiSnapshot | None = None
        self._help_visible = False
        self._replay_controller = None
        self._playback_timer = None
        self._live_timer = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        # Replay status bar (hidden in live mode)
        yield ReplayStatusBar(id="replay-status")

        # Run header (run identity, connection status)
        # NOTE: id="header" required by integration tests (test_app.py, test_integration.py)
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
        elif self._backend:
            # Live mode
            self._init_live()
        else:
            # Demo/standalone mode - hide replay bar
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

    def _init_live(self) -> None:
        """Initialize live telemetry mode."""
        # Hide replay status bar in live mode
        self.query_one(ReplayStatusBar).set_visible(False)

        # Start polling timer
        interval = self._poll_interval_ms / 1000.0
        self._live_timer = self.set_interval(interval, self._live_poll)

        # Initial poll
        self._live_poll()

    def _live_poll(self) -> None:
        """Poll backend for latest snapshot."""
        if not self._backend:
            return

        snapshot = self._backend.get_snapshot()

        # Only update if snapshot changed (by captured_at timestamp)
        if snapshot.captured_at != getattr(self._snapshot, "captured_at", None):
            self._snapshot = snapshot
            self._update_all_widgets()
