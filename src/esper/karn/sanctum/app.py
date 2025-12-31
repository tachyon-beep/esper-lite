"""Sanctum Textual Application.

Developer diagnostic TUI for debugging PPO training runs.
Layout matches existing Rich TUI (tui.py _render() method).

LAYOUT FIX: TamiyoBrain spans full width as dedicated row (size=11),
NOT embedded in right column. Event Log included at bottom-left.

UNICODE GLYPH REQUIREMENTS:
    Sanctum requires a terminal with Unicode support and a font that includes:

    Status icons:     ● ◐ ○ ★ ✓ ✗ ⚠ ▼ ▲
    Progress bars:    █ ░
    Sparklines:       ▁ ▂ ▃ ▄ ▅ ▆ ▇ █
    Arrows:           ↑ ↓ → ↗ ↘ ▶ ▸
    Alpha curves:     ⌒ ⌢ ⌣ −
    Medals:           🥇 🥈 🥉 📌
    Severity:         💀 🔥 ⚠️
    Separators:       │ ─

    Recommended terminals: iTerm2, Kitty, Windows Terminal, Alacritty
    Recommended fonts: JetBrains Mono, Fira Code, Cascadia Code, Nerd Fonts

    For terminals without full Unicode/emoji support, use Overwatch (web UI) instead.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Input, Static

from esper.karn.sanctum.errors import SanctumTelemetryFatalError
from esper.karn.sanctum.widgets import (
    AnomalyStrip,
    EnvDetailScreen,
    EnvOverview,
    EventLog,
    EventLogDetail,
    HistoricalEnvDetail,
    RewardHealthPanel,
    RunHeader,
    Scoreboard,
    TamiyoBrain,
    ThreadDeathModal,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.schema import SanctumSnapshot
    from textual.timer import Timer
    from esper.karn.sanctum.widgets.reward_health import RewardHealthData


HELP_TEXT = """\
[bold cyan]Sanctum Keyboard Shortcuts[/bold cyan]

[bold]Navigation[/bold]
  [cyan]h/l[/cyan] [cyan]←/→[/cyan]  Switch between left/right panels
  [cyan]j/k[/cyan] [cyan]↑/↓[/cyan]  Navigate rows in table
  [cyan]g/G[/cyan]       Jump to top/bottom
  [cyan]1-9, 0[/cyan]    Jump to environment 0-9
  [cyan]Tab[/cyan]       Cycle to next panel
  [cyan]Enter[/cyan]     Open detail modal for selected item
  [cyan]d[/cyan]         Open env detail (same as Enter)

[bold]Actions[/bold]
  [cyan]/[/cyan]         Filter envs (by ID or status)
  [cyan]Esc[/cyan]       Clear filter
  [cyan]i[/cyan]         Show full run info (untruncated)
  [cyan]r[/cyan]         Manual refresh
  [cyan]q[/cyan]         Quit Sanctum

[bold]Pinning (in Best Runs / Decisions)[/bold]
  [cyan]p[/cyan]         Toggle pin on selected Best Runs item
  [cyan]Click[/cyan]     Click decision card to toggle pin (📌)
  [dim]Pinned items are never removed from the display[/dim]

[bold]Click Actions[/bold]
  [cyan]Click[/cyan]     Event Log → raw event detail view
  [cyan]Click[/cyan]     Best Runs row → historical env snapshot

[bold]In Detail Modal[/bold]
  [cyan]Esc[/cyan]       Close modal
  [cyan]q[/cyan]         Close modal

[bold]Status Icons[/bold]
  [green]●[/green] OK     [yellow]◐[/yellow] Warning    [red]○[/red] Error
  [green]★[/green] Excellent   [green]✓[/green] Improving
  [yellow]⚠[/yellow] Stalling    [red]✗[/red] Severely stalled

[dim]Press Esc, ?, Q, or click to close[/dim]
"""


@dataclass(frozen=True)
class SanctumView:
    """All data needed to render one UI refresh tick."""

    primary: "SanctumSnapshot"
    snapshots_by_group: dict[str, "SanctumSnapshot"]
    reward_health: "RewardHealthData"


class HelpScreen(ModalScreen[None]):
    """Help overlay showing keyboard shortcuts."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("?", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
        background: $surface-darken-1 80%;
    }

    HelpScreen > #help-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the help screen."""
        with Container(id="help-container"):
            yield Static(HELP_TEXT)

    def on_click(self) -> None:
        """Dismiss help screen on click."""
        self.dismiss()


class RunInfoScreen(ModalScreen[None]):
    """Modal showing full run info without truncation."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("i", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    RunInfoScreen {
        align: center middle;
        background: $surface-darken-1 80%;
    }

    RunInfoScreen > #info-container {
        width: 80;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    """

    def __init__(self, snapshot: "SanctumSnapshot") -> None:
        super().__init__()
        self._snapshot = snapshot

    def compose(self) -> ComposeResult:
        """Compose the run info screen."""
        s = self._snapshot
        runtime_str = self._format_runtime(s.runtime_seconds)

        info_text = f"""\
[bold cyan]Run Information[/bold cyan]

[bold]Task Name[/bold]
  [cyan]{s.task_name or '(not set)'}[/cyan]

[bold]Progress[/bold]
  Episode:    {s.current_episode}
  Epoch:      {s.current_epoch} / {s.max_epochs if s.max_epochs > 0 else '∞'}
  Batch:      {s.current_batch} / {s.max_batches}
  Runtime:    {runtime_str}

[bold]Throughput[/bold]
  Epochs/sec:   {s.vitals.epochs_per_second:.2f}
  Batches/hr:   {s.vitals.batches_per_hour:.1f}

[bold]System[/bold]
  Connected:    {'Yes' if s.connected else 'No'}
  Staleness:    {s.staleness_seconds:.1f}s
  Thread:       {'Alive' if s.training_thread_alive else 'Dead' if s.training_thread_alive is False else 'Unknown'}

[dim]Press Esc, i, or q to close[/dim]
"""
        with Container(id="info-container"):
            yield Static(info_text)

    def _format_runtime(self, seconds: float) -> str:
        """Format runtime as Xh Ym Zs."""
        if seconds <= 0:
            return "--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def on_click(self) -> None:
        """Dismiss on click."""
        self.dismiss()


class TelemetryFatalScreen(ModalScreen[None]):
    """Fatal telemetry error screen.

    Sanctum is a developer diagnostics surface. If telemetry contracts break,
    we stop refreshing and surface the full traceback loudly.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    TelemetryFatalScreen {
        align: center middle;
        background: $error-darken-3 90%;
    }

    TelemetryFatalScreen > #fatal-container {
        width: 95%;
        height: 95%;
        background: $surface;
        border: thick $error;
        padding: 1 2;
    }

    TelemetryFatalScreen .fatal-title {
        height: 1;
        text-align: center;
        color: $error;
        text-style: bold;
        margin-bottom: 1;
    }

    TelemetryFatalScreen .fatal-message {
        height: auto;
        margin-bottom: 1;
    }

    TelemetryFatalScreen #trace-scroll {
        height: 1fr;
        border: solid $error-darken-1;
        padding: 0 1;
    }
    """

    def __init__(self, error: SanctumTelemetryFatalError) -> None:
        super().__init__()
        self._error = error

    def compose(self) -> ComposeResult:
        with Container(id="fatal-container"):
            yield Static("TELEMETRY FAILURE", classes="fatal-title")
            yield Static(str(self._error), classes="fatal-message")
            with VerticalScroll(id="trace-scroll"):
                yield Static(self._error.traceback)


class SanctumApp(App[None]):
    """Sanctum diagnostic TUI for Esper training.

    Provides deep inspection of PPO training for debugging.
    Layout mirrors existing Rich TUI for 1:1 port.

    TERMINAL SIZE CONSTRAINTS:
    - Minimum: 120x40 (width x height) for readable display
    - Recommended: 140x50 or larger
    - TamiyoBrain requires width ≥ 80 for 4-column layout

    Args:
        backend: SanctumBackend providing snapshot data.
        num_envs: Number of training environments.
        refresh_rate: Snapshot refresh rate in Hz (default: 4).
    """

    TITLE = "Sanctum - Developer Diagnostics"
    SUB_TITLE = "Esper Training Debugger"

    CSS_PATH = "styles.tcss"

    view: SanctumView | None = reactive(None)

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "show_env_detail", "Detail", show=True),
        Binding("tab", "focus_next", "Next Panel", show=False),
        Binding("shift+tab", "focus_previous", "Prev Panel", show=False),
        # Vim-style navigation
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("g", "cursor_top", "Top", show=False),
        Binding("G", "cursor_bottom", "Bottom", show=False),
        # Number keys for quick env focus
        Binding("1", "focus_env(0)", "Env 0", show=False),
        Binding("2", "focus_env(1)", "Env 1", show=False),
        Binding("3", "focus_env(2)", "Env 2", show=False),
        Binding("4", "focus_env(3)", "Env 3", show=False),
        Binding("5", "focus_env(4)", "Env 4", show=False),
        Binding("6", "focus_env(5)", "Env 5", show=False),
        Binding("7", "focus_env(6)", "Env 6", show=False),
        Binding("8", "focus_env(7)", "Env 7", show=False),
        Binding("9", "focus_env(8)", "Env 8", show=False),
        Binding("0", "focus_env(9)", "Env 9", show=False),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("i", "show_run_info", "Info", show=True),
        Binding("?", "toggle_help", "Help", show=True),
        # Filter
        Binding("/", "start_filter", "Filter", show=True),
        Binding("escape", "clear_filter", "Clear", show=False, priority=True),
        # Panel switching
        Binding("h", "focus_left_panel", "Left Panel", show=False),
        Binding("l", "focus_right_panel", "Right Panel", show=False),
        Binding("left", "focus_left_panel", "Left Panel", show=False),
        Binding("right", "focus_right_panel", "Right Panel", show=False),
    ]

    def __init__(
        self,
        backend: "SanctumBackend",
        num_envs: int = 16,
        refresh_rate: float = 4.0,
        training_thread: threading.Thread | None = None,
        shutdown_event: threading.Event | None = None,
    ):
        """Initialize Sanctum app.

        Args:
            backend: SanctumBackend providing snapshot data.
            num_envs: Number of training environments.
            refresh_rate: Snapshot refresh rate in Hz.
            training_thread: Optional training thread to monitor.
            shutdown_event: Event to signal training to stop gracefully.
        """
        super().__init__()
        self._backend = backend
        self._num_envs = num_envs
        self._refresh_interval = 1.0 / refresh_rate
        self._focused_env_id: int = 0
        self._snapshot: "SanctumSnapshot" | None = None
        self._poll_count = 0  # Debug: track timer fires
        self._training_thread = training_thread  # Monitor thread status
        self._shutdown_event = shutdown_event  # Signal graceful shutdown
        self._filter_active = False  # Track filter input visibility
        self._thread_death_shown = False  # Track if we've shown death modal
        self._shutdown_requested = False  # Track if shutdown was requested
        self._telemetry_fatal_shown = False
        self._refresh_timer: "Timer | None" = None
        self._pending_view: SanctumView | None = None
        self._apply_view_scheduled = False

    def compose(self) -> ComposeResult:
        """Build the Sanctum layout.

        Layout structure:
        - Run Header: Episode, Epoch, Batch, Runtime, Best Accuracy, Connection, A/B comparison
        - Anomaly Strip: Single-line automatic problem surfacing
        - Top row: EnvOverview (80%) | Scoreboard (20%)
        - Bottom row: TamiyoBrain (80%) | EventLog (20%)
        - Footer: Keybindings
        """
        yield RunHeader(id="run-header")
        yield AnomalyStrip(id="anomaly-strip")

        # Filter input - hidden by default, shown when '/' pressed
        yield Input(
            placeholder="Filter: env ID or status (stalled, healthy...)",
            id="filter-input",
            classes="hidden",
        )

        with Container(id="sanctum-main"):
            # Top section: Environment Overview and Scoreboard
            with Horizontal(id="top-section"):
                yield EnvOverview(num_envs=self._num_envs, id="env-overview")
                with Vertical(id="metrics-column"):
                    yield Scoreboard(id="scoreboard")
                    yield RewardHealthPanel(id="metrics-reward-health")

            # Bottom section: TamiyoBrain (left) | Event Log (right)
            with Horizontal(id="bottom-section"):
                # Left side: TamiyoBrain container (70%) - widgets created dynamically
                with Horizontal(id="tamiyo-container"):
                    pass  # TamiyoBrain widgets mounted dynamically
                # Right side: Event Log (30%)
                yield EventLog(id="event-log")

        yield Footer()

    def on_mount(self) -> None:
        """Start refresh timer when app mounts."""
        self._refresh_timer = self.set_interval(self._refresh_interval, self._poll_and_refresh)

    def _get_or_create_tamiyo_widget(self, group_id: str) -> TamiyoBrain:
        """Get or create TamiyoBrain widget for a policy group.

        Args:
            group_id: Policy group identifier (e.g., "A", "B", "default")

        Returns:
            TamiyoBrain widget for this group.
        """
        widget_id = f"tamiyo-{group_id.lower()}"
        css_class = f"group-{group_id.lower()}"

        try:
            return self.query_one(f"#{widget_id}", TamiyoBrain)
        except NoMatches:
            # Create new widget and mount it
            widget = TamiyoBrain(id=widget_id, classes=css_class)
            try:
                container = self.query_one("#tamiyo-container")
                container.mount(widget)
                return widget
            except NoMatches:
                self.log.warning(f"Cannot mount TamiyoBrain for {group_id}: container not found")
                raise

    def _refresh_tamiyo_widgets(self, snapshots: dict[str, "SanctumSnapshot"]) -> None:
        """Refresh TamiyoBrain widgets from multi-group snapshots."""

        # Handle empty case (no events yet) - create default widget
        if not snapshots:
            try:
                widget = self._get_or_create_tamiyo_widget("default")
                from esper.karn.sanctum.schema import SanctumSnapshot as SnapshotClass
                widget.update_snapshot(SnapshotClass())
            except NoMatches:
                pass
            return

        # Create/update widget for each group
        for group_id, group_snapshot in snapshots.items():
            try:
                widget = self._get_or_create_tamiyo_widget(group_id)
                widget.update_snapshot(group_snapshot)
            except NoMatches:
                pass  # Container hasn't mounted yet

    def _show_telemetry_fatal(self, error: SanctumTelemetryFatalError) -> None:
        """Stop refreshing and surface telemetry failures loudly."""
        if self._telemetry_fatal_shown:
            return
        self._telemetry_fatal_shown = True

        if self._refresh_timer is not None:
            self._refresh_timer.pause()

        self.push_screen(TelemetryFatalScreen(error))

    def _poll_and_refresh(self) -> None:
        """Poll backend for new snapshot and refresh all panels.

        Called periodically by set_interval timer.
        Thread-safe: backend.get_snapshot() is thread-safe.
        """
        self._poll_count += 1

        if self._backend is None:
            self.log.warning("Backend is None, skipping refresh")
            return

        try:
            snapshots_by_group = self._backend.get_all_snapshots()
            reward_health = self._backend.compute_reward_health()
        except SanctumTelemetryFatalError as e:
            self._show_telemetry_fatal(e)
            return
        except Exception as e:
            self._show_telemetry_fatal(
                SanctumTelemetryFatalError(
                    f"Sanctum refresh failed: {e}",
                    traceback.format_exc(),
                )
            )
            return

        # Choose primary snapshot (used by non-policy-group widgets).
        if "default" in snapshots_by_group:
            primary = snapshots_by_group["default"]
        elif snapshots_by_group:
            primary = snapshots_by_group[sorted(snapshots_by_group.keys())[0]]
        else:
            from esper.karn.sanctum.schema import SanctumSnapshot

            primary = SanctumSnapshot()

        # Debug: Add poll count to snapshots for display
        primary.poll_count = self._poll_count
        for snapshot in snapshots_by_group.values():
            snapshot.poll_count = self._poll_count

        # Debug: Check if training thread is still alive
        thread_alive = self._training_thread.is_alive() if self._training_thread else None
        primary.training_thread_alive = thread_alive
        for snapshot in snapshots_by_group.values():
            snapshot.training_thread_alive = thread_alive

        # Check if training thread died (and we haven't shown modal yet)
        if thread_alive is False and not self._thread_death_shown:
            self._thread_death_shown = True
            self.push_screen(ThreadDeathModal())
            self.log.error("Training thread died! Showing death modal.")

        # Debug: Log snapshot state (visible in Textual console with Ctrl+Shift+D)
        self.log.info(
            f"Poll #{self._poll_count}: connected={primary.connected}, "
            f"ep={primary.current_episode}, "
            f"events={len(primary.event_log)}, "
            f"total_events={primary.total_events_received}, "
            f"thread_alive={thread_alive}"
        )

        try:
            self.view = SanctumView(
                primary=primary,
                snapshots_by_group=snapshots_by_group,
                reward_health=reward_health,
            )
        except SanctumTelemetryFatalError as e:
            self._show_telemetry_fatal(e)
        except Exception as e:
            self._show_telemetry_fatal(
                SanctumTelemetryFatalError(
                    f"Sanctum render update failed: {e}",
                    traceback.format_exc(),
                )
            )

    def watch_view(self, view: SanctumView | None) -> None:
        """Apply latest view model to widgets."""
        self._pending_view = view
        if self._apply_view_scheduled:
            return
        self._apply_view_scheduled = True
        self.call_after_refresh(self._apply_pending_view)

    def _apply_pending_view(self) -> None:
        """Apply the most recent view after the next refresh/layout pass."""
        self._apply_view_scheduled = False
        view = self._pending_view
        if view is None:
            return
        self._snapshot = view.primary
        self._apply_view(view)

    def _apply_view(self, view: SanctumView) -> None:
        """Refresh all panels from a single, consistent view model.

        Args:
            view: The full view model for this tick.
        """
        snapshot = view.primary
        # Update run header first (most important context)
        try:
            self.query_one("#run-header", RunHeader).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet

        # Update anomaly strip (after run header)
        try:
            self.query_one("#anomaly-strip", AnomalyStrip).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet

        # Update each widget - query by ID and call update_snapshot
        try:
            self.query_one("#env-overview", EnvOverview).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet

        try:
            self.query_one("#scoreboard", Scoreboard).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet

        # Update reward health panels (metrics column and TamiyoBrain)
        try:
            self.query_one("#metrics-reward-health", RewardHealthPanel).update_data(view.reward_health)
        except NoMatches:
            pass  # Widget hasn't mounted yet

        # Update TamiyoBrain widgets using multi-group snapshots
        self._refresh_tamiyo_widgets(view.snapshots_by_group)

        try:
            self.query_one("#event-log", EventLog).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet

        # Update EnvDetailScreen modal if displayed
        # Check if we have a modal screen on the stack
        if len(self.screen_stack) > 1:
            current_screen = self.screen_stack[-1]
            if isinstance(current_screen, EnvDetailScreen):
                # Get updated env state from snapshot
                env = snapshot.envs.get(current_screen.env_id)
                if env is not None:
                    current_screen.update_env_state(env)

    def action_focus_env(self, env_id: int) -> None:
        """Focus on specific environment for detail panels.

        Args:
            env_id: Environment ID to focus (0-indexed).
        """
        if 0 <= env_id < self._num_envs:
            self._focused_env_id = env_id
            # Focused env is used when opening EnvDetailScreen
            # No longer need to refresh per-env widgets since TrainingHealth shows aggregates

    def action_refresh(self) -> None:
        """Manually trigger refresh."""
        self._poll_and_refresh()

    def action_toggle_help(self) -> None:
        """Toggle help display."""
        self.push_screen(HelpScreen())

    def action_show_run_info(self) -> None:
        """Show full run information modal (untruncated task name, etc.)."""
        if self._snapshot is None:
            return
        self.push_screen(RunInfoScreen(self._snapshot))

    def action_cursor_down(self) -> None:
        """Move cursor down in EnvOverview table (vim: j)."""
        try:
            overview = self.query_one("#env-overview", EnvOverview)
            overview.table.action_cursor_down()
        except NoMatches:
            pass

    def action_cursor_up(self) -> None:
        """Move cursor up in EnvOverview table (vim: k)."""
        try:
            overview = self.query_one("#env-overview", EnvOverview)
            overview.table.action_cursor_up()
        except NoMatches:
            pass

    def action_cursor_top(self) -> None:
        """Move cursor to top of EnvOverview table (vim: gg)."""
        try:
            overview = self.query_one("#env-overview", EnvOverview)
            overview.table.move_cursor(row=0)
        except NoMatches:
            pass

    def action_cursor_bottom(self) -> None:
        """Move cursor to bottom of EnvOverview table (vim: G)."""
        try:
            overview = self.query_one("#env-overview", EnvOverview)
            overview.table.move_cursor(row=overview.table.row_count - 1)
        except NoMatches:
            pass

    def action_start_filter(self) -> None:
        """Show filter input and focus it (triggered by '/')."""
        try:
            filter_input = self.query_one("#filter-input", Input)
            filter_input.remove_class("hidden")
            filter_input.focus()
            self._filter_active = True
        except NoMatches:
            pass

    def action_clear_filter(self) -> None:
        """Clear and hide filter input (triggered by ESC)."""
        if not self._filter_active:
            return  # Don't consume ESC if filter not active

        try:
            filter_input = self.query_one("#filter-input", Input)
            filter_input.value = ""
            filter_input.add_class("hidden")
            self._filter_active = False

            # Clear filter in EnvOverview
            overview = self.query_one("#env-overview", EnvOverview)
            overview.set_filter("")

            # Return focus to the table
            overview.table.focus()
        except NoMatches:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes - update EnvOverview filter."""
        if event.input.id != "filter-input":
            return

        try:
            overview = self.query_one("#env-overview", EnvOverview)
            overview.set_filter(event.value)
        except NoMatches:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter on filter input - hide input, keep filter active."""
        if event.input.id != "filter-input":
            return

        try:
            filter_input = self.query_one("#filter-input", Input)
            filter_input.add_class("hidden")
            self._filter_active = False

            # Focus back on table
            overview = self.query_one("#env-overview", EnvOverview)
            overview.table.focus()
        except NoMatches:
            pass

    def action_focus_left_panel(self) -> None:
        """Focus the left panel (EnvOverview table)."""
        try:
            overview = self.query_one("#env-overview", EnvOverview)
            overview.table.focus()
        except NoMatches:
            pass

    def action_focus_right_panel(self) -> None:
        """Focus the right panel (Scoreboard table)."""
        try:
            scoreboard = self.query_one("#scoreboard", Scoreboard)
            scoreboard.table.focus()
        except NoMatches:
            pass

    def action_show_env_detail(self) -> None:
        """Show detailed view of focused environment.

        Opens a full-screen modal with comprehensive seed and environment metrics.
        """
        if self._snapshot is None:
            return

        env = self._snapshot.envs.get(self._focused_env_id)
        if env is None:
            return

        self.push_screen(
            EnvDetailScreen(
                env_state=env,
                slot_ids=self._snapshot.slot_ids,
            )
        )

    async def action_quit(self) -> None:
        """Handle quit with graceful shutdown.

        Signals the training thread to stop at the end of the current batch,
        then waits for it to finish before exiting the TUI.
        """
        if self._shutdown_requested:
            # Already shutting down, force quit
            self.exit()
            return

        self._shutdown_requested = True

        # Signal training to stop gracefully
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # Update title to show shutdown in progress
        self.sub_title = "Shutting down... (waiting for batch to complete)"

        # If training thread is alive, let the main script handle the wait
        # Just exit the TUI - the main script will wait for the thread
        if self._training_thread is not None and self._training_thread.is_alive():
            self.notify("Signaling graceful shutdown...", severity="warning")
            # Small delay so user sees the notification
            self.set_timer(0.5, lambda: self.exit())
        else:
            # Training already done, exit immediately
            self.exit()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle Enter key on DataTable row to show detail modal.

        Handles clicks on:
        - EnvOverview table → EnvDetailScreen (live env)
        - Scoreboard best runs table → HistoricalEnvDetail (frozen snapshot)
        - Scoreboard bottom 3 table → HistoricalEnvDetail (frozen snapshot)

        Args:
            event: The row selection event from DataTable.
        """
        if self._snapshot is None:
            return

        # Get row key
        row_key = event.row_key
        if row_key is None:
            return

        row_key_str = str(row_key.value) if row_key.value is not None else ""

        # Check if this is a scoreboard click (best runs or bottom 3)
        if row_key_str.startswith("bottom_"):
            # Bottom 3 table - extract record_id
            record_id = row_key_str.replace("bottom_", "")
            self._show_historical_env_detail(record_id)
            return
        elif not row_key_str.isdigit():
            # Best runs table - record_id is the key directly
            self._show_historical_env_detail(row_key_str)
            return

        # Otherwise, it's an env_id from EnvOverview
        try:
            env_id = int(row_key_str)
        except (ValueError, TypeError):
            return

        env = self._snapshot.envs.get(env_id)
        if env is None:
            return

        # Update focused env to match selection
        self._focused_env_id = env_id

        self.push_screen(
            EnvDetailScreen(
                env_state=env,
                slot_ids=self._snapshot.slot_ids,
            )
        )

    def _show_historical_env_detail(self, record_id: str) -> None:
        """Show historical env detail for a best run record.

        Args:
            record_id: The record_id of the BestRunRecord to display.
        """
        if self._snapshot is None:
            return

        # Find the record in best_runs
        record = None
        for r in self._snapshot.best_runs:
            if r.record_id == record_id:
                record = r
                break

        if record is None:
            return

        self.push_screen(HistoricalEnvDetail(record=record))

    def on_tamiyo_brain_decision_pin_toggled(
        self, event: TamiyoBrain.DecisionPinToggled
    ) -> None:
        """Handle click on decision panel to toggle pin status.

        Args:
            event: The pin toggle event with decision_id.
        """
        if self._backend is None:
            return

        # Toggle pin in aggregator
        try:
            new_status = self._backend.toggle_decision_pin(event.group_id, event.decision_id)
        except SanctumTelemetryFatalError as e:
            self._show_telemetry_fatal(e)
            return
        self.log.info(f"Decision {event.decision_id} pin toggled: {new_status}")

        # Refresh to show updated pin status
        self._poll_and_refresh()

    def on_scoreboard_best_run_selected(
        self, event: Scoreboard.BestRunSelected
    ) -> None:
        """Handle left-click on Best Runs row to show historical detail.

        Opens a modal showing the frozen env snapshot from when the
        run achieved its peak accuracy.

        Args:
            event: The selection event with the BestRunRecord.
        """
        self.push_screen(HistoricalEnvDetail(record=event.record))
        self.log.info(
            f"Opened historical detail for Ep {event.record.episode + 1} "
            f"(peak: {event.record.peak_accuracy:.1f}%)"
        )

    def on_scoreboard_best_run_pin_toggled(
        self, event: Scoreboard.BestRunPinToggled
    ) -> None:
        """Handle right-click on Best Runs row to toggle pin status.

        Pinned records are never removed from the leaderboard.

        Args:
            event: The pin toggle event with record_id.
        """
        if self._backend is None:
            return

        # Toggle pin in aggregator
        try:
            new_status = self._backend.toggle_best_run_pin(event.record_id)
        except SanctumTelemetryFatalError as e:
            self._show_telemetry_fatal(e)
            return
        self.log.info(f"Best run {event.record_id} pin toggled: {new_status}")

        # Refresh to show updated pin status
        self._poll_and_refresh()

    def on_event_log_detail_requested(
        self, event: EventLog.DetailRequested
    ) -> None:
        """Handle click on EventLog to show raw event detail modal.

        Args:
            event: The detail request with event list.
        """
        self.push_screen(EventLogDetail(events=event.events))
        self.log.info(f"Opened EventLogDetail with {len(event.events)} events")
