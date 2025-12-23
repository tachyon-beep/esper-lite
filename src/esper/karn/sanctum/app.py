"""Sanctum Textual Application.

Developer diagnostic TUI for debugging PPO training runs.
Layout matches existing Rich TUI (tui.py _render() method).

LAYOUT FIX: TamiyoBrain spans full width as dedicated row (size=11),
NOT embedded in right column. Event Log included at bottom-left.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Input, Static

from esper.karn.sanctum.widgets import (
    AnomalyStrip,
    EnvDetailScreen,
    EnvOverview,
    EventLog,
    HistoricalEnvDetail,
    RunHeader,
    Scoreboard,
    TamiyoBrain,
    ThreadDeathModal,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.schema import SanctumSnapshot


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
  [cyan]p[/cyan]         Toggle pin on Best Runs item
  [cyan]r[/cyan]         Manual refresh
  [cyan]q[/cyan]         Quit Sanctum

[bold]In Detail Modal[/bold]
  [cyan]Esc[/cyan]       Close modal
  [cyan]q[/cyan]         Close modal

[bold]Status Icons[/bold]
  [green]●[/green] OK     [yellow]◐[/yellow] Warning    [red]○[/red] Error
  [green]★[/green] Excellent   [green]✓[/green] Improving
  [yellow]⚠[/yellow] Stalling    [red]✗[/red] Severely stalled

[dim]Press Esc or ? to close this help[/dim]
"""


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


class SanctumApp(App):
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
    ):
        """Initialize Sanctum app.

        Args:
            backend: SanctumBackend providing snapshot data.
            num_envs: Number of training environments.
            refresh_rate: Snapshot refresh rate in Hz.
            training_thread: Optional training thread to monitor.
        """
        super().__init__()
        self._backend = backend
        self._num_envs = num_envs
        self._refresh_interval = 1.0 / refresh_rate
        self._focused_env_id: int = 0
        self._snapshot: "SanctumSnapshot" | None = None
        self._lock = threading.Lock()
        self._poll_count = 0  # Debug: track timer fires
        self._training_thread = training_thread  # Monitor thread status
        self._filter_active = False  # Track filter input visibility
        self._thread_death_shown = False  # Track if we've shown death modal

    def compose(self) -> ComposeResult:
        """Build the Sanctum layout.

        Layout structure:
        - Run Header: Episode, Epoch, Batch, Runtime, Best Accuracy, Connection
        - Anomaly Strip: Single-line automatic problem surfacing
        - Top row: EnvOverview (65%) | Scoreboard (35%)
        - Bottom row: TamiyoBrain (50%) | EventLog (50%)
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
                yield Scoreboard(id="scoreboard")

            # Bottom section: TamiyoBrain (left) | Event Log (right)
            with Horizontal(id="bottom-section"):
                # Left side: TamiyoBrain (50%)
                yield TamiyoBrain(id="tamiyo-brain")
                # Right side: Event Log (50%)
                yield EventLog(id="event-log")

        yield Footer()

    def on_mount(self) -> None:
        """Start refresh timer when app mounts."""
        self.set_interval(self._refresh_interval, self._poll_and_refresh)

    def _poll_and_refresh(self) -> None:
        """Poll backend for new snapshot and refresh all panels.

        Called periodically by set_interval timer.
        Thread-safe: backend.get_snapshot() is thread-safe.
        """
        self._poll_count += 1

        if self._backend is None:
            self.log.warning("Backend is None, skipping refresh")
            return

        # Get snapshot from backend (thread-safe)
        snapshot = self._backend.get_snapshot()

        # Debug: Add poll count to snapshot for display
        snapshot.poll_count = self._poll_count

        # Debug: Check if training thread is still alive
        thread_alive = self._training_thread.is_alive() if self._training_thread else None
        snapshot.training_thread_alive = thread_alive

        # Check if training thread died (and we haven't shown modal yet)
        if thread_alive is False and not self._thread_death_shown:
            self._thread_death_shown = True
            self.push_screen(ThreadDeathModal())
            self.log.error("Training thread died! Showing death modal.")

        # Debug: Log snapshot state (visible in Textual console with Ctrl+Shift+D)
        self.log.info(
            f"Poll #{self._poll_count}: connected={snapshot.connected}, "
            f"ep={snapshot.current_episode}, "
            f"events={len(snapshot.event_log)}, "
            f"total_events={snapshot.total_events_received}, "
            f"thread_alive={thread_alive}"
        )

        with self._lock:
            self._snapshot = snapshot

        # Update all widgets
        self._refresh_all_panels(snapshot)

    def _refresh_all_panels(self, snapshot: "SanctumSnapshot") -> None:
        """Refresh all panels with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        # Update run header first (most important context)
        try:
            self.query_one("#run-header", RunHeader).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update run-header: {e}")

        # Update anomaly strip (after run header)
        try:
            self.query_one("#anomaly-strip", AnomalyStrip).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update anomaly-strip: {e}")

        # Update each widget - query by ID and call update_snapshot
        try:
            self.query_one("#env-overview", EnvOverview).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update env-overview: {e}")

        try:
            self.query_one("#scoreboard", Scoreboard).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update scoreboard: {e}")

        try:
            self.query_one("#tamiyo-brain", TamiyoBrain).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update tamiyo-brain: {e}")

        try:
            self.query_one("#event-log", EventLog).update_snapshot(snapshot)
        except NoMatches:
            pass  # Widget hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update event-log: {e}")

        # Update EnvDetailScreen modal if displayed
        # Check if we have a modal screen on the stack
        if len(self.screen_stack) > 1:
            current_screen = self.screen_stack[-1]
            if isinstance(current_screen, EnvDetailScreen):
                # Get updated env state from snapshot
                env = snapshot.envs.get(current_screen.env_id)
                if env is not None:
                    try:
                        current_screen.update_env_state(env)
                    except Exception as e:
                        self.log.warning(f"Failed to update env-detail-screen: {e}")

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

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle Enter key on DataTable row to show detail modal.

        Args:
            event: The row selection event from DataTable.
        """
        if self._snapshot is None:
            return

        # Get env_id from the row key
        row_key = event.row_key
        if row_key is None:
            return

        # Extract env_id from row_key.value (set in EnvOverview._add_env_row)
        try:
            env_id = int(row_key.value) if row_key.value is not None else None
        except (ValueError, TypeError):
            return

        if env_id is None:
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
        new_status = self._backend.toggle_decision_pin(event.decision_id)
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
        new_status = self._backend.toggle_best_run_pin(event.record_id)
        self.log.info(f"Best run {event.record_id} pin toggled: {new_status}")

        # Refresh to show updated pin status
        self._poll_and_refresh()
