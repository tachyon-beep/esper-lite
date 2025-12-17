"""Flight Board Widget.

The main display surface showing all training environments.
Supports navigation, selection, and expansion of env rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static

from esper.karn.overwatch.display_state import DisplayState, HysteresisSorter
from esper.karn.overwatch.widgets.env_row import EnvRow

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import EnvSummary, TuiSnapshot


class FlightBoard(VerticalScroll):
    """Scrollable container displaying all training environments.

    Features:
    - Sorts envs by anomaly score with hysteresis
    - j/k or arrow navigation between envs
    - Enter to expand env details, Esc to collapse
    - Visual focus indicator on selected row

    Messages:
    - EnvSelected: Fired when selection changes
    - EnvExpanded: Fired when env is expanded/collapsed
    """

    DEFAULT_CSS = """
    FlightBoard {
        width: 100%;
        height: 100%;
        background: $surface;
    }

    FlightBoard > EnvRow {
        width: 100%;
    }

    FlightBoard:focus {
        border: tall $primary;
    }

    FlightBoard .empty-state {
        width: 100%;
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("j", "navigate_down", "Down", show=False),
        Binding("k", "navigate_up", "Up", show=False),
        Binding("down", "navigate_down", "Down", show=False),
        Binding("up", "navigate_up", "Up", show=False),
        Binding("enter", "toggle_expand", "Expand", show=False),
    ]

    class EnvSelected(Message):
        """Fired when env selection changes."""

        def __init__(self, env_id: int | None) -> None:
            super().__init__()
            self.env_id = env_id

    class EnvExpanded(Message):
        """Fired when env expansion state changes."""

        def __init__(self, env_id: int, expanded: bool) -> None:
            super().__init__()
            self.env_id = env_id
            self.expanded = expanded

    def __init__(self, **kwargs) -> None:
        """Initialize the flight board."""
        super().__init__(**kwargs)
        self._display_state = DisplayState()
        self._snapshot: TuiSnapshot | None = None
        self._envs_by_id: dict[int, EnvSummary] = {}
        self._display_order: list[int] = []

    @property
    def selected_env_id(self) -> int | None:
        """Currently selected environment ID."""
        return self._display_state.selected_env_id

    def get_display_order(self) -> list[int]:
        """Get current display order of env IDs."""
        return self._display_order.copy()

    def is_expanded(self, env_id: int) -> bool:
        """Check if env is expanded."""
        return self._display_state.is_expanded(env_id)

    def update_snapshot(self, snapshot: TuiSnapshot) -> None:
        """Update with new snapshot data.

        Args:
            snapshot: New TuiSnapshot to display
        """
        self._snapshot = snapshot

        # Build lookup
        self._envs_by_id = {e.env_id: e for e in snapshot.flight_board}

        # Get sorted order with hysteresis
        scores = {e.env_id: e.anomaly_score for e in snapshot.flight_board}
        self._display_order = self._display_state.get_sorted_env_ids(scores)

        # Auto-select first if nothing selected
        if self._display_state.selected_env_id is None and self._display_order:
            self._display_state.select_env(self._display_order[0])

        # Validate selection still exists
        if self._display_state.selected_env_id not in self._envs_by_id:
            if self._display_order:
                self._display_state.select_env(self._display_order[0])
            else:
                self._display_state.selected_env_id = None

        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Compose the flight board content."""
        if not self._display_order:
            yield Static("No environments", classes="empty-state")
            return

        for env_id in self._display_order:
            env = self._envs_by_id.get(env_id)
            if env is None:
                continue

            is_selected = env_id == self._display_state.selected_env_id
            is_expanded = self._display_state.is_expanded(env_id)

            yield EnvRow(
                env,
                selected=is_selected,
                expanded=is_expanded,
                id=f"env-{env_id}",
            )

    def navigate_down(self) -> None:
        """Move selection down one row."""
        if not self._display_order:
            return

        current = self._display_state.selected_env_id
        if current is None:
            # Select first
            self._select(self._display_order[0])
            return

        try:
            idx = self._display_order.index(current)
            if idx < len(self._display_order) - 1:
                self._select(self._display_order[idx + 1])
        except ValueError:
            # Current not in list, select first
            self._select(self._display_order[0])

    def navigate_up(self) -> None:
        """Move selection up one row."""
        if not self._display_order:
            return

        current = self._display_state.selected_env_id
        if current is None:
            # Select last
            self._select(self._display_order[-1])
            return

        try:
            idx = self._display_order.index(current)
            if idx > 0:
                self._select(self._display_order[idx - 1])
        except ValueError:
            # Current not in list, select first
            self._select(self._display_order[0])

    def toggle_expand(self) -> None:
        """Toggle expansion of selected env."""
        env_id = self._display_state.selected_env_id
        if env_id is None:
            return

        expanded = self._display_state.toggle_expand(env_id)
        self.post_message(self.EnvExpanded(env_id, expanded))
        self.refresh(recompose=True)

    def _select(self, env_id: int) -> None:
        """Select an env and update UI."""
        old_id = self._display_state.selected_env_id
        self._display_state.select_env(env_id)

        # Update old row
        if old_id is not None:
            try:
                old_row = self.query_one(f"#env-{old_id}", EnvRow)
                old_row.set_selected(False)
            except Exception:
                # Widget not in tree yet (unmounted or testing)
                pass

        # Update new row
        try:
            new_row = self.query_one(f"#env-{env_id}", EnvRow)
            new_row.set_selected(True)
            self.scroll_to_widget(new_row)
        except Exception:
            # Widget not in tree yet (unmounted or testing)
            pass

        self.post_message(self.EnvSelected(env_id))

    # Action handlers for bindings
    def action_navigate_down(self) -> None:
        """Action: navigate down."""
        self.navigate_down()

    def action_navigate_up(self) -> None:
        """Action: navigate up."""
        self.navigate_up()

    def action_toggle_expand(self) -> None:
        """Action: toggle expand."""
        self.toggle_expand()
