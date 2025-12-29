"""TamiyoBrainV2 - Main container widget.

Redesigned policy agent dashboard with CSS-driven layout and
composable sub-widgets for better maintainability.

Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ StatusBanner (1 line)                                           │
    ├─────────────────────────────────────────┬───────────────────────┤
    │ VitalsColumn (2/3 width)                │ DecisionsColumn (1/3) │
    │ ├── PrimaryMetrics (sparklines)         │ ├── DecisionCard      │
    │ ├── PPOHealthPanel (gauges + metrics)   │ ├── DecisionCard      │
    │ ├── HeadsPanel (entropy + gradients)    │ └── DecisionCard      │
    │ └── ActionContext (bar + slots)         │                       │
    └─────────────────────────────────────────┴───────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Static

from .status_banner import StatusBanner
from .primary_metrics import PrimaryMetrics
from .ppo_health import PPOHealthPanel
from .heads_grid import HeadsPanel
from .action_context import ActionContext
from .decisions_column import DecisionCard, DecisionsColumn

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class TamiyoBrainV2(Container):
    """Redesigned Tamiyo policy agent dashboard.

    Drop-in replacement for TamiyoBrain with improved visual design
    and composable architecture.
    """

    class DecisionPinToggled(Message):
        """Posted when user clicks a decision to toggle pin status.

        Bubbles up to the app for backend persistence.
        """

        def __init__(self, decision_id: str) -> None:
            super().__init__()
            self.decision_id = decision_id

    DEFAULT_CSS = """
    TamiyoBrainV2 {
        layout: vertical;
        height: 100%;
        border: round $surface-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
    }

    #status-banner {
        height: 1;
        width: 100%;
        padding: 0 1;
        background: $surface;
    }

    #main-content {
        height: 1fr;
        width: 100%;
    }

    #vitals-column {
        width: 2fr;
        height: 100%;
        padding: 0 1;
    }

    #decisions-column {
        width: 1fr;
        min-width: 50;
        height: 100%;
        border-left: solid $surface-lighten-1;
        padding: 0 1;
    }

    .panel {
        border: round $surface-lighten-2;
        margin: 0 0 1 0;
        padding: 0 1;
        height: auto;
    }

    /* Sub-panels - no padding or margin between them */
    #primary-metrics {
        margin: 0;
    }

    #ppo-health {
        height: 6;
        max-height: 6;
        padding: 0;
        margin: 0;
    }

    #heads-panel, #action-context {
        padding: 0;
        margin: 0;
    }

    #ppo-content {
        height: auto;
    }

    #gauge-column, #metrics-column {
        height: auto;
    }

    .panel-header {
        height: 1;
    }

    .panel-title {
        text-style: bold;
        color: $text-muted;
        margin-bottom: 0;
    }

    .decisions-header {
        height: 1;
        text-style: bold;
        color: $text-muted;
        margin-bottom: 0;
    }

    DecisionCard {
        height: auto;
        border: round $surface-lighten-2;
        padding: 0 1;
        margin-bottom: 0;
    }

    DecisionCard.newest {
        border: round $primary;
    }

    DecisionCard.oldest {
        border: round $warning-darken-1;
    }

    DecisionCard.pinned {
        border: double $success;
    }

    DecisionCard:focus {
        border: thick $accent;
        background: $panel-darken-1;
    }

    DecisionCard.pinned:focus {
        border: thick $success;
        background: $panel-darken-1;
    }
    """

    # Enable keyboard focus for Tab navigation
    can_focus = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.border_title = "TAMIYO"

    def compose(self) -> ComposeResult:
        """Compose the widget tree."""
        # Status banner spans full width
        yield StatusBanner(id="status-banner")

        # Main content: vitals left, decisions right
        with Horizontal(id="main-content"):
            with VerticalScroll(id="vitals-column"):
                yield PrimaryMetrics(id="primary-metrics")
                yield PPOHealthPanel(id="ppo-health")
                yield HeadsPanel(id="heads-panel")
                yield ActionContext(id="action-context")

            yield DecisionsColumn(id="decisions-column")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update all child widgets with new snapshot data.

        This is the main interface - same as TamiyoBrain for drop-in compatibility.
        """
        self._snapshot = snapshot

        # Update border title for A/B testing visibility
        if snapshot.tamiyo.group_id:
            self.border_title = f"TAMIYO [{snapshot.tamiyo.group_id}]"
        else:
            self.border_title = "TAMIYO"

        # Propagate snapshot to all child widgets
        # Use try-except to handle case where widgets haven't mounted yet
        try:
            self.query_one("#status-banner", StatusBanner).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#primary-metrics", PrimaryMetrics).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#ppo-health", PPOHealthPanel).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#heads-panel", HeadsPanel).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#action-context", ActionContext).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#decisions-column", DecisionsColumn).update_snapshot(snapshot)
        except NoMatches:
            pass

    @property
    def snapshot(self) -> "SanctumSnapshot | None":
        """Access current snapshot for testing."""
        return self._snapshot

    def on_decision_card_pinned(self, event: DecisionCard.Pinned) -> None:
        """Handle pin toggle from decision card and bubble up to app."""
        self.post_message(self.DecisionPinToggled(event.decision_id))
