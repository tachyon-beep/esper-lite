"""TamiyoBrainV2 - Main container widget.

Redesigned policy agent dashboard with CSS-driven layout and
composable sub-widgets for better maintainability.

Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ StatusBanner (1 line)                                           │
    ├─────────────────────────────────────────┬───────────────────────┤
    │ VitalsColumn (2/3 width)                │ DecisionsColumn (1/3) │
    │ ├── PPOHealth | Losses | Health         │ ├── DecisionCard      │
    │ ├── HeadsPanel (entropy + gradients)    │ ├── DecisionCard      │
    │ └── ActionContext | Slots | RewardHealth│ └── DecisionCard      │
    └─────────────────────────────────────────┴───────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message

from .status_banner import StatusBanner
from .ppo_health import PPOHealthPanel
from .losses_panel import LossesPanel
from .health_status_panel import HealthStatusPanel
from .heads_grid import HeadsPanel
from .action_context import ActionContext
from .slots_panel import SlotsPanel
from .decisions_column import DecisionCard, DecisionsColumn
from ..reward_health import RewardHealthPanel

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
        background: $surface-lighten-1;
    }

    #banner-content {
        width: 100%;
        padding: 0 1;
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
        margin: 0;
        padding: 0 1;
        height: auto;
    }

    /* Sub-panels - now Static-based, height sizes to content */
    #ppo-health, #losses-panel, #health-panel, #heads-panel, #action-context, #slots-panel {
        height: auto;
        padding: 0;
        margin: 0;
    }

    /* PPO row - three panels */
    #ppo-row {
        height: auto;
        width: 100%;
    }

    #ppo-row #ppo-health {
        width: 0.9fr;
    }

    #ppo-row #losses-panel {
        width: 0.9fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #ppo-row #health-panel {
        width: 1.2fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #reward-health {
        height: auto;
        padding: 0;
        margin: 0;
    }

    #action-row {
        height: auto;
        width: 100%;
    }

    /* Action row: ActionContext and Slots wider, RewardHealth half-width */
    #action-row #action-context {
        width: 1.25fr;
    }

    #action-row #slots-panel {
        width: 1.25fr;
        height: auto;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #action-row #reward-health {
        width: 0.5fr;
        height: auto;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
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

    /* Remove padding from decision column internals */
    #decisions-header, #cards-container {
        padding: 0;
        margin: 0;
    }
    """

    # Enable keyboard focus for Tab navigation
    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
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
                # PPO row - three panels
                with Horizontal(id="ppo-row"):
                    yield PPOHealthPanel(id="ppo-health")
                    yield LossesPanel(id="losses-panel")
                    yield HealthStatusPanel(id="health-panel")
                yield HeadsPanel(id="heads-panel")
                # ActionContext, Slots, and RewardHealth side by side
                with Horizontal(id="action-row"):
                    yield ActionContext(id="action-context")
                    yield SlotsPanel(id="slots-panel")
                    yield RewardHealthPanel(id="reward-health")

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
            self.query_one("#ppo-health", PPOHealthPanel).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#losses-panel", LossesPanel).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#health-panel", HealthStatusPanel).update_snapshot(snapshot)
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
            self.query_one("#slots-panel", SlotsPanel).update_snapshot(snapshot)
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
