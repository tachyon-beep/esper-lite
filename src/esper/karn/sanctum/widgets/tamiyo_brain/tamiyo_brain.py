"""TamiyoBrain - Main container widget.

Policy agent dashboard with CSS-driven layout and
composable sub-widgets for better maintainability.

=== COLOR HIERARCHY (3 concerns only) ===
1. IDENTITY - Container border shows A/B/C group (green/cyan/magenta)
   - Applied via .group-a, .group-b, .group-c CSS classes
2. HEALTH - Status overrides identity when unhealthy (yellow=warning, red=critical)
   - Applied via .status-ok, .status-warning, .status-critical CSS classes
   - CSS cascade: status classes come AFTER group classes to override
3. EMPHASIS - Individual metric values highlight only when problematic
   - ok metrics: dim/muted (reduce visual noise)
   - warning metrics: yellow
   - critical metrics: red bold

Layout (CSS dimensions):
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ StatusBanner (h:1)                                                           │
    ├───────────────────────────────────────────────────────────────────┬──────────┤
    │ VitalsColumn (w:3fr)                                              │Decisions │
    │ ┌─────────┬───────────────────────────────────┬───────────────┐   │(w:1fr,   │
    │ │PPOLosses│ HealthStatus (1fr)                │ Slots (49ch)  │h13│ min:45)  │
    │ │ (36ch)  │                                   │               │   │          │
    │ ├────────────────────────────────────────────┬────────────────┤   │          │
    │ │ ActionHeadsPanel (69%)                     │                │   │          │
    │ │                                            │ ActionContext  │1fr│          │
    │ ├──────────────────────┬─────────────────────┤ (31%)          │   │          │
    │ │ EpisodeMetrics (1fr) │ ValueDiagnostics    │                │ h5│          │
    │ └──────────────────────┴─────────────────────┴────────────────┘   │          │
    └───────────────────────────────────────────────────────────────────┴──────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll

from .status_banner import StatusBanner
from .ppo_losses_panel import PPOLossesPanel
from .health_status_panel import HealthStatusPanel
from .action_heads_panel import ActionHeadsPanel
from .action_distribution import ActionContext
from .slots_panel import SlotsPanel
from .episode_metrics_panel import EpisodeMetricsPanel
from .value_diagnostics_panel import ValueDiagnosticsPanel
from .decisions_column import (
    DecisionDetailRequested,
    DecisionsColumn,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot
    from esper.karn.sanctum.widgets.reward_health import RewardHealthData


class TamiyoBrain(Container):
    """Tamiyo policy agent dashboard.

    Policy agent diagnostics with composable architecture.
    """

    DEFAULT_CSS = """
    TamiyoBrain {
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
        width: 3fr;
        height: 100%;
        padding: 0 1;
    }

    #decisions-column {
        width: 1fr;
        min-width: 45;
        height: 100%;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
        align: left top;
    }

    /* Row 1: PPO Losses (narrow) | Health (wide) | Slots */
    #top-row {
        height: 12;
        width: 100%;
        margin: 0;
    }

    #ppo-losses-panel {
        width: 47;  /* Fixed narrow width for content */
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #health-panel {
        width: 1fr;  /* Takes remaining space */
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #slots-panel {
        width: 50;  /* Fixed width for slot grid */
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    /* Bottom section: Left column (ActionHeads + Episode) | Right column (ActionContext) */
    #bottom-row {
        height: 1fr;
        width: 100%;
        margin: 0;
    }

    #left-column {
        width: 1fr;  /* Takes remaining space after action-context */
        height: 100%;
    }

    #action-heads-panel {
        width: 100%;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    /* Bottom metrics row: Episode Health (left) | Value Diagnostics (right) */
    #bottom-metrics-row {
        width: 100%;
        height: 9;
    }

    #episode-metrics-panel {
        width: 1fr;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #value-diagnostics-panel {
        width: 1fr;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #action-context {
        width: 50;  /* Same fixed width as slots-panel for vertical alignment */
        height: 100%;
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

    /* Remove padding from decision column internals */
    #cards-container {
        padding: 0;
        margin: 0;
        height: auto;
        border: none;
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
                # Top row - three panels: PPO (narrow) | Health (wide) | Slots
                with Horizontal(id="top-row"):
                    yield PPOLossesPanel(id="ppo-losses-panel")
                    yield HealthStatusPanel(id="health-panel")
                    yield SlotsPanel(id="slots-panel")
                # Bottom section: Left column (ActionHeads + metrics row) | Right column (ActionContext)
                with Horizontal(id="bottom-row"):
                    with Vertical(id="left-column"):
                        yield ActionHeadsPanel(id="action-heads-panel")
                        # Bottom metrics: Episode Health | Value Diagnostics
                        with Horizontal(id="bottom-metrics-row"):
                            yield EpisodeMetricsPanel(id="episode-metrics-panel")
                            yield ValueDiagnosticsPanel(id="value-diagnostics-panel")
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
        # Note: If called before widgets are mounted, this will raise NoMatches.
        # That's a bug in calling code - fix the caller, not the symptom.
        self.query_one("#status-banner", StatusBanner).update_snapshot(snapshot)
        self.query_one("#ppo-losses-panel", PPOLossesPanel).update_snapshot(snapshot)
        self.query_one("#health-panel", HealthStatusPanel).update_snapshot(snapshot)
        self.query_one("#action-heads-panel", ActionHeadsPanel).update_snapshot(
            snapshot
        )
        self.query_one("#action-context", ActionContext).update_snapshot(snapshot)
        self.query_one("#slots-panel", SlotsPanel).update_snapshot(snapshot)
        self.query_one("#episode-metrics-panel", EpisodeMetricsPanel).update_snapshot(
            snapshot
        )
        self.query_one(
            "#value-diagnostics-panel", ValueDiagnosticsPanel
        ).update_snapshot(snapshot)
        self.query_one("#decisions-column", DecisionsColumn).update_snapshot(snapshot)

    def update_reward_health(self, data: "RewardHealthData") -> None:
        """Update ActionContext with reward health data.

        Called by SanctumApp to pass reward health metrics to ActionContext.
        """
        self.query_one("#action-context", ActionContext).update_reward_health(data)

    def on_decision_detail_requested(self, event: DecisionDetailRequested) -> None:
        """Open drill-down screen for a decision."""
        from .decision_detail_screen import DecisionDetailScreen

        self.app.push_screen(
            DecisionDetailScreen(decision=event.decision, group_id=event.group_id)
        )

    @property
    def snapshot(self) -> "SanctumSnapshot | None":
        """Access current snapshot for testing."""
        return self._snapshot
