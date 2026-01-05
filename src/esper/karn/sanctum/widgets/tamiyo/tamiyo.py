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
    ├───────────────────────────────────────────────────────────────────┬──────────┤
    │ VitalsColumn (w:4fr)                                              │ Decisions│
    │ ┌─────────┬────────────┬───────────────┐                         │ (1fr)   │
    │ │Narrative│ PPOLosses  │ Slots (52ch)  │ h13                     │          │
    │ ├──────────────────────┬──────────┬──────────────┐              │          │
    │ │ ActionHeadsPanel     │ Health   │ ActionContext │             │          │
    │ │ ┌────────┬─────────┐ │ (54ch)   │ (52ch)        │              │          │
    │ │ │Episode │ Value   │ │          │               │              │          │
    │ │ └────────┴─────────┘ │          │               │              │          │
    │ └──────────────────────┴──────────┴──────────────┘              │ EventLog │
    └───────────────────────────────────────────────────────────────────┴──────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll

from esper.karn.sanctum.widgets.event_log import EventLog

from .narrative_panel import NarrativePanel
from .ppo_losses_panel import PPOLossesPanel
from .health_status_panel import HealthStatusPanel
from .action_heads_panel import ActionHeadsPanel
from .action_distribution import ActionContext
from .critic_calibration_panel import CriticCalibrationPanel
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

    #main-content {
        height: 1fr;
        width: 100%;
    }

    #vitals-column {
        width: 4fr;
        height: 100%;
        padding: 0 1;
    }

    #right-column {
        width: 1fr;
        min-width: 36;
        height: 100%;
        padding: 0;
    }

    EventLog {
        width: 100%;
        height: 12;
        min-height: 12;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-size: 0 0;
        padding: 0 1;
    }

    #narrative-panel {
        width: 1fr;
        min-width: 38;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #decisions-panel {
        width: 100%;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    /* Row 1: Narrative | PPO | Slots */
    #top-row {
        height: 14;
        width: 100%;
        margin: 0;
    }

    #ppo-losses-panel {
        width: 56;  /* Wider for PPO diagnostics */
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #health-column {
        width: 56;  /* Fixed width for Health + Critic Calibration */
        height: 100%;
    }

    #health-panel {
        width: 100%;
        min-width: 0;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #slots-panel {
        width: 52;  /* Slightly narrower than PPO/Health */
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    /* Bottom section: Left column (ActionHeads + metrics) | Health | ActionContext */
    #bottom-row {
        height: 1fr;
        width: 100%;
        margin: 0;
    }

    #left-column {
        width: 1fr;  /* Expand to fill remaining space */
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

    #critic-calibration-panel {
        width: 100%;
        height: 9;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #action-context {
        width: 52;  /* Match slots width */
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
        height: 1fr;
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
        # Main content: vitals left, narrative/events right
        with Horizontal(id="main-content"):
            with VerticalScroll(id="vitals-column"):
                # Top row - three panels: Narrative | PPO | Slots
                with Horizontal(id="top-row"):
                    yield NarrativePanel(id="narrative-panel")
                    yield PPOLossesPanel(id="ppo-losses-panel")
                    yield SlotsPanel(id="slots-panel")
                # Bottom section: Left column (ActionHeads + metrics) | Health | ActionContext
                with Horizontal(id="bottom-row"):
                    with Vertical(id="left-column"):
                        yield ActionHeadsPanel(id="action-heads-panel")
                        # Bottom metrics: Episode Health | Value Diagnostics
                        with Horizontal(id="bottom-metrics-row"):
                            yield EpisodeMetricsPanel(id="episode-metrics-panel")
                            yield ValueDiagnosticsPanel(id="value-diagnostics-panel")
                    with Vertical(id="health-column"):
                        yield HealthStatusPanel(id="health-panel")
                        yield CriticCalibrationPanel(id="critic-calibration-panel")
                    yield ActionContext(id="action-context")

            with Vertical(id="right-column"):
                yield DecisionsColumn(id="decisions-panel")
                yield EventLog()

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update all child widgets with new snapshot data.

        This is the main interface - same as TamiyoBrain for drop-in compatibility.
        """
        self._snapshot = snapshot

        # Narrative first (used to compute status class)
        narrative = self.query_one("#narrative-panel", NarrativePanel)
        narrative.update_snapshot(snapshot)

        # Update A/B identity + health coloring (used by styles.tcss)
        self.remove_class("group-a", "group-b", "group-c")
        if snapshot.tamiyo.group_id:
            self.add_class(f"group-{snapshot.tamiyo.group_id.lower()}")

        self.remove_class(
            "status-ok", "status-warning", "status-critical", "status-warmup"
        )
        status, _, _ = narrative._get_overall_status()
        self.add_class(f"status-{status}")

        self.border_title = (
            f"TAMIYO ─ {snapshot.tamiyo.group_id}"
            if snapshot.tamiyo.group_id
            else "TAMIYO"
        )

        # Propagate snapshot to all child widgets
        # Note: If called before widgets are mounted, this will raise NoMatches.
        # That's a bug in calling code - fix the caller, not the symptom.
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
        self.query_one(
            "#critic-calibration-panel", CriticCalibrationPanel
        ).update_snapshot(snapshot)
        self.query_one("#decisions-panel", DecisionsColumn).update_snapshot(snapshot)
        self.query_one(EventLog).update_snapshot(snapshot)

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
