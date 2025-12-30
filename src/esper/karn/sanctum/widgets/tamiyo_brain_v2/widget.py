"""TamiyoBrainV2 - Main container widget.

Redesigned policy agent dashboard with CSS-driven layout and
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

Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ StatusBanner (1 line)                                           │
    ├───────────────────────────────────────────────────┬─────────────┤
    │ VitalsColumn (75%)                                │ Decisions   │
    │ ├── PPOLosses (50%)  | Health (50%)               │ (25%)       │
    │ ├── HeadsPanel (68%) | Slots (32%)                │ ├── Card    │
    │ └── AttentionHeatmap (68%) | ActionContext (32%)  │ └── Card    │
    └───────────────────────────────────────────────────┴─────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll

from .status_banner import StatusBanner
from .ppo_losses_panel import PPOLossesPanel
from .health_status_panel import HealthStatusPanel
from .heads_grid import HeadsPanel
from .action_context import ActionContext
from .slots_panel import SlotsPanel
from .decisions_column import DecisionsColumn
from .attention_heatmap import AttentionHeatmapPanel

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class TamiyoBrainV2(Container):
    """Redesigned Tamiyo policy agent dashboard.

    Drop-in replacement for TamiyoBrain with improved visual design
    and composable architecture.
    """

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

    /* Row containers - explicit heights based on content + border */
    #ppo-row {
        height: 12;  /* Reduced from 13 to give space to bottom rows */
        width: 100%;
    }

    #heads-row {
        height: 12;  /* Reduced from 13 - removed padding lines */
        width: 100%;
    }

    #action-row {
        height: 12;  /* Increased from 11 - gained 1 row from ppo-row */
        width: 100%;
    }

    /* All panels - fill their row heights */
    #ppo-losses-panel, #health-panel {
        width: 1fr;
        height: 1fr;  /* Fill ppo-row height */
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #heads-panel {
        width: 68%;
        height: 1fr;  /* Fill heads-row height */
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #slots-panel {
        width: 32%;
        height: 1fr;  /* Fill heads-row height */
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #action-context {
        width: 32%;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    #attention-heatmap {
        width: 68%;
        height: 1fr;  /* Fill available height in action-row */
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

    DecisionCard.pinned {
        border: double $success;
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
                # PPO row - two panels (50/50 split)
                with Horizontal(id="ppo-row"):
                    yield PPOLossesPanel(id="ppo-losses-panel")
                    yield HealthStatusPanel(id="health-panel")
                # Heads row - HeadsPanel (60%) | SlotsPanel (40%)
                with Horizontal(id="heads-row"):
                    yield HeadsPanel(id="heads-panel")
                    yield SlotsPanel(id="slots-panel")
                # Action row - AttentionHeatmap | ActionContext
                with Horizontal(id="action-row"):
                    yield AttentionHeatmapPanel(id="attention-heatmap")
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
        self.query_one("#heads-panel", HeadsPanel).update_snapshot(snapshot)
        self.query_one("#action-context", ActionContext).update_snapshot(snapshot)
        self.query_one("#slots-panel", SlotsPanel).update_snapshot(snapshot)
        self.query_one("#attention-heatmap", AttentionHeatmapPanel).update_snapshot(snapshot)
        self.query_one("#decisions-column", DecisionsColumn).update_snapshot(snapshot)

    @property
    def snapshot(self) -> "SanctumSnapshot | None":
        """Access current snapshot for testing."""
        return self._snapshot
