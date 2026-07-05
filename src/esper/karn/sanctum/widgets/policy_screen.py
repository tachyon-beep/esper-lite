"""PolicyScreen - the actor-side deep-diagnostics tab.

Hosts the panels that were crammed into TamiyoBrain's bottom row: the full
per-head ACTION HEADS table finally gets the whole terminal width (entropy
bars, grad trends, ratio maxima, decision carousel, flow + advantage-norm
footers), with ACTION CONTEXT, the INNER LOOP episode metrics, and TORCH
STABILITY beneath it. TamiyoBrain on the Overview tab keeps only the digest
(narrative, PPO update, slots, health rollup).

Shows the primary policy group; 't' switches which leg drives it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal

from esper.karn.sanctum.widgets.tamiyo.action_distribution import ActionContext
from esper.karn.sanctum.widgets.tamiyo.action_heads_panel import ActionHeadsPanel
from esper.karn.sanctum.widgets.tamiyo.episode_metrics_panel import EpisodeMetricsPanel
from esper.karn.sanctum.widgets.tamiyo.torch_stability_panel import TorchStabilityPanel

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot
    from esper.karn.sanctum.widgets.reward_health import RewardHealthData


class PolicyScreen(Container):
    """Full-width actor diagnostics tab."""

    DEFAULT_CSS = """
    PolicyScreen {
        layout: vertical;
        height: 1fr;
        padding: 0 1;
    }

    PolicyScreen #policy-action-heads {
        width: 100%;
        height: 2fr;
        min-height: 18;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
        overflow-y: auto;
    }

    PolicyScreen #policy-bottom-row {
        width: 100%;
        height: 1fr;
        min-height: 10;
    }

    /* Child selector, NOT a class: the tamiyo panels overwrite self.classes
       with "panel" in __init__, so classes passed at construction are lost. */
    PolicyScreen #policy-bottom-row > * {
        width: 1fr;
        min-width: 40;
        height: 100%;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        yield ActionHeadsPanel(id="policy-action-heads")
        with Horizontal(id="policy-bottom-row"):
            yield ActionContext(id="policy-action-context")
            yield EpisodeMetricsPanel(id="policy-episode-metrics")
            yield TorchStabilityPanel(id="policy-torch-stability")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Fan the snapshot out to all actor panels."""
        self._snapshot = snapshot
        self.query_one("#policy-action-heads", ActionHeadsPanel).update_snapshot(snapshot)
        self.query_one("#policy-action-context", ActionContext).update_snapshot(snapshot)
        self.query_one("#policy-episode-metrics", EpisodeMetricsPanel).update_snapshot(
            snapshot
        )
        self.query_one("#policy-torch-stability", TorchStabilityPanel).update_snapshot(
            snapshot
        )

    def update_reward_health(self, data: "RewardHealthData") -> None:
        self.query_one("#policy-action-context", ActionContext).update_reward_health(data)

    @property
    def snapshot(self) -> "SanctumSnapshot | None":
        """Latest snapshot routed to this tab (test seam)."""
        return self._snapshot
