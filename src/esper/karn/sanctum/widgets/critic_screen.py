"""CriticScreen - the consolidated critic/value tab (EV-stab surface).

Gathers the previously fragmented critic story into one full-width view:
calibration (EV fit, V-corr, TD/Bellman, per-stream EV on the HRA leg),
return distribution, the Stage-0 return-variance shares, and reward health.
The same panel classes also render inside TamiyoBrain's compact overview
columns; these are independent instances fed by the same snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal

from esper.karn.sanctum.widgets.reward_health import RewardHealthData, RewardHealthPanel
from esper.karn.sanctum.widgets.tamiyo.critic_calibration_panel import (
    CriticCalibrationPanel,
)
from esper.karn.sanctum.widgets.tamiyo.return_variance_panel import ReturnVariancePanel
from esper.karn.sanctum.widgets.tamiyo.value_diagnostics_panel import (
    ValueDiagnosticsPanel,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class CriticScreen(Container):
    """Full-width critic & value diagnostics tab."""

    DEFAULT_CSS = """
    CriticScreen {
        layout: vertical;
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    CriticScreen Horizontal {
        height: auto;
        width: 100%;
    }

    /* Child selector, NOT a class: the tamiyo panels overwrite self.classes
       with "panel" in __init__, so classes passed at construction are lost. */
    CriticScreen Horizontal > * {
        width: 1fr;
        min-width: 44;
        height: auto;
        min-height: 9;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield CriticCalibrationPanel(id="critic-tab-calibration")
            yield ValueDiagnosticsPanel(id="critic-tab-value-diagnostics")
        with Horizontal():
            yield ReturnVariancePanel(id="critic-tab-return-variance")
            yield RewardHealthPanel(id="critic-tab-reward-health")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Fan the snapshot out to all critic panels."""
        self._snapshot = snapshot
        self.query_one("#critic-tab-calibration", CriticCalibrationPanel).update_snapshot(
            snapshot
        )
        self.query_one(
            "#critic-tab-value-diagnostics", ValueDiagnosticsPanel
        ).update_snapshot(snapshot)
        self.query_one(
            "#critic-tab-return-variance", ReturnVariancePanel
        ).update_snapshot(snapshot)

    def update_reward_health(self, data: RewardHealthData) -> None:
        self.query_one("#critic-tab-reward-health", RewardHealthPanel).update_data(data)

    @property
    def snapshot(self) -> "SanctumSnapshot | None":
        """Latest snapshot routed to this tab (test seam)."""
        return self._snapshot
