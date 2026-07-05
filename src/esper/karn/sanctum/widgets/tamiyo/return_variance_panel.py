"""ReturnVariancePanel - EV-stab Stage-0 value-free return-variance shares.

Renders the RVT-leg decomposition of Var[return] by reward stream:
- CF share  = Cov(R_cf, R)/Var(R) — the Stage-0 gate reading. Crossing
  RETURN_VAR_CF_SHARE_GATE is a DIAGNOSIS (cf stream dominates; de-shaping
  justified), so it renders with a gate marker in identity colors, NOT the
  green/red health semantics of the neighbouring panels.
- Main share = Var(R_main)/Var(R) — the smoothness leg.
- Residual   = completeness keystone, ~0 when every additend is tracked;
  a breach IS a health fact (✗) and also alarms on the AnomalyStrip.

On flags-off runs the panel says WHY it is empty (the config flag name)
instead of rendering dead dashes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from esper.leyline import RETURN_VAR_CF_SHARE_GATE, RETURN_VAR_RESIDUAL_ALARM_SHARE

from .sparkline_utils import render_sparkline

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class ReturnVariancePanel(Static):
    """Value-free Stage-0 return-variance share panel (RVT leg)."""

    LABEL_W = 11
    SPARKLINE_WIDTH = 8

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "RETURN VARIANCE (S0)"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        if not tamiyo.rvt_leg_active:
            return Text(
                "stage-0 telemetry off (return_variance_telemetry)", style="dim"
            )

        result = Text()

        # Line 1: CF share + gate marker + sparkline
        result.append("CF share".ljust(self.LABEL_W), style="dim")
        cf_share = tamiyo.return_var_cf_share
        if cf_share is None:
            result.append("---", style="dim")
        else:
            result.append(f"{cf_share:.2f} ", style="bold magenta")
            if cf_share > RETURN_VAR_CF_SHARE_GATE:
                result.append(f"▲>{RETURN_VAR_CF_SHARE_GATE:.2f}", style="magenta")
            else:
                result.append(f"(gate {RETURN_VAR_CF_SHARE_GATE:.2f})", style="dim")
        history = list(tamiyo.return_var_cf_share_history)
        if len(history) >= 2:
            result.append("  ", style="dim")
            result.append_text(
                render_sparkline(history, width=self.SPARKLINE_WIDTH, style="magenta")
            )
        result.append("\n")

        # Line 2: Main (smoothness) share
        result.append("Main share".ljust(self.LABEL_W), style="dim")
        main_share = tamiyo.return_var_main_share
        if main_share is None:
            result.append("---", style="dim")
        else:
            result.append(f"{main_share:.2f}", style="cyan")
        result.append("\n")

        # Line 3: Residual keystone (the one HEALTH verdict on this panel)
        result.append("Residual".ljust(self.LABEL_W), style="dim")
        residual = tamiyo.return_var_residual_share
        if residual is None:
            result.append("---", style="dim")
        elif abs(residual) > RETURN_VAR_RESIDUAL_ALARM_SHARE:
            result.append(f"{residual:+.2f} ✗ decomp leak", style="red bold")
        else:
            result.append(f"{residual:+.2f} ●", style="green")

        return result
