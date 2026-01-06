"""ValueDiagnosticsPanel - Return distribution + infrastructure metrics.

Per DRL expert review: Return distribution reveals whether the policy is
consistent (or bimodal) even when point metrics look stable.

Layout:
    ┌─ VALUE DIAGNOSTICS ────────────────────────┐
    │ Returns   p10:-12  p50:+34  p90:+78        │
    │ Ret σ     14.2     Skew     +0.3           │
    │ R Mean    +12.4    Trend     ↗             │
    └────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from .trends import trend_arrow_for_history

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class ValueDiagnosticsPanel(Static):
    """Return distribution diagnostics panel (PPO update stats)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "VALUE DIAGNOSTICS (PPO)"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    # Column layout: two sub-columns with label + value each
    # |  Label1  |  Val1  |  Label2  |  Val2  |
    LABEL_W = 10  # Label width
    VAL_W = 8     # Value width
    GAP = 2       # Gap between sub-columns

    def render(self) -> Text:
        """Render value function diagnostics.

        Layout (3 lines, two sub-columns):
            Returns    p10:-12   p50:+34   p90:+78 ⚠
            Ret σ      14.2      Skew        +0.3
            R Mean     +12.4     Trend       ↗
        """
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        vf = tamiyo.value_function
        history = list(tamiyo.episode_return_history)
        result = Text()

        # Line 1: Return Percentiles (three values, special layout)
        self._render_label(result, "Returns")
        p10_style = "red" if vf.return_p10 < 0 else "green"
        p50_style = "red" if vf.return_p50 < 0 else "green"
        p90_style = "red" if vf.return_p90 < 0 else "green"

        result.append("p10:", style="dim")
        result.append(f"{vf.return_p10:+.0f}".ljust(5), style=p10_style)
        result.append("p50:", style="dim")
        result.append(f"{vf.return_p50:+.0f}".ljust(5), style=p50_style)
        result.append("p90:", style="dim")
        result.append(f"{vf.return_p90:+.0f}", style=p90_style)

        # Show spread warning if bimodal (large p90-p10 gap)
        spread = vf.return_p90 - vf.return_p10
        if spread > 50:
            result.append(" ⚠", style="yellow bold")
        result.append("\n")

        # Line 2: Return Std Dev | Skewness
        self._render_label(result, "Ret σ")
        var_style = "yellow" if vf.return_variance > 100 else "cyan"
        self._render_value(result, f"{vf.return_variance ** 0.5:.1f}", var_style)

        self._render_label(result, "Skew")
        skew_style = self._get_skewness_style(vf.return_skewness)
        self._render_value(result, f"{vf.return_skewness:+.1f}", skew_style, last=True)
        result.append("\n")

        # Line 3: Return Mean | Trend
        self._render_label(result, "R Mean")
        if history:
            mean = sum(history) / len(history)
            mean_style = "green" if mean >= 0 else "red"
            self._render_value(result, f"{mean:+.1f}", mean_style)
        else:
            self._render_value(result, "---", "dim")

        self._render_label(result, "Trend")
        arrow, arrow_style = trend_arrow_for_history(
            history, metric_name="episode_return", metric_type="accuracy"
        )
        if arrow:
            self._render_value(result, arrow, arrow_style, last=True)
        else:
            self._render_value(result, "—", "dim", last=True)

        return result

    def _render_label(self, result: Text, label: str) -> None:
        """Render a column label with fixed width."""
        result.append(label.ljust(self.LABEL_W), style="dim")

    def _render_value(
        self, result: Text, value: str, style: str, *, last: bool = False
    ) -> None:
        """Render a column value with fixed width and gap."""
        result.append(value.ljust(self.VAL_W), style=style)
        if not last:
            result.append(" " * self.GAP)

    def _get_skewness_style(self, skew: float) -> str:
        """Get style for return skewness.

        Healthy: near 0 (symmetric distribution)
        Warning: |skew| > 1 (moderately asymmetric)
        Critical: |skew| > 2 (severely skewed - policy very inconsistent)
        """
        abs_skew = abs(skew)
        if abs_skew < 1.0:
            return "cyan"
        elif abs_skew < 2.0:
            return "yellow"
        else:
            return "red bold"

    def _get_correlation_style(self, correlation: float) -> tuple[str, str]:
        """Get style + icon for V-Return correlation (TELE-220)."""
        if correlation >= 0.8:
            return ("green bold", "↗")
        if correlation >= 0.5:
            return ("green", "→")
        if correlation >= 0.3:
            return ("yellow", "→")
        return ("red bold", "↘")

    def _get_td_error_style(self, td_error_mean: float, _td_error_std: float) -> str:
        """Get style for TD error mean (TELE-221)."""
        abs_mean = abs(td_error_mean)
        if abs_mean < 5.0:
            return "green"
        if abs_mean < 15.0:
            return "yellow"
        return "red bold"

    def _get_bellman_style(self, bellman_error: float) -> str:
        """Get style for Bellman error (TELE-223)."""
        if bellman_error < 20.0:
            return "green"
        if bellman_error < 50.0:
            return "yellow"
        return "red bold"
