"""ValueDiagnosticsPanel - Value function quality + infrastructure metrics.

Per DRL expert review: Value function quality is THE primary diagnostic
for RL training failures. This panel surfaces value-specific issues that
are often the root cause when policy metrics look fine but training fails.

Per PyTorch expert: Compile status should be visible (fallback to eager = 3-5x slower).

Layout:
    ┌─ VALUE DIAGNOSTICS ────────────────────────┐
    │ V-Ret Corr: 0.87↗  TD μ/σ: 2.4/8.1        │
    │ Bellman: 12.3      p10/p50/p90: -12/34/78  │
    │ Compile: inductor:reduce-overhead ✓        │
    └────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class ValueDiagnosticsPanel(Static):
    """Value function quality diagnostics panel.

    Surfaces metrics that indicate whether the value network is learning:
    - V-Return correlation: Can the value network predict returns?
    - TD error distribution: Is there bias or excessive noise?
    - Bellman error: Early warning for value collapse
    - Return percentiles: Catch bimodal policies
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "VALUE DIAGNOSTICS"

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

        Layout (5 lines, two sub-columns):
            V-Corr     0.87↗     Bellman      12.3
            TD Mean    +2.4      TD Std        8.1
            Returns    p10:-12   p50:+34   p90:+78 ⚠
            Ret σ      14.2      Skew        +0.3
            Compile    inductor:reduce-overhead ✓
        """
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        vf = self._snapshot.tamiyo.value_function
        result = Text()

        # Line 1: V-Return Correlation | Bellman Error
        self._render_label(result, "V-Corr")
        corr_style, corr_icon = self._get_correlation_style(vf.v_return_correlation)
        self._render_value(result, f"{vf.v_return_correlation:.2f}{corr_icon}", corr_style)

        self._render_label(result, "Bellman")
        bellman_style = self._get_bellman_style(vf.bellman_error)
        self._render_value(result, f"{vf.bellman_error:.1f}", bellman_style, last=True)
        result.append("\n")

        # Line 2: TD Mean | TD Std
        self._render_label(result, "TD Mean")
        td_style = self._get_td_error_style(vf.td_error_mean, vf.td_error_std)
        self._render_value(result, f"{vf.td_error_mean:+.1f}", td_style)

        self._render_label(result, "TD Std")
        self._render_value(result, f"{vf.td_error_std:.1f}", "cyan", last=True)
        result.append("\n")

        # Line 3: Return Percentiles (three values, special layout)
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

        # Line 4: Return Std Dev | Skewness
        self._render_label(result, "Ret σ")
        var_style = "yellow" if vf.return_variance > 100 else "cyan"
        self._render_value(result, f"{vf.return_variance ** 0.5:.1f}", var_style)

        self._render_label(result, "Skew")
        skew_style = self._get_skewness_style(vf.return_skewness)
        self._render_value(result, f"{vf.return_skewness:+.1f}", skew_style, last=True)
        result.append("\n")

        # Line 5: Compile status (full width)
        self._render_label(result, "Compile")
        infra = self._snapshot.tamiyo.infrastructure

        if infra.compile_enabled:
            backend = infra.compile_backend or "inductor"
            mode = infra.compile_mode or "default"
            result.append(f"{backend}:{mode}", style="green")
            result.append(" ✓", style="green bold")
        else:
            result.append("EAGER", style="red bold reverse")
            result.append(" (3-5x slower)", style="dim")

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

    def _get_correlation_style(self, corr: float) -> tuple[str, str]:
        """Get style and trend icon for V-Return correlation.

        Returns:
            Tuple of (style, trend_icon)
        """
        # Correlation quality thresholds
        if corr >= 0.8:
            return ("green bold", "↗")
        elif corr >= 0.5:
            return ("green", "→")
        elif corr >= 0.3:
            return ("yellow", "→")
        else:
            # Low correlation = value network not learning
            return ("red bold", "↘")

    def _get_td_error_style(self, mean: float, std: float) -> str:
        """Get style for TD error mean.

        High mean = biased value estimates (bad)
        The std is shown separately for context.
        """
        abs_mean = abs(mean)
        if abs_mean < 5:
            return "green"
        elif abs_mean < 15:
            return "yellow"
        else:
            return "red bold"

    def _get_bellman_style(self, bellman: float) -> str:
        """Get style for Bellman error.

        Bellman error spikes often precede NaN losses.
        """
        if bellman < 20:
            return "green"
        elif bellman < 50:
            return "yellow"
        else:
            return "red bold"

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
