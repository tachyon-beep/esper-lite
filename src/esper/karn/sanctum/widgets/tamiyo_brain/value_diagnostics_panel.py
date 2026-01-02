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

    def render(self) -> Text:
        """Render value function diagnostics."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        vf = self._snapshot.tamiyo.value_function
        result = Text()

        # Line 1: V-Return Correlation + TD Error
        result.append("V-Ret Corr:", style="dim")
        corr_style, corr_icon = self._get_correlation_style(vf.v_return_correlation)
        result.append(f"{vf.v_return_correlation:.2f}", style=corr_style)
        result.append(corr_icon, style=corr_style)
        result.append("  ", style="dim")

        result.append("TD μ/σ:", style="dim")
        td_style = self._get_td_error_style(vf.td_error_mean, vf.td_error_std)
        result.append(f"{vf.td_error_mean:.1f}", style=td_style)
        result.append("/", style="dim")
        result.append(f"{vf.td_error_std:.1f}", style="cyan")
        result.append("\n")

        # Line 2: Bellman Error + Return Percentiles
        result.append("Bellman:", style="dim")
        bellman_style = self._get_bellman_style(vf.bellman_error)
        result.append(f"{vf.bellman_error:.1f}", style=bellman_style)
        result.append("  ", style="dim")

        result.append("p10/p50/p90:", style="dim")
        # Color code percentiles: negative = red, positive = green
        p10_style = "red" if vf.return_p10 < 0 else "green"
        p50_style = "red" if vf.return_p50 < 0 else "green"
        p90_style = "red" if vf.return_p90 < 0 else "green"

        result.append(f"{vf.return_p10:+.0f}", style=p10_style)
        result.append("/", style="dim")
        result.append(f"{vf.return_p50:+.0f}", style=p50_style)
        result.append("/", style="dim")
        result.append(f"{vf.return_p90:+.0f}", style=p90_style)

        # Show spread warning if bimodal (large p90-p10 gap)
        spread = vf.return_p90 - vf.return_p10
        if spread > 50:  # Large spread indicates inconsistent policy
            result.append(" ⚠", style="yellow bold")
        result.append("\n")

        # Line 3: Compile status (critical for performance)
        infra = self._snapshot.tamiyo.infrastructure
        result.append("Compile:", style="dim")

        if infra.compile_enabled:
            # Show backend:mode
            backend = infra.compile_backend or "inductor"
            mode = infra.compile_mode or "default"
            result.append(f" {backend}:{mode}", style="green")
            result.append(" ✓", style="green bold")
        else:
            # Eager mode = potential 3-5x slower
            result.append(" EAGER ", style="red bold reverse")
            result.append("(no compile)", style="dim")

        return result

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
