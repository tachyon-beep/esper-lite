"""LossesPanel - Episode return and loss metrics.

Displays:
- Episode Return sparkline with trend
- Policy Loss sparkline with trend
- Value Loss sparkline with trend
- Value/Policy loss ratio
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from .primary_metrics import render_sparkline, detect_trend, trend_style

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class LossesPanel(Static):
    """Episode return and loss metrics panel.

    Extends Static directly for minimal layout overhead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "LOSSES"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        """Render episode return and loss metrics."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Episode Return sparkline (primary metric)
        result.append("Ep.Return ", style="bold cyan")
        if tamiyo.episode_return_history:
            sparkline = render_sparkline(tamiyo.episode_return_history, width=8)
            ep_trend = detect_trend(list(tamiyo.episode_return_history))
            result.append(sparkline)
            result.append(f" {tamiyo.current_episode_return:>5.1f}", style="white")
            result.append(ep_trend, style=trend_style(ep_trend, "accuracy"))
        else:
            result.append("─" * 8, style="dim")
        result.append("\n")

        # Policy loss with sparkline
        result.append("P.Loss    ", style="dim")
        if tamiyo.policy_loss_history:
            pl_sparkline = render_sparkline(tamiyo.policy_loss_history, width=8)
            pl_trend = detect_trend(list(tamiyo.policy_loss_history))
            result.append(pl_sparkline)
            result.append(f" {tamiyo.policy_loss:>6.3f}", style="bright_cyan")
            result.append(pl_trend, style=trend_style(pl_trend, "loss"))
        else:
            result.append(f"{tamiyo.policy_loss:>6.3f}", style="bright_cyan")
        result.append("\n")

        # Value loss with sparkline
        result.append("V.Loss    ", style="dim")
        if tamiyo.value_loss_history:
            vl_sparkline = render_sparkline(tamiyo.value_loss_history, width=8)
            vl_trend = detect_trend(list(tamiyo.value_loss_history))
            result.append(vl_sparkline)
            result.append(f" {tamiyo.value_loss:>6.3f}", style="bright_cyan")
            result.append(vl_trend, style=trend_style(vl_trend, "loss"))
        else:
            result.append(f"{tamiyo.value_loss:>6.3f}", style="bright_cyan")
        result.append("\n")

        # Value/Policy loss ratio
        result.append(self._render_loss_ratio(tamiyo.value_loss, tamiyo.policy_loss))
        result.append("\n\n\n")  # Extra lines to match Health panel height

        return result

    def _render_loss_ratio(self, value_loss: float, policy_loss: float) -> Text:
        """Render Value/Policy loss ratio with DRL-appropriate thresholds."""
        result = Text()
        result.append("L_v/L_p   ", style="dim")

        # Handle edge cases
        if abs(policy_loss) < 1e-10:
            result.append("---", style="dim")
            return result

        ratio = abs(value_loss) / abs(policy_loss)

        # Determine status
        if ratio < 0.1 or ratio > 10.0:
            status = "critical"
        elif ratio < 0.2 or ratio > 5.0:
            status = "warning"
        else:
            status = "ok"

        style = {"ok": "bright_cyan", "warning": "yellow", "critical": "red bold"}[status]
        result.append(f"{ratio:>7.2f}", style=style)

        # Add interpretation hint
        if status == "ok":
            result.append(" ✓", style="green dim")
        elif ratio < 0.2:
            result.append(" P>V", style=style)  # Policy dominating
        elif ratio > 5.0:
            result.append(" V>P", style=style)  # Value dominating

        return result
