"""PPOLossesPanel - Combined PPO gauges and loss sparklines.

Displays PPO health metrics with visual gauges in top section,
and loss sparklines with trends in bottom section.

Top Section (PPO Gauges):
- Explained Variance (gauge bar)
- Entropy (gauge bar)
- Clip Fraction (gauge bar) + directional breakdown (↑↓)

Separator line

Bottom Section (Loss Sparklines):
- Episode Return (sparkline + value + trend)
- Policy Loss (sparkline + value + trend)
- Value Loss (sparkline + value + trend)
- Loss Ratio (value + status)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

from .sparkline_utils import detect_trend, render_sparkline, trend_style

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class PPOLossesPanel(Static):
    """Combined PPO health gauges and loss sparklines panel.

    Extends Static directly for minimal layout overhead.
    Designed for narrow width (~38 chars) in the new layout.
    """

    WARMUP_BATCHES: ClassVar[int] = 50
    GAUGE_WIDTH: ClassVar[int] = 8  # Narrower for compact layout

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "PPO LOSSES"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot

        # Update border title with collapse risk or warmup status
        batch = snapshot.current_batch
        if snapshot.tamiyo.collapse_risk_score > 0.7:
            velocity = snapshot.tamiyo.entropy_velocity
            if velocity < 0:
                distance = snapshot.tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
                # Clamp negative distance (already below critical) to 0
                distance = max(0.0, distance)
                batches = int(distance / abs(velocity)) if velocity != 0 else 999
                self.border_title = f"PPO LOSSES !! COLLAPSE ~{batches}b"
            else:
                self.border_title = "PPO LOSSES"
        elif batch < self.WARMUP_BATCHES:
            self.border_title = (
                f"PPO LOSSES \u2500 WARMING UP [{batch}/{self.WARMUP_BATCHES}]"
            )
        else:
            self.border_title = "PPO LOSSES"

        self.refresh()  # Trigger render()

    def render(self) -> Text:
        """Render the combined panel."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        result = Text()

        # Top section: PPO gauges
        result.append(self._render_gauges())

        # Separator line
        result.append("\n")
        result.append("\u2500" * 32, style="dim")
        result.append("\n")

        # Bottom section: Loss sparklines
        result.append(self._render_sparklines())

        return result

    def _render_gauges(self) -> Text:
        """Render the PPO gauge rows."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        tamiyo = self._snapshot.tamiyo
        batch = self._snapshot.current_batch
        is_warmup = batch < self.WARMUP_BATCHES

        result = Text()

        # Row 1: Explained Variance
        result.append(
            self._render_gauge_row(
                label="Expl.Var ",
                value=tamiyo.explained_variance,
                min_val=-1.0,
                max_val=1.0,
                status=self._get_ev_status(tamiyo.explained_variance),
                is_warmup=is_warmup,
            )
        )
        result.append("\n")

        # Row 2: Entropy
        result.append(
            self._render_gauge_row(
                label="Entropy  ",
                value=tamiyo.entropy,
                min_val=0.0,
                max_val=2.0,
                status=self._get_entropy_status(tamiyo.entropy),
                is_warmup=is_warmup,
            )
        )
        result.append("\n")

        # Row 3: Clip Fraction (with directional breakdown)
        result.append(
            self._render_gauge_row(
                label="Clip Frac",
                value=tamiyo.clip_fraction,
                min_val=0.0,
                max_val=0.5,
                status=self._get_clip_status(tamiyo.clip_fraction),
                is_warmup=is_warmup,
            )
        )
        # Add directional breakdown with arrows (always show, dim when zero)
        clip_pos = tamiyo.gradient_quality.clip_fraction_positive
        clip_neg = tamiyo.gradient_quality.clip_fraction_negative
        # Style: dim when both zero, otherwise show direction that's active
        dir_style = "dim" if clip_pos == 0 and clip_neg == 0 else "cyan"
        result.append(f" (\u2191{clip_pos:.1%} \u2193{clip_neg:.1%})", style=dir_style)

        return result

    def _render_gauge_row(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        status: str,
        is_warmup: bool = False,
    ) -> Text:
        """Render a single gauge row with value-first layout.

        Format: Label  Value [bar] Status (compact for narrow panel)
        """
        result = Text()

        # Label (left-aligned, 8 chars for compact layout)
        result.append(f"{label:<8}", style="dim")

        # Value (right-aligned, 6 chars)
        if abs(value) < 0.1:
            result.append(f"{value:>6.3f}", style=self._status_style(status))
        else:
            result.append(f"{value:>6.2f}", style=self._status_style(status))

        # Gap
        result.append(" ")

        # Bar (compact)
        normalized = (
            (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        )
        normalized = max(0, min(1, normalized))
        filled = int(normalized * self.GAUGE_WIDTH)
        empty = self.GAUGE_WIDTH - filled

        bar_style = self._status_style(status)
        result.append("[")
        result.append("\u2588" * filled, style=bar_style)
        result.append("\u2591" * empty, style="dim")
        result.append("]")

        # Status indicator (only show if not warmup)
        if not is_warmup:
            if status == "critical":
                result.append(" !", style="red bold")
            elif status == "warning":
                result.append(" *", style="yellow")

        return result

    def _render_sparklines(self) -> Text:
        """Render the loss sparkline rows.

        Fixed layout: Label(7) + Sparkline(12) + Value(8) + Trend(1)
        Values use space flag for sign alignment: " 0.123" vs "-0.123"
        """
        if self._snapshot is None:
            return Text("No data", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Sparkline width - longer for better trend visibility
        SPARK_W = 12

        # Policy loss with sparkline
        result.append("P.Loss ", style="dim")  # 7 chars
        if tamiyo.policy_loss_history:
            pl_sparkline = render_sparkline(tamiyo.policy_loss_history, width=SPARK_W)
            pl_trend = detect_trend(list(tamiyo.policy_loss_history))
            result.append(pl_sparkline)
            result.append(f" {tamiyo.policy_loss: 7.4f}", style="bright_cyan")
            result.append(pl_trend, style=trend_style(pl_trend, "loss"))
        else:
            result.append("─" * SPARK_W, style="dim")
            result.append(f" {tamiyo.policy_loss: 7.4f}", style="bright_cyan")
        result.append("\n")

        # Value loss with sparkline
        result.append("V.Loss ", style="dim")  # 7 chars
        if tamiyo.value_loss_history:
            vl_sparkline = render_sparkline(tamiyo.value_loss_history, width=SPARK_W)
            vl_trend = detect_trend(list(tamiyo.value_loss_history))
            result.append(vl_sparkline)
            result.append(f" {tamiyo.value_loss: 7.4f}", style="bright_cyan")
            result.append(vl_trend, style=trend_style(vl_trend, "loss"))
        else:
            result.append("─" * SPARK_W, style="dim")
            result.append(f" {tamiyo.value_loss: 7.4f}", style="bright_cyan")
        result.append("\n")

        # Value/Policy loss ratio
        result.append(self._render_loss_ratio(tamiyo.value_loss, tamiyo.policy_loss))

        return result

    def _render_loss_ratio(self, value_loss: float, policy_loss: float) -> Text:
        """Render Value/Policy loss ratio with DRL-appropriate thresholds.

        Aligned with sparkline rows: Label(7) + Pad(12) + Space(1) + Value(8)
        """
        result = Text()
        result.append("Lv/Lp  ", style="dim")  # 7 chars

        # Padding to align with sparkline rows (12 chars for sparkline)
        result.append(" " * 12, style="dim")

        # Handle edge cases
        if abs(policy_loss) < 1e-10:
            result.append("     ---", style="dim")  # 8 chars
            return result

        ratio = abs(value_loss) / abs(policy_loss)

        # Determine status
        if ratio < 0.1 or ratio > 10.0:
            status = "critical"
        elif ratio < 0.2 or ratio > 5.0:
            status = "warning"
        else:
            status = "ok"

        style = {"ok": "bright_cyan", "warning": "yellow", "critical": "red bold"}[
            status
        ]
        # Use space flag for sign alignment (ratio is always positive but aligns with losses)
        result.append(f" {ratio: 7.2f}", style=style)  # 8 chars total

        # Add interpretation hint
        if status == "ok":
            result.append(" ✓", style="green dim")
        elif ratio < 0.2:
            result.append(" P>V", style=style)  # Policy dominating
        elif ratio > 5.0:
            result.append(" V>P", style=style)  # Value dominating

        return result

    # Status helpers
    def _get_ev_status(self, ev: float) -> str:
        if ev < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical"
        if ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning"
        return "ok"

    def _get_entropy_status(self, entropy: float) -> str:
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "critical"
        if entropy < TUIThresholds.ENTROPY_WARNING:
            return "warning"
        return "ok"

    def _get_clip_status(self, clip: float) -> str:
        if clip > TUIThresholds.CLIP_CRITICAL:
            return "critical"
        if clip > TUIThresholds.CLIP_WARNING:
            return "warning"
        return "ok"

    def _status_style(self, status: str) -> str:
        # Use cyan for ok (visible but not loud), yellow/red for problems
        return {"ok": "cyan", "warning": "yellow", "critical": "red bold"}[status]
