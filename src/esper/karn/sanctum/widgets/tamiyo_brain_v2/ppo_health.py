"""PPOHealthPanel - Gauges and metrics for PPO training health.

Consolidates warmup status to panel header (not repeated per-gauge).

Layout:
    ┌─ PPO HEALTH ─────────────────────────── WARMING UP [5/50] ─┐
    │ Expl.Var   -0.005  [██░░░░░░░░]    Advantage  +0.00±1.00  │
    │ Entropy     7.89   [██████████]    Ratio   0.98 < r < 1.02│
    │ Clip Frac   0.000  [░░░░░░░░░░]    Policy Loss   -0.350   │
    │ KL Div      0.000  [░░░░░░░░░░]    Value Loss    33.757   │
    │                                    Grad Norm      1.00    │
    │                                    Layers       12/12 ✓   │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

from .primary_metrics import render_sparkline, detect_trend, trend_style

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class PPOHealthPanel(Container):
    """PPO training health panel with gauges and metrics."""

    WARMUP_BATCHES: ClassVar[int] = 50
    TOTAL_LAYERS: ClassVar[int] = 12
    GAUGE_WIDTH: ClassVar[int] = 10

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "PPO HEALTH"

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        with Horizontal(id="ppo-content"):
            yield Static(id="gauge-column")
            yield Static(id="metrics-column", classes="metrics-column")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot

        # Update border title with collapse risk or warmup status
        batch = snapshot.current_batch
        if snapshot.tamiyo.collapse_risk_score > 0.7:
            velocity = snapshot.tamiyo.entropy_velocity
            if velocity < 0:
                distance = snapshot.tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
                batches = int(distance / abs(velocity)) if velocity != 0 else 999
                self.border_title = f"PPO HEALTH !! COLLAPSE ~{batches}b"
            else:
                self.border_title = "PPO HEALTH"
        elif batch < self.WARMUP_BATCHES:
            self.border_title = f"PPO HEALTH ─ WARMING UP [{batch}/{self.WARMUP_BATCHES}]"
        else:
            self.border_title = "PPO HEALTH"

        # Update gauge column
        gauge_col = self.query_one("#gauge-column", Static)
        gauge_col.update(self._render_gauges())

        # Update metrics column
        metrics_col = self.query_one("#metrics-column", Static)
        metrics_col.update(self._render_metrics())

    def _render_gauges(self) -> Text:
        """Render the 2x2 gauge grid with value-first layout."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        tamiyo = self._snapshot.tamiyo
        batch = self._snapshot.current_batch
        is_warmup = batch < self.WARMUP_BATCHES

        result = Text()

        # Row 1: Explained Variance
        result.append(self._render_gauge_row(
            label="Expl.Var",
            value=tamiyo.explained_variance,
            min_val=-1.0,
            max_val=1.0,
            status=self._get_ev_status(tamiyo.explained_variance),
            is_warmup=is_warmup,
        ))
        result.append("\n")

        # Row 2: Entropy
        result.append(self._render_gauge_row(
            label="Entropy",
            value=tamiyo.entropy,
            min_val=0.0,
            max_val=2.0,
            status=self._get_entropy_status(tamiyo.entropy),
            is_warmup=is_warmup,
        ))
        result.append("\n")

        # Row 3: Clip Fraction
        result.append(self._render_gauge_row(
            label="Clip Frac",
            value=tamiyo.clip_fraction,
            min_val=0.0,
            max_val=0.5,
            status=self._get_clip_status(tamiyo.clip_fraction),
            is_warmup=is_warmup,
        ))
        result.append("\n")

        # Row 4: KL Divergence
        result.append(self._render_gauge_row(
            label="KL Div",
            value=tamiyo.kl_divergence,
            min_val=0.0,
            max_val=0.1,
            status=self._get_kl_status(tamiyo.kl_divergence),
            is_warmup=is_warmup,
        ))

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

        Format: Label      Value  [████░░░░░░]  Status
        """
        result = Text()

        # Label (left-aligned, 10 chars)
        result.append(f"{label:<10}", style="dim")

        # Value (right-aligned, 8 chars) - value first!
        if abs(value) < 0.1:
            result.append(f"{value:>8.3f}", style=self._status_style(status))
        else:
            result.append(f"{value:>8.2f}", style=self._status_style(status))

        # Gap
        result.append("  ")

        # Bar
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        normalized = max(0, min(1, normalized))
        filled = int(normalized * self.GAUGE_WIDTH)
        empty = self.GAUGE_WIDTH - filled

        bar_style = self._status_style(status)
        result.append("[")
        result.append("█" * filled, style=bar_style)
        result.append("░" * empty, style="dim")
        result.append("]")

        # Status indicator (only show if not warmup)
        if not is_warmup:
            if status == "critical":
                result.append(" !", style="red bold")
            elif status == "warning":
                result.append(" *", style="yellow")

        return result

    def _render_metrics(self) -> Text:
        """Render the metrics column."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Advantage stats
        adv_status = self._get_advantage_status(tamiyo.advantage_std)
        adv_style = self._status_style(adv_status)
        result.append("Advantage    ", style="dim")
        result.append(f"{tamiyo.advantage_mean:+.2f} ± {tamiyo.advantage_std:.2f}", style=adv_style)
        if adv_status != "ok":
            result.append(" !", style=adv_style)
        result.append("\n")

        # Ratio bounds
        ratio_status = self._get_ratio_status(tamiyo.ratio_min, tamiyo.ratio_max)
        ratio_style = self._status_style(ratio_status)
        result.append("Ratio        ", style="dim")
        result.append(f"{tamiyo.ratio_min:.2f} < r < {tamiyo.ratio_max:.2f}", style=ratio_style)
        result.append("\n")

        # Policy loss with sparkline
        if tamiyo.policy_loss_history:
            pl_sparkline = render_sparkline(tamiyo.policy_loss_history, width=15)
            pl_trend = detect_trend(list(tamiyo.policy_loss_history))
            result.append("Policy Loss  ", style="dim")
            result.append(pl_sparkline)
            result.append(f" {tamiyo.policy_loss:>7.3f} ", style="bright_cyan")
            result.append(pl_trend, style=trend_style(pl_trend, "loss"))
        else:
            result.append("Policy Loss  ", style="dim")
            result.append(f"{tamiyo.policy_loss:>7.3f}", style="bright_cyan")
        result.append("\n")

        # Value loss with sparkline
        if tamiyo.value_loss_history:
            vl_sparkline = render_sparkline(tamiyo.value_loss_history, width=15)
            vl_trend = detect_trend(list(tamiyo.value_loss_history))
            result.append("Value Loss   ", style="dim")
            result.append(vl_sparkline)
            result.append(f" {tamiyo.value_loss:>7.3f} ", style="bright_cyan")
            result.append(vl_trend, style=trend_style(vl_trend, "loss"))
        else:
            result.append("Value Loss   ", style="dim")
            result.append(f"{tamiyo.value_loss:>7.3f}", style="bright_cyan")
        result.append("\n")

        # Grad norm
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        gn_style = self._status_style(gn_status)
        result.append("Grad Norm    ", style="dim")
        result.append(f"{tamiyo.grad_norm:>7.2f}", style=gn_style)
        if gn_status != "ok":
            result.append(" !", style=gn_style)
        result.append("\n")

        # Layer health
        result.append("Layers       ", style="dim")
        healthy = self.TOTAL_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            result.append(
                f"{tamiyo.dead_layers}D/{tamiyo.exploding_layers}E",
                style="red",
            )
        else:
            result.append(f"{healthy}/{self.TOTAL_LAYERS} ✓", style="green")
        result.append("\n")

        # Entropy trend (velocity and collapse countdown)
        result.append(self._render_entropy_trend())

        return result

    def _render_entropy_trend(self) -> Text:
        """Render entropy trend with velocity and countdown."""
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        velocity = tamiyo.entropy_velocity
        risk = tamiyo.collapse_risk_score

        result = Text()
        result.append("Entropy D    ", style="dim")

        EPSILON = 1e-6
        if abs(velocity) < 0.005:
            result.append("stable [--]", style="green")
            return result

        # Trend arrows
        if velocity < -0.03:
            arrow = "[vv]"
            arrow_style = "red bold"
        elif velocity < -0.01:
            arrow = "[v]"
            arrow_style = "yellow"
        elif velocity > 0.01:
            arrow = "[^]"
            arrow_style = "green"
        else:
            arrow = "[~]"
            arrow_style = "dim"

        result.append(f"{velocity:+.3f}/b ", style=arrow_style)
        result.append(arrow, style=arrow_style)

        # Countdown (only if declining toward critical)
        if velocity < -EPSILON and tamiyo.entropy > TUIThresholds.ENTROPY_CRITICAL:
            distance = tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
            batches_to_collapse = int(distance / abs(velocity))

            if batches_to_collapse < 100:
                result.append(f" ~{batches_to_collapse}b", style="yellow")

            if risk > 0.7:
                result.append(" [ALERT]", style="red bold")

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

    def _get_kl_status(self, kl: float) -> str:
        if kl > TUIThresholds.KL_CRITICAL:
            return "critical"
        if kl > TUIThresholds.KL_WARNING:
            return "warning"
        return "ok"

    def _get_advantage_status(self, adv_std: float) -> str:
        if adv_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            return "warning"
        if adv_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
            return "warning"
        return "ok"

    def _get_ratio_status(self, ratio_min: float, ratio_max: float) -> str:
        if ratio_max > TUIThresholds.RATIO_MAX_CRITICAL or ratio_min < TUIThresholds.RATIO_MIN_CRITICAL:
            return "critical"
        if ratio_max > TUIThresholds.RATIO_MAX_WARNING or ratio_min < TUIThresholds.RATIO_MIN_WARNING:
            return "warning"
        return "ok"

    def _get_grad_norm_status(self, grad_norm: float) -> str:
        if grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical"
        if grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning"
        return "ok"

    def _status_style(self, status: str) -> str:
        return {"ok": "bright_cyan", "warning": "yellow", "critical": "red bold"}[status]
