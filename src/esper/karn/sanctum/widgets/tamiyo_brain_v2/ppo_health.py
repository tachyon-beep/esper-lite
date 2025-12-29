"""PPOHealthPanel - Gauges for PPO training health.

Displays core PPO metrics with visual gauges:
- Explained Variance
- Entropy
- Clip Fraction
- KL Divergence
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class PPOHealthPanel(Static):
    """PPO training gauges panel.

    Extends Static directly for minimal layout overhead.
    """

    WARMUP_BATCHES: ClassVar[int] = 50
    GAUGE_WIDTH: ClassVar[int] = 10

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "PPO HEALTH"

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

        self.refresh()  # Trigger render()

    def render(self) -> Text:
        """Render the gauges panel."""
        result = self._render_gauges()
        result.append("\n\n\n")  # Extra lines to match Health panel height
        return result

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

    def _status_style(self, status: str) -> str:
        return {"ok": "bright_cyan", "warning": "yellow", "critical": "red bold"}[status]
