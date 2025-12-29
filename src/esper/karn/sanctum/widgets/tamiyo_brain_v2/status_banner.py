"""StatusBanner - One-line status summary with optional warmup spinner.

Displays:
    [OK] LEARNING   EV:0.72 Clip:0.18 KL:0.008 Adv:0.12±0.94 GradHP:12/12✓ batch:47/100

Or during warmup:
    ⠋ WARMING UP [5/50]   EV:0.00 Clip:0.00 KL:0.000 ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class StatusBanner(Static):
    """One-line status banner with key metrics."""

    WARMUP_BATCHES: ClassVar[int] = 50
    TOTAL_LAYERS: ClassVar[int] = 12

    # Braille spinner characters for warmup animation
    SPINNER_CHARS: ClassVar[str] = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # A/B Group styling
    GROUP_COLORS: ClassVar[dict[str, str]] = {
        "A": "bright_green",
        "B": "bright_cyan",
        "C": "bright_magenta",
    }

    GROUP_LABELS: ClassVar[dict[str, str]] = {
        "A": "🟢 A",
        "B": "🔵 B",
        "C": "🟣 C",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._spinner_frame: int = 0

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot and refresh display."""
        self._snapshot = snapshot
        self._update_status_classes()
        self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_CHARS)
        self.update(self._render_content())

    def _update_status_classes(self) -> None:
        """Update CSS classes based on current status."""
        status, _, _ = self._get_overall_status()

        # Remove all status classes
        self.remove_class("status-ok", "status-warning", "status-critical", "status-warmup")
        self.add_class(f"status-{status}")

        # Add group class if in A/B mode
        if self._snapshot and self._snapshot.tamiyo.group_id:
            self.remove_class("group-a", "group-b", "group-c")
            group = self._snapshot.tamiyo.group_id.lower()
            self.add_class(f"group-{group}")

    def _render_content(self) -> Text:
        """Render the banner content."""
        if self._snapshot is None:
            return Text("[?] NO DATA", style="cyan")

        status, label, style = self._get_overall_status()
        tamiyo = self._snapshot.tamiyo

        banner = Text()

        # NaN/Inf indicator FIRST - leftmost position for F-pattern visibility
        if tamiyo.nan_grad_count > 0 or tamiyo.inf_grad_count > 0:
            # Severity graduation: >5 issues = reverse video for maximum visibility
            if tamiyo.nan_grad_count > 5 or tamiyo.inf_grad_count > 5:
                nan_inf_style = "red bold reverse"
            else:
                nan_inf_style = "red bold"

            if tamiyo.nan_grad_count > 0:
                banner.append(f"NaN:{tamiyo.nan_grad_count}", style=nan_inf_style)
                banner.append(" ")
            if tamiyo.inf_grad_count > 0:
                banner.append(f"Inf:{tamiyo.inf_grad_count}", style=nan_inf_style)
                banner.append(" ")

        # Group label for A/B testing
        if tamiyo.group_id:
            group_color = self.GROUP_COLORS.get(tamiyo.group_id, "white")
            group_label = self.GROUP_LABELS.get(tamiyo.group_id, f"[{tamiyo.group_id}]")
            banner.append(f" {group_label} ", style=group_color)
            banner.append(" ┃ ", style="dim")

        # Status icon (spinner during warmup, static otherwise)
        if status == "warmup":
            spinner = self.SPINNER_CHARS[self._spinner_frame]
            banner.append(f"{spinner} ", style="cyan")
        else:
            icons = {"ok": "✓", "warning": "!", "critical": "✗"}
            icon = icons.get(status, "?")
            banner.append(f"[{icon}] ", style=style)

        banner.append(f"{label}   ", style=style)

        # Metrics (only after PPO data received)
        if tamiyo.ppo_data_received:
            self._append_metrics(banner, tamiyo)
        else:
            banner.append("Waiting for PPO data...", style="cyan italic")

        return banner

    def _append_metrics(self, banner: Text, tamiyo) -> None:
        """Append metric values to the banner."""
        # Explained Variance
        ev_style = self._metric_style(self._get_ev_status(tamiyo.explained_variance))
        banner.append(f"EV:{tamiyo.explained_variance:.2f}", style=ev_style)
        if tamiyo.explained_variance <= 0:
            banner.append("!", style="red")
        banner.append("  ")

        # Clip Fraction
        clip_style = self._metric_style(self._get_clip_status(tamiyo.clip_fraction))
        banner.append(f"Clip:{tamiyo.clip_fraction:.2f}", style=clip_style)
        if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            banner.append("!", style=clip_style)
        banner.append("  ")

        # KL Divergence
        kl_style = self._metric_style(self._get_kl_status(tamiyo.kl_divergence))
        banner.append(f"KL:{tamiyo.kl_divergence:.3f}", style=kl_style)
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            banner.append("!", style=kl_style)
        banner.append("  ")

        # Advantage
        adv_status = self._get_advantage_status(tamiyo.advantage_std)
        adv_style = self._metric_style(adv_status)
        banner.append(
            f"Adv:{tamiyo.advantage_mean:+.2f}±{tamiyo.advantage_std:.2f}",
            style=adv_style,
        )
        if adv_status != "ok":
            banner.append("!", style=adv_style)
        banner.append("  ")

        # Gradient Health
        healthy = self.TOTAL_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            banner.append(
                f"GradHP:{tamiyo.dead_layers}D/{tamiyo.exploding_layers}E",
                style="red",
            )
        else:
            banner.append(f"GradHP:{healthy}/{self.TOTAL_LAYERS}✓", style="green")
        banner.append("  ")

        # Batch progress
        batch = self._snapshot.current_batch
        max_batches = self._snapshot.max_batches
        banner.append(f"batch:{batch}/{max_batches}", style="dim")

    def _get_overall_status(self) -> tuple[str, str, str]:
        """Determine overall status using DRL decision tree.

        Returns:
            (status, label, style) tuple
        """
        if self._snapshot is None:
            return "ok", "WAITING", "cyan"

        tamiyo = self._snapshot.tamiyo

        if not tamiyo.ppo_data_received:
            return "ok", "WAITING", "cyan"

        # NaN/Inf check (HIGHEST PRIORITY - before all other checks)
        # These indicate numerical instability and should always be surfaced first
        # These return immediately without counting - they are special cases
        if tamiyo.nan_grad_count > 0:
            return "critical", "NaN DETECTED", "red bold"
        if tamiyo.inf_grad_count > 0:
            return "critical", "Inf DETECTED", "red bold"

        # Warmup period
        current_batch = self._snapshot.current_batch
        if current_batch < self.WARMUP_BATCHES:
            return "warmup", f"WARMING UP [{current_batch}/{self.WARMUP_BATCHES}]", "cyan"

        # Collect all critical issues
        critical_issues: list[str] = []
        if tamiyo.entropy < TUIThresholds.ENTROPY_CRITICAL:
            critical_issues.append("Entropy")
        if tamiyo.explained_variance <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            critical_issues.append("Value")
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            critical_issues.append("AdvLow")
        if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            critical_issues.append("AdvHigh")
        if tamiyo.kl_divergence > TUIThresholds.KL_CRITICAL:
            critical_issues.append("KL")
        if tamiyo.clip_fraction > TUIThresholds.CLIP_CRITICAL:
            critical_issues.append("Clip")
        if tamiyo.grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            critical_issues.append("Grad")

        if critical_issues:
            primary = critical_issues[0]
            if len(critical_issues) > 1:
                label = f"FAIL:{primary} (+{len(critical_issues) - 1})"
            else:
                label = f"FAIL:{primary}"
            return "critical", label, "red bold"

        # Collect all warning issues
        warning_issues: list[str] = []
        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            warning_issues.append("Value")
        if tamiyo.entropy < TUIThresholds.ENTROPY_WARNING:
            warning_issues.append("Entropy")
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            warning_issues.append("KL")
        if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            warning_issues.append("Clip")
        if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            warning_issues.append("AdvHigh")
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
            warning_issues.append("AdvLow")
        if tamiyo.grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            warning_issues.append("Grad")

        if warning_issues:
            primary = warning_issues[0]
            if len(warning_issues) > 1:
                label = f"WARN:{primary} (+{len(warning_issues) - 1})"
            else:
                label = f"WARN:{primary}"
            return "warning", label, "yellow"

        return "ok", "LEARNING", "green"

    def _get_ev_status(self, ev: float) -> str:
        if ev < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical"
        if ev < TUIThresholds.EXPLAINED_VAR_WARNING:
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

    def _metric_style(self, status: str) -> str:
        """Convert status to Rich style."""
        return {"ok": "green", "warning": "yellow", "critical": "red bold"}[status]
