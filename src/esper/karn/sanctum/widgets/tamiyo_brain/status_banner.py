"""StatusBanner - Always-visible training narrative strip.

Top line is compact status + a few core metrics.
Two additional lines provide a quick mental model:
  NOW: what the system is doing
  WHY: top two drivers behind the current state
  NEXT: the expected recovery trigger / what to watch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

from .trends import trend_arrow_for_history

if TYPE_CHECKING:
    from collections import deque

    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState


class StatusBanner(Container):
    """Multi-line status + narrative banner."""

    WARMUP_BATCHES: ClassVar[int] = 50

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._spinner_frame: int = 0

    def compose(self) -> ComposeResult:
        """Compose the banner layout."""
        yield Static(id="banner-content")

    def on_mount(self) -> None:
        """Set initial content when widget mounts."""
        self.query_one("#banner-content", Static).update(self._render_banner_text())

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot and refresh display."""
        self._snapshot = snapshot
        self._update_status_classes()
        self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_CHARS)
        self.query_one("#banner-content", Static).update(self._render_banner_text())

    def _update_status_classes(self) -> None:
        """Update CSS classes and border title based on current status."""
        status, _, _ = self._get_overall_status()

        # Remove all status classes
        self.remove_class("status-ok", "status-warning", "status-critical", "status-warmup")
        self.add_class(f"status-{status}")

        # Update border title with compile status (per UX review)
        # Note: [[ escapes to [ in Textual markup
        if self._snapshot and self._snapshot.tamiyo.infrastructure.compile_enabled:
            self.border_title = "TAMIYO [[compiled]]"
        else:
            self.border_title = "TAMIYO"

        # Add group class if in A/B mode
        if self._snapshot and self._snapshot.tamiyo.group_id:
            self.remove_class("group-a", "group-b", "group-c")
            group = self._snapshot.tamiyo.group_id.lower()
            self.add_class(f"group-{group}")

    def _render_banner_text(self) -> Text:
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

        banner.append("NOW: ", style="dim")
        banner.append(f"{label}", style=style)
        banner.append("  ", style="dim")

        # Metrics (only after PPO data received)
        if tamiyo.ppo_data_received:
            self._append_metrics(banner, tamiyo)
            now_tail = self._render_now_tail()
            if now_tail:
                banner.append("  ", style="dim")
                banner.append(now_tail)
        else:
            banner.append("Waiting for PPO data...", style="cyan italic")

        banner.append("\n")
        banner.append(self._render_why_line(), style="dim")
        banner.append("\n")
        banner.append(self._render_next_line(), style="dim")

        return banner

    def _render_now_tail(self) -> Text:
        if self._snapshot is None:
            return Text()

        snapshot = self._snapshot
        tamiyo = snapshot.tamiyo

        parts = Text()

        # Env stall rate (derived from per-env status)
        n_envs = len(snapshot.envs)
        if n_envs > 0:
            stalled = sum(1 for env in snapshot.envs.values() if env.status == "stalled")
            stalled_rate = stalled / n_envs
            st_style = "dim"
            if stalled_rate > 0.9:
                st_style = "red bold"
            elif stalled_rate > 0.5:
                st_style = "yellow"
            parts.append(f"stall:{stalled_rate:.0%}", style=st_style)

        # Op mix (from per-batch action_counts)
        if tamiyo.total_actions > 0:
            wait_rate = tamiyo.action_counts["WAIT"] / tamiyo.total_actions
            prune_rate = tamiyo.action_counts["PRUNE"] / tamiyo.total_actions
            if parts.plain:
                parts.append("  ", style="dim")
            wait_style = "dim"
            if wait_rate > 0.95:
                wait_style = "red bold"
            elif wait_rate > 0.8:
                wait_style = "yellow"
            parts.append(f"WAIT:{wait_rate:.0%}", style=wait_style)
            if prune_rate > 0:
                parts.append(" ", style="dim")
                parts.append(f"PR:{prune_rate:.0%}", style="dim")

        # Slot occupancy (batch-size aware: uses counts, not norms)
        if snapshot.total_slots > 0:
            empty = snapshot.total_slots - snapshot.active_slots
            if parts.plain:
                parts.append("  ", style="dim")
            empty_style = "dim"
            if empty == 0 and snapshot.active_slots > 0:
                empty_style = "yellow"
            parts.append(f"empty:{empty}/{snapshot.total_slots}", style=empty_style)

        # Policy vs critic summary (compact identity for the operator)
        if parts.plain:
            parts.append("  ", style="dim")
        parts.append("pol:", style="dim")
        parts.append(self._policy_state_label(), style=self._policy_state_style())
        parts.append("  ", style="dim")
        parts.append("V:", style="dim")
        parts.append(self._critic_state_label(), style=self._critic_state_style())

        return parts

    def _render_why_line(self) -> str:
        drivers = self._top_drivers(max_items=2)
        if not drivers:
            return "WHY: —"
        return "WHY: " + "; ".join(drivers)

    def _render_next_line(self) -> str:
        return "NEXT: " + self._next_hint()

    def _top_drivers(self, *, max_items: int) -> list[str]:
        if self._snapshot is None:
            return []

        snapshot = self._snapshot
        tamiyo = snapshot.tamiyo

        scored: list[tuple[float, str]] = []

        if not tamiyo.ppo_data_received:
            scored.append((1.0, "telemetry → waiting PPO update"))

        # Env stall dominance
        n_envs = len(snapshot.envs)
        if n_envs > 0:
            stalled = sum(1 for env in snapshot.envs.values() if env.status == "stalled")
            stalled_rate = stalled / n_envs
            if stalled_rate > 0.5:
                scored.append((stalled_rate, f"envs → stalled {stalled_rate:.0%}"))

        # Slot occupancy bottleneck
        if snapshot.total_slots > 0:
            empty = snapshot.total_slots - snapshot.active_slots
            if empty == 0 and snapshot.active_slots > 0:
                scored.append((0.95, "slots → full (no empty)"))
            elif empty / snapshot.total_slots < 0.1:
                scored.append((0.7, f"slots → scarce empty ({empty}/{snapshot.total_slots})"))

        # Op distribution (WAIT dominance)
        if tamiyo.total_actions > 0:
            wait_rate = tamiyo.action_counts["WAIT"] / tamiyo.total_actions
            if wait_rate > 0.8:
                scored.append((wait_rate, f"op mix → WAIT {wait_rate:.0%}"))

        # Learning signal / critic health
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            scored.append((0.9, "signal → advantages collapsed"))
        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            scored.append((0.7, f"critic → EV {tamiyo.explained_variance:.2f}"))

        # Policy instability
        if tamiyo.collapse_risk_score > 0.7:
            scored.append((0.85, "policy → collapse risk"))
        elif tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            scored.append((0.6, f"policy → KL {tamiyo.kl_divergence:.3f}"))

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        return [msg for _, msg in scored_sorted[:max_items]]

    def _next_hint(self) -> str:
        if self._snapshot is None:
            return "—"

        snapshot = self._snapshot
        tamiyo = snapshot.tamiyo
        batch = snapshot.current_batch

        if not tamiyo.ppo_data_received:
            return "wait for first PPO update (watch Batch tick + KL)"

        if batch < self.WARMUP_BATCHES:
            return f"warmup ends at {self.WARMUP_BATCHES}; watch EV + Lv/Lp stabilize"

        if snapshot.total_slots > 0:
            empty = snapshot.total_slots - snapshot.active_slots
            if empty == 0 and snapshot.active_slots > 0:
                return "free an empty slot (PRUNE/ADVANCE) → enables GERMINATE"

        if tamiyo.total_actions > 0:
            wait_rate = tamiyo.action_counts["WAIT"] / tamiyo.total_actions
            if wait_rate > 0.8:
                return "watch WAIT% drop; indicates masks opened / actions diversified"

        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "watch EV rise above warning; critic learning signal recovering"

        if tamiyo.collapse_risk_score > 0.7:
            return "watch Entropy D turn positive; collapse risk receding"

        return "watch mean accuracy trend ↑ and stalled% ↓"

    def _policy_state_label(self) -> str:
        if self._snapshot is None:
            return "—"

        tamiyo = self._snapshot.tamiyo
        batch = self._snapshot.current_batch
        if batch < self.WARMUP_BATCHES:
            return "warmup"
        if tamiyo.collapse_risk_score > 0.7:
            return "risk"
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING or tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            return "drift"
        return "stable"

    def _policy_state_style(self) -> str:
        if self._snapshot is None:
            return "dim"

        tamiyo = self._snapshot.tamiyo
        if tamiyo.collapse_risk_score > 0.7:
            return "red bold"
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING or tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            return "yellow"
        return "green"

    def _critic_state_label(self) -> str:
        if self._snapshot is None:
            return "—"

        ev = self._snapshot.tamiyo.explained_variance
        if ev <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "bad"
        if ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "weak"
        return "ok"

    def _critic_state_style(self) -> str:
        if self._snapshot is None:
            return "dim"

        ev = self._snapshot.tamiyo.explained_variance
        if ev <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "red bold"
        if ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "yellow"
        return "green"

    def _append_metrics(self, banner: Text, tamiyo: "TamiyoState") -> None:
        """Append metric values to the banner with trend arrows."""
        # KL Divergence (lower is better = "loss" type)
        kl_style = self._metric_style(self._get_kl_status(tamiyo.kl_divergence))
        banner.append(f"KL:{tamiyo.kl_divergence:.3f}", style=kl_style)
        kl_arrow, kl_arrow_style = self._trend_arrow(
            tamiyo.kl_divergence_history, "kl_divergence", "loss"
        )
        if kl_arrow:
            banner.append(kl_arrow, style=kl_arrow_style)
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            banner.append("!", style=kl_style)
        banner.append("  ")

        # Batch progress (snapshot is non-None when this method is called)
        assert self._snapshot is not None
        batch = self._snapshot.current_batch
        max_batches = self._snapshot.max_batches
        banner.append(f"Batch:{batch}/{max_batches}", style="dim")

        # Memory as percentage (per UX review - more scannable than absolute)
        mem_pct = self._snapshot.tamiyo.infrastructure.memory_usage_percent
        if mem_pct > 0:
            if mem_pct > 90:
                mem_style = "red bold"
            elif mem_pct > 75:
                mem_style = "yellow"
            else:
                mem_style = "dim"
            banner.append(f"  [Mem:{mem_pct:.0f}%]", style=mem_style)

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

    def _get_kl_status(self, kl: float) -> str:
        if kl > TUIThresholds.KL_CRITICAL:
            return "critical"
        if kl > TUIThresholds.KL_WARNING:
            return "warning"
        return "ok"

    def _metric_style(self, status: str) -> str:
        """Convert status to Rich style.

        Uses dim for ok metrics to reduce visual noise - only highlight problems.
        """
        return {"ok": "dim", "warning": "yellow", "critical": "red bold"}[status]

    def _trend_arrow(
        self,
        history: "list[float] | deque[float] | None",
        metric_name: str,
        metric_type: str,
    ) -> tuple[str, str]:
        """Get trend arrow and style for a metric.

        Args:
            history: Recent metric values (oldest first).
            metric_name: Name for threshold lookup (e.g., "expl_var").
            metric_type: "loss" (lower=better) or "accuracy" (higher=better).

        Returns:
            Tuple of (arrow_char, style).
        """
        return trend_arrow_for_history(
            history,
            metric_name=metric_name,
            metric_type=metric_type,
        )
