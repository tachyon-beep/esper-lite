"""NarrativePanel - Tamiyo's "Now / Why / Next" guidance.

This panel replaces the old StatusBanner and provides a single place to answer:
- NOW: what the system is doing (status + key metrics)
- WHY: top drivers behind the current state
- NEXT: what to watch / expected recovery trigger
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

from .trends import trend_arrow_for_history

if TYPE_CHECKING:
    from collections import deque

    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState


class NarrativePanel(Static):
    """Now/Why/Next narrative panel."""

    WARMUP_BATCHES: ClassVar[int] = 50

    # Braille spinner characters for warmup animation
    SPINNER_CHARS: ClassVar[str] = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # A/B Group styling (inline identity tag)
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
        self.classes = "panel"
        self.border_title = "NARRATIVE"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_CHARS)

        title = "NARRATIVE"
        if snapshot.tamiyo.group_id:
            title += f" {snapshot.tamiyo.group_id}"
        if snapshot.tamiyo.infrastructure.compile_enabled:
            title += " [[compiled]]"
        self.border_title = title

        self.refresh()

    def render(self) -> Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        status, label, status_style = self._get_overall_status()
        tamiyo = self._snapshot.tamiyo

        result = Text()

        result.append("NOW", style="bold")
        result.append("  ", style="dim")
        result.append_text(self._render_now_line(status, label, status_style, tamiyo))
        result.append("\n")

        result.append("WHY", style="bold")
        drivers = self._top_drivers(max_items=2)
        if not drivers:
            result.append("  —", style="dim")
            result.append("\n")
        else:
            result.append("\n")
            for driver in drivers:
                result.append("  - ", style="dim")
                result.append(driver, style="dim")
                result.append("\n")

        result.append("NEXT", style="bold")
        result.append("  ", style="dim")
        result.append(self._next_hint(), style="dim")

        return result

    def _render_now_line(
        self,
        status: str,
        label: str,
        status_style: str,
        tamiyo: "TamiyoState",
    ) -> Text:
        now = Text()

        # NaN/Inf indicator FIRST - leftmost position for F-pattern visibility
        if tamiyo.nan_grad_count > 0 or tamiyo.inf_grad_count > 0:
            # Severity graduation: >5 issues = reverse video for maximum visibility
            if tamiyo.nan_grad_count > 5 or tamiyo.inf_grad_count > 5:
                nan_inf_style = "red bold reverse"
            else:
                nan_inf_style = "red bold"

            if tamiyo.nan_grad_count > 0:
                now.append(f"NaN:{tamiyo.nan_grad_count}", style=nan_inf_style)
                now.append(" ")
            if tamiyo.inf_grad_count > 0:
                now.append(f"Inf:{tamiyo.inf_grad_count}", style=nan_inf_style)
                now.append(" ")

        # Group label for A/B testing
        if tamiyo.group_id:
            group_color = (
                self.GROUP_COLORS[tamiyo.group_id]
                if tamiyo.group_id in self.GROUP_COLORS
                else "white"
            )
            group_label = (
                self.GROUP_LABELS[tamiyo.group_id]
                if tamiyo.group_id in self.GROUP_LABELS
                else tamiyo.group_id
            )
            now.append(f" {group_label} ", style=group_color)
            now.append(" ┃ ", style="dim")

        # Status icon (spinner during warmup, static otherwise)
        if status == "warmup":
            spinner = self.SPINNER_CHARS[self._spinner_frame]
            now.append(f"{spinner} ", style="cyan")
        else:
            icons = {"ok": "✓", "warning": "!", "critical": "✗"}
            icon = icons.get(status, "?")
            now.append(f"[{icon}] ", style=status_style)

        now.append(label, style=status_style)
        now.append("  ", style="dim")

        # Metrics (only after PPO data received)
        if tamiyo.ppo_data_received:
            self._append_metrics(now, tamiyo)
            now_tail = self._render_now_tail()
            if now_tail.plain:
                now.append("\n  ", style="dim")
                now.append(now_tail)
        else:
            now.append("Waiting for PPO data...", style="cyan italic")

        return now

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

        # Op mix (from per-round action_counts)
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

        # Slot occupancy (round-size aware: uses counts, not norms)
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
                scored.append(
                    (0.7, f"slots → scarce empty ({empty}/{snapshot.total_slots})")
                )

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
            return "wait for first PPO update (watch Round tick + KL)"

        if batch < self.WARMUP_BATCHES:
            return f"warmup ends at {self.WARMUP_BATCHES} rounds; watch EV + Lv/Lp stabilize"

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
        if (
            tamiyo.kl_divergence > TUIThresholds.KL_WARNING
            or tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING
        ):
            return "drift"
        return "stable"

    def _policy_state_style(self) -> str:
        if self._snapshot is None:
            return "dim"

        tamiyo = self._snapshot.tamiyo
        if tamiyo.collapse_risk_score > 0.7:
            return "red bold"
        if (
            tamiyo.kl_divergence > TUIThresholds.KL_WARNING
            or tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING
        ):
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

    def _append_metrics(self, now: Text, tamiyo: "TamiyoState") -> None:
        """Append core metrics to the NOW line."""
        # KL Divergence (lower is better = "loss" type)
        kl_style = self._metric_style(self._get_kl_status(tamiyo.kl_divergence))
        now.append(f"KL:{tamiyo.kl_divergence:.3f}", style=kl_style)
        kl_arrow, kl_arrow_style = self._trend_arrow(
            tamiyo.kl_divergence_history, "kl_divergence", "loss"
        )
        if kl_arrow:
            now.append(kl_arrow, style=kl_arrow_style)
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            now.append("!", style=kl_style)
        now.append("  ", style="dim")

        # Round progress
        assert self._snapshot is not None
        batch = self._snapshot.current_batch
        max_batches = self._snapshot.max_batches
        now.append(f"Round:{batch}/{max_batches}", style="dim")

        # Memory as percentage (more scannable than absolute)
        mem_pct = self._snapshot.tamiyo.infrastructure.memory_usage_percent
        if mem_pct > 0:
            if mem_pct > 90:
                mem_style = "red bold"
            elif mem_pct > 75:
                mem_style = "yellow"
            else:
                mem_style = "dim"
            now.append(f"  [Mem:{mem_pct:.0f}%]", style=mem_style)

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
        if tamiyo.nan_grad_count > 0:
            return "critical", "NaN DETECTED", "red bold"
        if tamiyo.inf_grad_count > 0:
            return "critical", "Inf DETECTED", "red bold"

        # Warmup period
        current_batch = self._snapshot.current_batch
        if current_batch < self.WARMUP_BATCHES:
            return (
                "warmup",
                f"WARMING UP [round {current_batch}/{self.WARMUP_BATCHES}]",
                "cyan",
            )

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
        """Get trend arrow and style for a metric."""
        return trend_arrow_for_history(
            history,
            metric_name=metric_name,
            metric_type=metric_type,
        )
