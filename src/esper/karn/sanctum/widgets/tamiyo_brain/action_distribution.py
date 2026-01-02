"""ActionContext - Consolidated decision context panel.

Unified view of critic preferences, reward health, returns, and action distribution.

Layout:
    ┌─ ACTION CONTEXT ─────────────────────────────────┐
    │ ▶ Critic Preference ────────────────────────────│
    │   GERM ████████░░░░░░░░░  +1.2  ← BEST          │
    │   ADVN █████░░░░░░░░░░░░  +0.5                  │
    │   ...                                           │
    │   Var:0.34✓  Spread:2.3                         │
    │─────────────────────────────────────────────────│
    │ ▶ Reward Signal ────────────────────────────────│
    │   PBRS:25%✓  Gaming:2%✓  HV:1.2                 │
    │   Σ:+0.35  Sig:+0.42  Rent:-0.07               │
    │   αShk:-0.01 Stage:+0.10 HS:+0.05              │
    │─────────────────────────────────────────────────│
    │ ▶ Returns ──────────────────────────────────────│
    │   +1.2 -0.3 +2.1 -0.8 +0.5                      │
    │   min:-0.8 max:+2.1 μ:+0.5 σ:1.1 ↗              │
    │─────────────────────────────────────────────────│
    │ ▶ Chosen Actions ───────────────────────────────│
    │   Batch: [▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░]       │
    │          G:09 A:03 F:00 P:15 V:02 W:71          │
    │   Run:   [▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░]       │
    │          G:22 A:05 F:00 P:21 V:02 W:50          │
    │─────────────────────────────────────────────────│
    │ ▶ Sequence ─────────────────────────────────────│
    │   G✓→G✓→A✗→W✓→W✓→W✓→W✓→W✓→P✓→G✓→A✓→F✓           │
    │   Last: GERMINATE ✓                             │
    └─────────────────────────────────────────────────┘

Section order follows decision causality:
  Critic → Reward → Returns → Actions → Sequence
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot
    from esper.karn.sanctum.widgets.reward_health import RewardHealthData


# Action colors (must match decisions_column.py and action_heads_panel.py)
ACTION_COLORS: dict[str, str] = {
    "GERMINATE": "green",
    "SET_ALPHA_TARGET": "cyan",
    "FOSSILIZE": "blue",
    "PRUNE": "red",
    "WAIT": "dim",
    "ADVANCE": "cyan",
}

ACTION_ABBREVS: dict[str, str] = {
    "GERMINATE": "G",
    "SET_ALPHA_TARGET": "A",
    "FOSSILIZE": "F",
    "PRUNE": "P",
    "WAIT": "W",
    "ADVANCE": "V",  # V for adVance (A is taken by SET_ALPHA_TARGET)
}

# Full names for Q-value display (sorted by preference)
ACTION_NAMES: dict[str, str] = {
    "GERMINATE": "GERM",
    "SET_ALPHA_TARGET": "ALPH",
    "FOSSILIZE": "FOSS",
    "PRUNE": "PRUN",
    "WAIT": "WAIT",
    "ADVANCE": "ADVN",
}


def detect_action_patterns(
    decisions: list["DecisionSnapshot"],
    slot_states: dict[str, str],
) -> list[str]:
    """Detect problematic action patterns.

    - STUCK: All WAIT when dormant slots exist (missed opportunities)
    - THRASH: Germinate→Prune cycles (wasted compute)
    - ALPHA_OSC: Too many alpha changes (unstable policy)
    """
    patterns: list[str] = []
    if not decisions:
        return patterns

    # Reverse to chronological order for pattern analysis
    actions = [d.chosen_action for d in reversed(decisions[:12])]

    # STUCK: All WAIT when dormant slots exist
    if len(actions) >= 8 and all(a == "WAIT" for a in actions[-8:]):
        has_dormant = any(
            "Dormant" in str(s) or "Empty" in str(s) for s in slot_states.values()
        )
        has_training = any("Training" in str(s) for s in slot_states.values())
        if has_dormant and not has_training:
            patterns.append("STUCK")

    # THRASH: Germinate-Prune cycles
    germ_prune_cycles = 0
    for i in range(len(actions) - 1):
        if actions[i] == "GERMINATE" and actions[i + 1] == "PRUNE":
            germ_prune_cycles += 1
    if germ_prune_cycles >= 2:
        patterns.append("THRASH")

    # ALPHA_OSC: Too many alpha changes
    alpha_count = sum(1 for a in actions if a == "SET_ALPHA_TARGET")
    if alpha_count >= 4:
        patterns.append("ALPHA_OSC")

    return patterns


class ActionContext(Static):
    """Consolidated decision context panel.

    Combines critic preferences, reward health, returns, action distribution,
    and sequence into a unified view following decision causality order.
    """

    # Q-value bar rendering
    Q_BAR_WIDTH: ClassVar[int] = 16
    SEPARATOR_WIDTH: ClassVar[int] = 38

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._reward_health: RewardHealthData | None = None
        self.classes = "panel"
        self.border_title = "ACTION CONTEXT"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def update_reward_health(self, data: "RewardHealthData") -> None:
        """Update reward health data (from SanctumApp)."""
        self._reward_health = data
        self.refresh()

    def render(self) -> Text:
        """Render consolidated action context."""
        result = Text()

        # Section 1: Critic Preference (sorted Q-values with bars)
        result.append("▶ Critic Preference\n", style="bold")
        result.append(self._render_critic_preference())

        # Separator
        result.append("─" * self.SEPARATOR_WIDTH + "\n", style="dim")

        # Section 2: Reward Signal
        result.append("▶ Reward Signal\n", style="bold")
        result.append(self._render_reward_signal())

        # Separator
        result.append("─" * self.SEPARATOR_WIDTH + "\n", style="dim")

        # Section 3: Returns (5 values + stats)
        result.append("▶ Returns\n", style="bold")
        result.append(self._render_returns())

        # Separator
        result.append("─" * self.SEPARATOR_WIDTH + "\n", style="dim")

        # Section 4: Chosen Actions (batch + run bars)
        result.append("▶ Chosen Actions\n", style="bold")
        result.append(self._render_action_bars())

        # Separator
        result.append("─" * self.SEPARATOR_WIDTH + "\n", style="dim")

        # Section 5: Sequence
        result.append("▶ Sequence\n", style="bold")
        result.append(self._render_action_sequence())

        return result

    # =========================================================================
    # Section 1: Critic Preference
    # =========================================================================

    def _render_critic_preference(self) -> Text:
        """Render Q-values sorted by preference with normalized bars.

        Shows critic's preference ranking across operations.
        Bars are min-max normalized to show relative preference.
        """
        if self._snapshot is None:
            return Text("  [no data]\n", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Collect Q-values with action info
        q_values = [
            ("GERMINATE", tamiyo.q_germinate),
            ("ADVANCE", tamiyo.q_advance),
            ("SET_ALPHA_TARGET", tamiyo.q_set_alpha),
            ("FOSSILIZE", tamiyo.q_fossilize),
            ("PRUNE", tamiyo.q_prune),
            ("WAIT", tamiyo.q_wait),
        ]

        # Handle NaN values
        valid_qs = [(a, q) for a, q in q_values if not math.isnan(q)]
        if not valid_qs:
            result.append("  [waiting for data]\n", style="dim")
            return result

        # Sort by Q-value (highest first)
        sorted_qs = sorted(valid_qs, key=lambda x: x[1], reverse=True)

        # Get min/max for normalization
        q_min = min(q for _, q in sorted_qs)
        q_max = max(q for _, q in sorted_qs)
        q_range = q_max - q_min if q_max != q_min else 1.0

        # Render each Q-value row
        for i, (action, q_val) in enumerate(sorted_qs):
            name = ACTION_NAMES.get(action, action[:4])
            color = ACTION_COLORS.get(action, "white")

            # Normalize to 0-1 for bar fill
            fill_pct = (q_val - q_min) / q_range

            # Render: "  GERM ████████░░░░░░░░  +1.2"
            result.append(f"  {name:<4} ", style=color)

            # Bar
            filled = int(fill_pct * self.Q_BAR_WIDTH)
            empty = self.Q_BAR_WIDTH - filled
            result.append("█" * filled, style=color)
            result.append("░" * empty, style="dim")

            # Value
            result.append(f"  {q_val:+.1f}", style=color)

            # Best/Worst markers
            if i == 0:
                result.append("  ← BEST", style="green dim")
            elif i == len(sorted_qs) - 1:
                result.append("  ← WORST", style="red dim")

            result.append("\n")

        # Q-variance and spread on one line
        q_var = tamiyo.q_variance
        var_status = self._get_q_variance_status(q_var)
        var_style = {"ok": "green", "warning": "yellow", "critical": "red bold"}[var_status]

        result.append("  Var:", style="dim")
        result.append(f"{q_var:.2f}", style=var_style)

        if var_status == "critical":
            result.append("✗", style="red")
        elif var_status == "ok":
            result.append("✓", style="green")

        result.append(f"  Spread:{tamiyo.q_spread:.1f}\n", style="dim")

        return result

    def _get_q_variance_status(self, q_variance: float) -> str:
        """Check if Q-variance indicates op conditioning is working."""
        if math.isnan(q_variance):
            return "ok"
        if q_variance < 0.01:
            return "critical"
        if q_variance < 0.1:
            return "warning"
        return "ok"

    # =========================================================================
    # Section 2: Reward Signal
    # =========================================================================

    def _render_reward_signal(self) -> Text:
        """Render reward health metrics and component breakdown.

        4 lines × 3 columns grid layout:
        Line 1: PBRS | Gaming | HV
        Line 2: Σ | Sig | Rent
        Line 3: αShk | Ratio | Stage
        Line 4: Foss | HS | (empty)
        """
        result = Text()
        col_width = 10  # Fixed column width for alignment

        # Line 1: Health metrics (PBRS, Gaming, HV)
        result.append("  ")
        if self._reward_health is not None:
            rh = self._reward_health

            # PBRS fraction (10-40% healthy)
            pbrs_color = "green" if rh.is_pbrs_healthy else "red"
            pbrs_icon = "✓" if rh.is_pbrs_healthy else "✗"
            pbrs_text = f"PBRS:{rh.pbrs_fraction:.0%}{pbrs_icon}"
            result.append(pbrs_text.ljust(col_width), style=pbrs_color)
            result.append(" ")

            # Gaming rate (<5% healthy)
            gaming_color = "green" if rh.is_gaming_healthy else "red"
            gaming_icon = "✓" if rh.is_gaming_healthy else "✗"
            gaming_text = f"Gam:{rh.anti_gaming_trigger_rate:.0%}{gaming_icon}"
            result.append(gaming_text.ljust(col_width), style=gaming_color)
            result.append(" ")

            # Hypervolume
            hv_text = f"HV:{rh.hypervolume:.1f}"
            result.append(hv_text, style="cyan")
        else:
            result.append("[health pending]", style="dim")
        result.append("\n")

        # Lines 2-4: Component breakdown from snapshot
        if self._snapshot is not None:
            rc = self._snapshot.rewards

            # Line 2: Σ | Sig | Rent
            result.append("  ")
            total_style = "green" if rc.total >= 0 else "red"
            result.append(f"Σ:{rc.total:+.2f}".ljust(col_width), style=total_style)
            result.append(" ")

            sig = rc.bounded_attribution if rc.bounded_attribution != 0 else rc.base_acc_delta
            sig_style = "green" if sig >= 0 else "red"
            result.append(f"Sig:{sig:+.2f}".ljust(col_width), style=sig_style)
            result.append(" ")

            rent_style = "yellow" if rc.compute_rent != 0 else "dim"
            result.append(f"Rent:{rc.compute_rent:.2f}", style=rent_style)
            result.append("\n")

            # Line 3: αShk | Ratio | Stage
            result.append("  ")
            ashk_style = "red" if rc.alpha_shock != 0 else "dim"
            result.append(f"αShk:{rc.alpha_shock:.2f}".ljust(col_width), style=ashk_style)
            result.append(" ")

            ratio_style = "red" if rc.ratio_penalty != 0 else "dim"
            result.append(f"Rat:{rc.ratio_penalty:.2f}".ljust(col_width), style=ratio_style)
            result.append(" ")

            stage_style = "green" if rc.stage_bonus != 0 else "dim"
            result.append(f"Stg:{rc.stage_bonus:+.2f}", style=stage_style)
            result.append("\n")

            # Line 4: Foss | HS
            result.append("  ")
            foss_style = "blue bold" if rc.fossilize_terminal_bonus != 0 else "dim"
            result.append(f"Foss:{rc.fossilize_terminal_bonus:+.2f}".ljust(col_width), style=foss_style)
            result.append(" ")

            hs_style = "cyan" if rc.hindsight_credit != 0 else "dim"
            result.append(f"HS:{rc.hindsight_credit:+.2f}", style=hs_style)
            result.append("\n")

        return result

    # =========================================================================
    # Section 3: Returns
    # =========================================================================

    def _render_returns(self) -> Text:
        """Render returns: 5 recent values + percentiles + stats.

        Layout (3 lines):
          Line 1: Last 5 returns (most recent first)
          Line 2: Percentiles p10/p50/p90 + spread warning
          Line 3: min/max/μ/σ + trend
        """
        if self._snapshot is None:
            return Text("  [no data]\n", style="dim")

        tamiyo = self._snapshot.tamiyo
        history = list(tamiyo.episode_return_history)

        if not history:
            # Show structure preview in dim grey
            result = Text()
            result.append("  ---  ---  ---  ---  ---\n", style="dim")
            result.append("  p10:---  p50:---  p90:---\n", style="dim")
            result.append("  min:---  max:---  μ:---  σ:---\n", style="dim")
            return result

        result = Text()

        # Line 1: Last 5 returns (most recent first)
        result.append("  ")
        recent = history[-5:][::-1]
        for i, ret in enumerate(recent):
            style = "green" if ret >= 0 else "red"
            result.append(f"{ret:+.1f}", style=style)
            if i < len(recent) - 1:
                result.append(" ")
        result.append("\n")

        # Line 2: Percentiles (p10/p50/p90) - catches bimodal policies
        result.append("  ")
        if len(history) >= 5:
            sorted_h = sorted(history)
            n = len(sorted_h)
            p10 = sorted_h[int(n * 0.1)]
            p50 = sorted_h[int(n * 0.5)]  # Median
            p90 = sorted_h[int(n * 0.9)]

            p10_style = "red" if p10 < 0 else "green"
            p50_style = "red" if p50 < 0 else "green"
            p90_style = "red" if p90 < 0 else "green"

            result.append("p10:", style="dim")
            result.append(f"{p10:+.1f}", style=p10_style)
            result.append(" p50:", style="dim")
            result.append(f"{p50:+.1f}", style=p50_style)
            result.append(" p90:", style="dim")
            result.append(f"{p90:+.1f}", style=p90_style)

            # Spread warning: large p90-p10 gap indicates bimodal/inconsistent policy
            spread = p90 - p10
            if spread > 50:
                result.append(" ⚠⚠", style="red bold")
            elif spread > 20:
                result.append(" ⚠", style="yellow bold")
        else:
            result.append("p10:---  p50:---  p90:---", style="dim")
        result.append("\n")

        # Line 3: Stats (min, max, mean, std, trend)
        result.append("  ")

        if len(history) >= 2:
            h_min = min(history)
            h_max = max(history)
            h_mean = sum(history) / len(history)

            # Std dev
            variance = sum((x - h_mean) ** 2 for x in history) / len(history)
            h_std = variance ** 0.5

            # Trend (compare recent half to older half)
            mid = len(history) // 2
            if mid > 0:
                old_mean = sum(history[:mid]) / mid
                new_mean = sum(history[mid:]) / (len(history) - mid)
                delta = new_mean - old_mean

                if delta > 0.3:
                    trend = "↗"
                    trend_style = "green bold"
                elif delta < -0.3:
                    trend = "↘"
                    trend_style = "red bold"
                else:
                    trend = "─"
                    trend_style = "dim"
            else:
                trend = ""
                trend_style = "dim"

            min_style = "red" if h_min < 0 else "green"
            max_style = "green" if h_max >= 0 else "red"
            mean_style = "green" if h_mean >= 0 else "red"

            result.append("min:", style="dim")
            result.append(f"{h_min:+.1f}", style=min_style)
            result.append(" max:", style="dim")
            result.append(f"{h_max:+.1f}", style=max_style)
            result.append(" μ:", style="dim")
            result.append(f"{h_mean:+.1f}", style=mean_style)
            result.append(" σ:", style="dim")
            result.append(f"{h_std:.1f}", style="cyan")
            result.append(f" {trend}", style=trend_style)
        else:
            # Single value
            result.append(f"μ:{history[0]:+.1f}", style="dim")

        result.append("\n")
        return result

    # =========================================================================
    # Section 4: Chosen Actions
    # =========================================================================

    def _render_action_bars(self) -> Text:
        """Render batch and run action distribution bars."""
        if self._snapshot is None:
            return Text("  [no data]\n", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        actions = [
            "GERMINATE",
            "SET_ALPHA_TARGET",
            "FOSSILIZE",
            "PRUNE",
            "ADVANCE",
            "WAIT",
        ]
        bar_width = 28

        # === THIS BATCH ===
        batch_counts = tamiyo.action_counts
        batch_total = tamiyo.total_actions

        result.append("  Batch: [")
        if batch_total > 0:
            widths = self._compute_bar_widths(batch_counts, actions, batch_total, bar_width)
            for action, width in zip(actions, widths):
                if width > 0:
                    result.append("▓" * width, style=ACTION_COLORS.get(action, "white"))
        else:
            result.append("░" * bar_width, style="dim")
        result.append("]\n")

        # Batch percentages
        result.append("         ")
        for i, action in enumerate(actions):
            count = batch_counts.get(action, 0)
            pct = int((count / batch_total) * 100) if batch_total > 0 else 0
            abbrev = ACTION_ABBREVS[action]
            color = ACTION_COLORS[action] if count > 0 else "dim"
            if i > 0:
                result.append(" ")
            result.append(f"{abbrev}:", style="dim")
            result.append(f"{pct:02d}", style=color)
        result.append("\n")

        # === THIS RUN ===
        cumulative_counts = tamiyo.cumulative_action_counts
        cumulative_total = tamiyo.cumulative_total_actions

        result.append("  Run:   [")
        if cumulative_total > 0:
            widths = self._compute_bar_widths(cumulative_counts, actions, cumulative_total, bar_width)
            for action, width in zip(actions, widths):
                if width > 0:
                    result.append("▓" * width, style=ACTION_COLORS.get(action, "white"))
        else:
            result.append("░" * bar_width, style="dim")
        result.append("]\n")

        # Run percentages
        result.append("         ")
        for i, action in enumerate(actions):
            count = cumulative_counts.get(action, 0)
            pct = int((count / cumulative_total) * 100) if cumulative_total > 0 else 0
            abbrev = ACTION_ABBREVS[action]
            color = ACTION_COLORS[action] if count > 0 else "dim"
            if i > 0:
                result.append(" ")
            result.append(f"{abbrev}:", style="dim")
            result.append(f"{pct:02d}", style=color)
        result.append("\n")

        return result

    def _compute_bar_widths(
        self,
        counts: dict[str, int],
        actions: list[str],
        total: int,
        bar_width: int,
    ) -> list[int]:
        """Compute proportional bar segment widths."""
        widths = []
        for action in actions:
            count = counts.get(action, 0)
            pct = (count / total) * 100 if total > 0 else 0
            widths.append(int((pct / 100) * bar_width))

        # Fix rounding: add remainder to first non-zero action
        total_width = sum(widths)
        if total_width < bar_width:
            remainder = bar_width - total_width
            for i, action in enumerate(actions):
                if counts.get(action, 0) > 0:
                    widths[i] += remainder
                    break

        return widths

    # =========================================================================
    # Section 5: Sequence
    # =========================================================================

    def _render_action_sequence(self) -> Text:
        """Render recent action sequence with pattern warnings."""
        if self._snapshot is None:
            return Text("  [no data]\n", style="dim")

        tamiyo = self._snapshot.tamiyo
        decisions = tamiyo.recent_decisions[:24]
        if not decisions:
            return Text("  [no actions yet]\n", style="dim")

        # Get slot states for pattern detection
        slot_states = decisions[0].slot_states if decisions else {}
        patterns = detect_action_patterns(decisions[:12], slot_states)

        result = Text()

        # Pattern warnings (prominent)
        is_stuck = "STUCK" in patterns
        is_thrash = "THRASH" in patterns
        is_alpha_osc = "ALPHA_OSC" in patterns

        result.append("  ")
        if is_stuck:
            result.append("⚠ STUCK ", style="yellow bold reverse")
        if is_thrash:
            result.append("⚡ THRASH ", style="red bold reverse")
        if is_alpha_osc:
            result.append("↔ ALPHA ", style="cyan bold reverse")

        # Recent actions (12 most recent, oldest first for L→R reading)
        recent_decisions = decisions[:12]
        recent_actions = [
            (
                ACTION_ABBREVS.get(d.chosen_action, "?"),
                ACTION_COLORS.get(d.chosen_action, "white"),
            )
            for d in recent_decisions
        ]
        recent_actions.reverse()

        for i, (char, color) in enumerate(recent_actions):
            style = color
            if is_stuck:
                style = "yellow"
            elif is_thrash:
                style = "red"

            result.append(char, style=style)
            result.append("✓", style="green")

            if i < len(recent_actions) - 1:
                result.append("→", style="dim")

        result.append("\n")

        # Last action
        result.append("  Last: ", style="dim")
        result.append(
            f"{tamiyo.last_action_op} ",
            style=ACTION_COLORS.get(tamiyo.last_action_op, "white"),
        )
        marker = "✓" if tamiyo.last_action_success else "✗"
        marker_style = "green" if tamiyo.last_action_success else "red bold"
        result.append(marker, style=marker_style)
        result.append("\n")

        return result
