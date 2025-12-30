"""ActionContext - Action distribution and sequence display.

Layout (6 lines):
1. This batch: [▓▓▓▓░░░░░░░░░░░░░░░░] bar for current training batch
2. G:09 A:03 F:00 P:15 V:02 W:71 (this batch percentages)
3. Total run: [▓▓▓▓▓▓▓▓░░░░░░░░░░░░] cumulative bar across all batches
4. G:22 A:05 F:00 P:21 V:02 W:50 (total run percentages)
5. ⚠ STUCK G→G→A→W→W→W→W→W→P→G→A→F (recent sequence + patterns)
6. Returns: +1.2 -0.3 +2.1 -0.8 +0.5 +1.7  avg:+0.7↗
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


# Action colors (must match decisions_column.py and attention_heatmap.py)
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
    """Action context panel with distribution, sequence, and returns.

    Extends Static directly (like DecisionCard) to eliminate Container
    layout overhead that causes whitespace issues.

    Layout (6 lines):
    1-2. This batch: bar + percentages
    3-4. Total run: bar + percentages
    5. Recent sequence with pattern warnings
    6. Returns with trend
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "ACTION CONTEXT"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()  # Trigger render()

    def render(self) -> Text:
        """Render all action context sections (6 lines)."""
        result = Text()
        result.append(self._render_action_bars())  # Lines 1-4: both bars
        result.append("\n")
        result.append(self._render_action_sequence())  # Line 5
        result.append("\n")
        result.append(self._render_return_history())  # Line 6
        return result

    def _render_action_bars(self) -> Text:
        """Render both action bars (this batch + total run).

        Lines 1-2: This batch bar + percentages
        Lines 3-4: Total run bar + percentages

        Format:
        This batch: [▓▓▓▓░░░░░░░░░░░░░░░░]
        G:09 A:03 F:00 P:15 V:02 W:71
        Total run: [▓▓▓▓▓▓▓▓░░░░░░░░░░░░]
        G:22 A:05 F:00 P:21 V:02 W:50
        """
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # This batch: use tamiyo.action_counts (batch-specific, resets at BATCH_EPOCH_COMPLETED)
        batch_counts = tamiyo.action_counts
        batch_total = tamiyo.total_actions

        # === THIS BATCH (lines 1-2) ===
        bar_width = 30
        result.append("This Batch: [")

        if batch_total > 0:
            # Calculate widths proportionally, ensuring total = bar_width
            actions = ["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "ADVANCE", "WAIT"]
            widths = []
            for action in actions:
                count = batch_counts.get(action, 0)
                pct = (count / batch_total) * 100
                widths.append(int((pct / 100) * bar_width))

            # Fix rounding errors: distribute remainder to actions with counts
            total_width = sum(widths)
            if total_width < bar_width:
                remainder = bar_width - total_width
                # Add remainder to first action with non-zero count
                for i, action in enumerate(actions):
                    if batch_counts.get(action, 0) > 0:
                        widths[i] += remainder
                        break

            # Render segments
            for action, width in zip(actions, widths):
                if width > 0:
                    result.append("▓" * width, style=ACTION_COLORS.get(action, "white"))
        else:
            result.append("░" * bar_width, style="dim")

        result.append("]\n")

        # This batch percentages
        for i, action in enumerate(["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "ADVANCE", "WAIT"]):
            count = batch_counts.get(action, 0)
            pct = int((count / batch_total) * 100) if batch_total > 0 else 0
            abbrev = ACTION_ABBREVS[action]
            color = ACTION_COLORS[action] if count > 0 else "dim"

            if i > 0:
                result.append(" ")
            result.append(f"{abbrev}:", style="dim")
            result.append(f"{pct:02d}", style=color)

        result.append("\n\n")  # Extra newline for spacing between sections

        # === TOTAL RUN (lines 3-4) ===
        # Use cumulative counts across all batches
        cumulative_counts = tamiyo.cumulative_action_counts
        cumulative_total = tamiyo.cumulative_total_actions
        result.append("Total Run: [")

        if cumulative_total > 0:
            # Calculate widths proportionally, ensuring total = bar_width
            widths = []
            for action in actions:
                count = cumulative_counts.get(action, 0)
                pct = (count / cumulative_total) * 100
                widths.append(int((pct / 100) * bar_width))

            # Fix rounding errors: distribute remainder to actions with counts
            total_width = sum(widths)
            if total_width < bar_width:
                remainder = bar_width - total_width
                # Add remainder to first action with non-zero count
                for i, action in enumerate(actions):
                    if cumulative_counts.get(action, 0) > 0:
                        widths[i] += remainder
                        break

            # Render segments
            for action, width in zip(actions, widths):
                if width > 0:
                    result.append("▓" * width, style=ACTION_COLORS.get(action, "white"))
        else:
            result.append("░" * bar_width, style="dim")

        result.append("]\n")

        # Total run percentages
        for i, action in enumerate(["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "ADVANCE", "WAIT"]):
            count = cumulative_counts.get(action, 0)
            pct = int((count / cumulative_total) * 100) if cumulative_total > 0 else 0
            abbrev = ACTION_ABBREVS[action]
            color = ACTION_COLORS[action] if count > 0 else "dim"

            if i > 0:
                result.append(" ")
            result.append(f"{abbrev}:", style="dim")
            result.append(f"{pct:02d}", style=color)

        result.append("\n")  # Blank line between Total Run and Recent:

        return result

    def _render_action_sequence(self) -> Text:
        """Render recent action sequence with pattern warnings FIRST.

        Format: ⚠ STUCK  G→G→A→W→W→W→W→W→P→G→A→F
        Pattern warnings are placed first for visibility.
        Arrows between actions improve readability.
        """
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        decisions = self._snapshot.tamiyo.recent_decisions[:24]
        if not decisions:
            return Text("[no actions yet]", style="dim")

        # Get slot states for pattern detection
        slot_states = decisions[0].slot_states if decisions else {}
        patterns = detect_action_patterns(decisions[:12], slot_states)

        result = Text()

        # Pattern warnings FIRST (prominent placement)
        is_stuck = "STUCK" in patterns
        is_thrash = "THRASH" in patterns
        is_alpha_osc = "ALPHA_OSC" in patterns

        if is_stuck:
            result.append("⚠ STUCK ", style="yellow bold reverse")
        if is_thrash:
            result.append("⚡ THRASH ", style="red bold reverse")
        if is_alpha_osc:
            result.append("↔ ALPHA ", style="cyan bold reverse")

        if not patterns:
            result.append("Recent: ", style="dim")

        # Recent row (most recent 12) with arrows
        recent_decisions = decisions[:12]
        recent_actions = [
            (ACTION_ABBREVS.get(d.chosen_action, "?"), ACTION_COLORS.get(d.chosen_action, "white"))
            for d in recent_decisions
        ]
        recent_actions.reverse()  # Oldest first for left-to-right reading

        for i, (char, color) in enumerate(recent_actions):
            # Override color if pattern detected
            if is_stuck:
                style = "yellow"
            elif is_thrash:
                style = "red"
            else:
                style = color

            result.append(char, style=style)
            # Arrow separator (except for last)
            if i < len(recent_actions) - 1:
                result.append("→", style="dim")

        return result

    def _render_return_history(self) -> Text:
        """Render recent episode returns with mean and trend.

        Format: Returns: +1.2 -0.3 +2.1 -0.8 +0.5 +1.7  avg:+0.7↗
        Compact values (no EpN: prefix), mean + trend arrow at end.
        """
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        history = list(tamiyo.episode_return_history)

        if not history:
            return Text("Returns: [no episodes yet]", style="dim")

        result = Text()
        result.append("Returns: ", style="dim")

        # Show last 8 returns (most recent first), compact format
        recent_returns = history[-8:][::-1]

        for i, ret in enumerate(recent_returns):
            style = "green" if ret >= 0 else "red"
            result.append(f"{ret:+.1f}", style=style)
            if i < len(recent_returns) - 1:
                result.append(" ")

        # Calculate mean and trend
        if len(history) >= 2:
            mean_ret = sum(history) / len(history)
            # Trend: compare recent half to older half
            mid = len(history) // 2
            if mid > 0:
                old_mean = sum(history[:mid]) / mid
                new_mean = sum(history[mid:]) / (len(history) - mid)
                delta = new_mean - old_mean

                result.append("  ", style="dim")
                # Mean with trend arrow
                mean_style = "green" if mean_ret >= 0 else "red"
                result.append(f"avg:{mean_ret:+.1f}", style=mean_style)

                # Trend arrow
                if delta > 0.3:
                    result.append("↗", style="green bold")
                elif delta < -0.3:
                    result.append("↘", style="red bold")
                else:
                    result.append("─", style="dim")
        elif len(history) == 1:
            # Single episode - just show it
            result.append(f"  avg:{history[0]:+.1f}", style="dim")

        return result
