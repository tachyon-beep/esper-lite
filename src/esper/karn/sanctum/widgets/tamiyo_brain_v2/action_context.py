"""ActionContext - Action distribution, sequence, and slot summary.

Combines:
- Action distribution bar (stacked bar showing G/A/F/P/W proportions)
- Recent action sequence with pattern detection (STUCK, THRASH, ALPHA_OSC)
- Slot stage summary across all environments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.leyline import STAGE_COLORS, STAGE_ABBREVIATIONS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


# Action colors
ACTION_COLORS: dict[str, str] = {
    "GERMINATE": "green",
    "SET_ALPHA_TARGET": "cyan",
    "FOSSILIZE": "blue",
    "PRUNE": "red",
    "WAIT": "dim",
}

ACTION_ABBREVS: dict[str, str] = {
    "GERMINATE": "G",
    "SET_ALPHA_TARGET": "A",
    "FOSSILIZE": "F",
    "PRUNE": "P",
    "WAIT": "W",
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


class ActionContext(Container):
    """Action context panel with distribution, sequence, and slots."""

    BAR_WIDTH: ClassVar[int] = 30

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Static(id="action-bar")
        yield Static(id="action-sequence")
        yield Static(id="return-history")
        yield Static(id="slot-summary")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot

        self.query_one("#action-bar", Static).update(self._render_action_bar())
        self.query_one("#action-sequence", Static).update(self._render_action_sequence())
        self.query_one("#return-history", Static).update(self._render_return_history())
        self.query_one("#slot-summary", Static).update(self._render_slot_summary())

    def _render_action_bar(self) -> Text:
        """Render horizontal stacked bar for action distribution."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        total = tamiyo.total_actions

        if total == 0:
            return Text("[no actions yet]", style="dim")

        # Calculate percentages
        pcts = {a: (c / total) * 100 for a, c in tamiyo.action_counts.items()}

        # Build stacked bar
        result = Text()
        result.append("[")

        for action in ["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "WAIT"]:
            pct = pcts.get(action, 0)
            width = int((pct / 100) * self.BAR_WIDTH)
            if width > 0:
                result.append("▓" * width, style=ACTION_COLORS.get(action, "white"))

        result.append("]")

        # Compact legend: 18G 4A 0F 17P 40W
        result.append(" ")
        for action in ["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "WAIT"]:
            pct = pcts.get(action, 0)
            abbrev = ACTION_ABBREVS[action]
            color = ACTION_COLORS[action]
            result.append(f"{pct:.0f}{abbrev}", style=color)
            result.append(" ")

        return result

    def _render_action_sequence(self) -> Text:
        """Render recent action sequence with pattern warnings."""
        if self._snapshot is None:
            return Text("Recent: (no data)", style="dim")

        decisions = self._snapshot.tamiyo.recent_decisions[:24]
        if not decisions:
            return Text("Recent: (no actions yet)", style="dim")

        # Get slot states for pattern detection
        slot_states = decisions[0].slot_states if decisions else {}
        patterns = detect_action_patterns(decisions[:12], slot_states)

        result = Text()

        # Recent row (most recent 12)
        recent_decisions = decisions[:12]
        recent_actions = [
            (ACTION_ABBREVS.get(d.chosen_action, "?"), ACTION_COLORS.get(d.chosen_action, "white"))
            for d in recent_decisions
        ]
        recent_actions.reverse()  # Oldest first for left-to-right reading

        is_stuck = "STUCK" in patterns
        is_thrash = "THRASH" in patterns

        result.append("Recent: ", style="dim")
        for char, color in recent_actions:
            style = "yellow" if is_stuck else ("red" if is_thrash else color)
            result.append(f"{char} ", style=style)

        # Pattern warnings
        if is_stuck:
            result.append(" ⚠ STUCK", style="yellow bold")
        if is_thrash:
            result.append(" ⚡ THRASH", style="red bold")
        if "ALPHA_OSC" in patterns:
            result.append(" ↔ ALPHA", style="cyan bold")

        return result

    def _render_return_history(self) -> Text:
        """Render recent episode returns."""
        if self._snapshot is None:
            return Text("Returns: (no data)", style="dim")

        tamiyo = self._snapshot.tamiyo
        history = list(tamiyo.episode_return_history)

        if not history:
            return Text("Returns: (no episodes yet)", style="dim")

        result = Text()
        result.append("Returns: ", style="dim")

        # Show last 6 returns (most recent first)
        current_ep = tamiyo.current_episode
        recent_returns = history[-6:][::-1]

        for i, ret in enumerate(recent_returns):
            ep_num = current_ep - i
            style = "green" if ret >= 0 else "red"
            result.append(f"Ep{ep_num}:", style="dim")
            result.append(f"{ret:+.1f}", style=style)
            if i < len(recent_returns) - 1:
                result.append("  ")

        return result

    def _render_slot_summary(self) -> Text:
        """Render aggregate slot state across all environments."""
        if self._snapshot is None:
            return Text("SLOTS: (no data)", style="dim")

        snapshot = self._snapshot
        counts = snapshot.slot_stage_counts
        total = snapshot.total_slots

        if total == 0:
            return Text("SLOTS: (no environments)", style="dim")

        result = Text()

        # Header line
        n_envs = len(snapshot.envs) if snapshot.envs else 0
        result.append(f"SLOTS ({total} across {n_envs} envs)\n", style="bold dim")

        # Stage distribution with mini-bars
        stages = ["DORMANT", "GERMINATED", "TRAINING", "BLENDING", "HOLDING", "FOSSILIZED"]
        stage_abbrevs = {k: v.upper() for k, v in STAGE_ABBREVIATIONS.items()}

        for i, stage in enumerate(stages):
            count = counts.get(stage, 0)
            abbrev = stage_abbrevs.get(stage, stage[:4])
            color = STAGE_COLORS.get(stage, "dim")

            # Proportional bar (max 4 chars)
            bar_width = min(4, int((count / max(1, total)) * 16)) if total > 0 else 0
            bar_char = "█" if stage != "DORMANT" else "░"
            bar = bar_char * bar_width

            result.append(f"{abbrev}:", style="dim")
            result.append(f"{count}", style=color)
            if bar:
                result.append(f" {bar}", style=color)

            if i < len(stages) - 1:
                result.append("  ")

        result.append("\n")

        # Summary stats
        foss = snapshot.cumulative_fossilized
        pruned = snapshot.cumulative_pruned
        rate = (foss / max(1, foss + pruned)) * 100 if (foss + pruned) > 0 else 0
        avg_epochs = snapshot.avg_epochs_in_stage

        result.append("Foss:", style="dim")
        result.append(f"{foss}", style="blue")
        result.append("  Prune:", style="dim")
        result.append(f"{pruned}", style="red" if pruned > foss else "dim")
        result.append("  Rate:", style="dim")
        rate_color = "green" if rate >= 70 else "yellow" if rate >= 50 else "red"
        result.append(f"{rate:.0f}%", style=rate_color)
        result.append("  AvgAge:", style="dim")
        result.append(f"{avg_epochs:.1f}", style="cyan")
        result.append(" epochs", style="dim")

        return result
