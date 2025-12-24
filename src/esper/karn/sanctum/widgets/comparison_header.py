"""Comparison Header for A/B testing - shows delta metrics."""

from __future__ import annotations

from textual.widget import Widget
from rich.text import Text


class ComparisonHeader(Widget):
    """Shows comparison metrics between A/B policies."""

    DEFAULT_CSS = """
    ComparisonHeader {
        height: 3;
        dock: top;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._group_a_accuracy = 0.0
        self._group_b_accuracy = 0.0
        self._group_a_reward = 0.0
        self._group_b_reward = 0.0
        self._leader: str | None = None

    @property
    def leader(self) -> str | None:
        """Return group ID of current leader."""
        return self._leader

    def update_comparison(
        self,
        group_a_accuracy: float,
        group_b_accuracy: float,
        group_a_reward: float,
        group_b_reward: float,
    ) -> None:
        """Update comparison metrics."""
        self._group_a_accuracy = group_a_accuracy
        self._group_b_accuracy = group_b_accuracy
        self._group_a_reward = group_a_reward
        self._group_b_reward = group_b_reward

        # Determine leader: reward-first (primary RL objective), accuracy as tiebreaker
        reward_delta = group_a_reward - group_b_reward
        mean_reward = (abs(group_a_reward) + abs(group_b_reward)) / 2

        # Significant reward difference (>5% of mean) is decisive
        if mean_reward > 0 and abs(reward_delta) > 0.05 * mean_reward:
            self._leader = "A" if reward_delta > 0 else "B"
        # Fallback to accuracy for close reward races
        elif group_a_accuracy > group_b_accuracy:
            self._leader = "A"
        elif group_b_accuracy > group_a_accuracy:
            self._leader = "B"
        else:
            self._leader = None

        self.refresh()

    def render(self) -> Text:
        """Render comparison bar."""
        delta_acc = self._group_a_accuracy - self._group_b_accuracy
        delta_reward = self._group_a_reward - self._group_b_reward

        result = Text()
        result.append("A/B Comparison │ ")

        # Accuracy delta
        sign = "+" if delta_acc >= 0 else ""
        if abs(delta_acc) > 5:
            style = "green bold" if delta_acc > 0 else "red bold"
        else:
            style = "dim"
        result.append(f"Acc Δ: {sign}{delta_acc:.1f}% ", style=style)

        result.append("│ ")

        # Reward delta
        sign = "+" if delta_reward >= 0 else ""
        result.append(f"Reward Δ: {sign}{delta_reward:.2f} ", style="dim")

        result.append("│ ")

        # Leader indicator
        if self._leader:
            color = "green" if self._leader == "A" else "cyan"
            result.append(f"Leading: {self._leader}", style=f"{color} bold")
        else:
            result.append("Tied", style="dim italic")

        return result
