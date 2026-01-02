"""EpisodeMetricsPanel - Episode-level training metrics.

Displays (Training mode):
- Episode length statistics (mean/std/range)
- Outcome rates (timeout, success, early termination)
- Steps-per-action metrics + trend

Displays (Warmup mode - before PPO updates):
- Episodes collected progress
- Random policy baseline returns
- Reward sparsity check

Per DRL expert: Warmup metrics establish the floor that PPO needs to beat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class EpisodeMetricsPanel(Static):
    """Episode-level metrics panel with warmup/training mode switching.

    Shows warmup diagnostics before PPO updates start,
    then switches to training metrics once PPO data arrives.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "EPISODE HEALTH"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot

        # Update border title based on mode
        if not snapshot.tamiyo.ppo_data_received:
            self.border_title = "WARMUP"
        else:
            stats = snapshot.episode_stats
            if stats.total_episodes > 0:
                self.border_title = f"EPISODE HEALTH ─ {stats.total_episodes} ep"
            else:
                self.border_title = "EPISODE HEALTH"
        self.refresh()

    def render(self) -> Text:
        """Render episode metrics (warmup or training mode)."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        # Check if we're in warmup mode
        if not self._snapshot.tamiyo.ppo_data_received:
            return self._render_warmup()
        else:
            return self._render_training()

    def _render_warmup(self) -> Text:
        """Render warmup mode diagnostics.

        Shows metrics relevant before PPO updates start:
        - Collection progress
        - Random policy baseline
        - Data quality indicators
        """
        result = Text()
        tamiyo = self._snapshot.tamiyo
        history = list(tamiyo.episode_return_history)

        # Line 1: Collection status
        result.append("Collecting warmup data...\n", style="cyan")

        # Line 2: Random policy baseline (if we have episodes)
        if history:
            h_mean = sum(history) / len(history)
            h_std = (sum((x - h_mean) ** 2 for x in history) / len(history)) ** 0.5
            mean_style = "green" if h_mean >= 0 else "red"
            result.append("Baseline ", style="dim")
            result.append(f"μ:{h_mean:+.1f}", style=mean_style)
            result.append(f" σ:{h_std:.1f}", style="cyan")
            result.append(f"  n:{len(history)}", style="dim")
        else:
            result.append("Baseline ---  (waiting)", style="dim")

        return result

    def _render_training(self) -> Text:
        """Render training mode episode metrics."""
        stats = self._snapshot.episode_stats
        result = Text()

        # Line 1: Length statistics
        result.append("Length ", style="dim")
        result.append(f"μ:{stats.length_mean:.0f}", style="cyan")
        result.append(f" σ:{stats.length_std:.0f}", style="dim")
        result.append("\n")

        # Line 2: Outcome rates (compact)
        result.append("✗", style="red dim")
        timeout_style = self._rate_style(stats.timeout_rate, bad_high=True)
        result.append(f"{stats.timeout_rate:.0%}", style=timeout_style)

        result.append(" ✓", style="green dim")
        success_style = self._rate_style(stats.success_rate, bad_high=False)
        result.append(f"{stats.success_rate:.0%}", style=success_style)

        result.append(" ⊗", style="yellow dim")
        early_style = self._rate_style(stats.early_termination_rate, bad_high=True)
        result.append(f"{stats.early_termination_rate:.0%}", style=early_style)

        # Trend indicator
        trend_map = {
            "improving": ("↗", "green"),
            "stable": ("→", "dim"),
            "declining": ("↘", "red"),
        }
        arrow, style = trend_map.get(stats.completion_trend, ("→", "dim"))
        result.append(f" {arrow}", style=style)

        return result

    def _rate_style(self, rate: float, bad_high: bool = True) -> str:
        """Get style for a rate based on whether high is bad.

        Args:
            rate: Value between 0 and 1.
            bad_high: If True, high values are bad (red). If False, high is good (green).
        """
        if bad_high:
            # High is bad (timeout, early termination)
            if rate > 0.2:
                return "red"
            if rate > 0.1:
                return "yellow"
            return "green"
        else:
            # High is good (success rate)
            if rate > 0.7:
                return "green"
            if rate > 0.5:
                return "yellow"
            return "red" if rate < 0.3 else "dim"
