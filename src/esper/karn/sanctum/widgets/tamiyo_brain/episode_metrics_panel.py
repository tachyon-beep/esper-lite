"""EpisodeMetricsPanel - Episode-level training metrics.

Displays:
- Episode length statistics (mean/std/range)
- Outcome rates (timeout, success, early termination)
- Steps-per-action metrics
- Completion trend indicator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class EpisodeMetricsPanel(Static):
    """Episode-level metrics panel.

    Extends Static directly for minimal layout overhead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "EPISODE HEALTH"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        # Update border title with total episodes
        stats = snapshot.episode_stats
        if stats.total_episodes > 0:
            self.border_title = f"EPISODE HEALTH ─ {stats.total_episodes} episodes"
        self.refresh()

    def render(self) -> Text:
        """Render episode metrics."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        stats = self._snapshot.episode_stats
        result = Text()

        # Line 1: Length statistics
        result.append("Length   ", style="dim")
        result.append(f"μ:{stats.length_mean:.0f}", style="cyan")
        result.append(f" σ:{stats.length_std:.0f}", style="dim")
        if stats.length_min > 0 or stats.length_max > 0:
            result.append(f" [{stats.length_min}-{stats.length_max}]", style="dim")
        result.append("\n")

        # Line 2: Outcome rates
        result.append("Timeout  ", style="dim")
        timeout_style = self._rate_style(stats.timeout_rate, bad_high=True)
        result.append(f"{stats.timeout_rate:.0%}", style=timeout_style)

        result.append("  Success ", style="dim")
        success_style = self._rate_style(stats.success_rate, bad_high=False)
        result.append(f"{stats.success_rate:.0%}", style=success_style)

        result.append("  Early ", style="dim")
        early_style = self._rate_style(stats.early_termination_rate, bad_high=True)
        result.append(f"{stats.early_termination_rate:.0%}", style=early_style)
        result.append("\n")

        # Line 3: Steps-per-action efficiency + trend
        result.append("Steps/Op ", style="dim")
        result.append(f"Germ:{stats.steps_per_germinate:.0f}", style="dim")
        result.append(f" Foss:{stats.steps_per_fossilize:.0f}", style="dim")
        result.append(f" Prune:{stats.steps_per_prune:.0f}", style="dim")
        result.append("  ")

        # Trend indicator
        trend_map = {
            "improving": ("↗", "green"),
            "stable": ("→", "dim"),
            "declining": ("↘", "red"),
        }
        arrow, style = trend_map.get(stats.completion_trend, ("→", "dim"))
        result.append(arrow, style=style)

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
