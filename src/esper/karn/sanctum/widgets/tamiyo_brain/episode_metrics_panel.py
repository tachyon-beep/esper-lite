"""EpisodeMetricsPanel - Inner loop training metrics.

Border title shows: "INNER LOOP ─ 128 ep @ 2.3/s" (episode count + throughput)

Displays (Training mode - DRL diagnostic metrics per expert review):
- Columnar display: Entropy, Yield, Slots (policy health indicators)
- Interpretation: Human-readable diagnosis based on metric patterns
- Steps-per-action metrics (germ, prune, foss)
- Trend indicator

Displays (Warmup mode - before PPO updates):
- Collection progress
- Random policy baseline returns
- Episode count

Per DRL expert: These replace useless Length/Outcomes for fixed-length episodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EpisodeStats, SanctumSnapshot


class EpisodeMetricsPanel(Static):
    """Episode-level metrics panel with warmup/training mode switching.

    Shows warmup diagnostics before PPO updates start,
    then switches to DRL diagnostic metrics once PPO data arrives.
    """

    # Column width for label alignment
    COL1 = 13

    # Thresholds for metric interpretation
    ENTROPY_HIGH = 0.8  # Random policy
    ENTROPY_LOW = 0.15  # Collapsed policy
    YIELD_LOW = 0.2  # Thrashing
    YIELD_HIGH = 0.9  # Too conservative
    SLOTS_LOW = 0.2  # WAIT spam
    SLOTS_HIGH = 0.95  # Germinate spam

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "INNER LOOP"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot

        # Update border title based on mode
        if not snapshot.tamiyo.ppo_data_received:
            self.border_title = "WARMUP"
        else:
            stats = snapshot.episode_stats
            if stats.total_episodes > 0:
                # Show episode count and throughput in title
                eps = stats.episodes_per_second
                if eps >= 1.0:
                    self.border_title = f"INNER LOOP ─ {stats.total_episodes} ep @ {eps:.1f}/s"
                elif eps > 0:
                    self.border_title = f"INNER LOOP ─ {stats.total_episodes} ep @ {eps:.2f}/s"
                else:
                    self.border_title = f"INNER LOOP ─ {stats.total_episodes} ep"
            else:
                self.border_title = "INNER LOOP"
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

        Layout (5 lines):
            Status       Collecting warmup data...
            Baseline     μ:+12.3  σ:8.2
            Episodes     n:45 collected
            (waiting for PPO updates...)
        """
        assert self._snapshot is not None  # Guarded by render()
        result = Text()
        tamiyo = self._snapshot.tamiyo
        history = list(tamiyo.episode_return_history)

        # Line 1: Status
        result.append("Status".ljust(self.COL1), style="dim")
        result.append("Collecting warmup data...", style="cyan")
        result.append("\n")

        # Line 2: Random policy baseline
        result.append("Baseline".ljust(self.COL1), style="dim")
        if history:
            h_mean = sum(history) / len(history)
            h_std = (sum((x - h_mean) ** 2 for x in history) / len(history)) ** 0.5
            mean_style = "green" if h_mean >= 0 else "red"
            result.append(f"μ:{h_mean:+.1f}", style=mean_style)
            result.append(f"  σ:{h_std:.1f}", style="cyan")
        else:
            result.append("---  (waiting)", style="dim")
        result.append("\n")

        # Line 3: Episode count
        result.append("Episodes".ljust(self.COL1), style="dim")
        if history:
            result.append(f"n:{len(history)}", style="cyan")
            result.append(" collected", style="dim")
        else:
            result.append("0", style="dim")
        result.append("\n")

        # Line 4: Waiting message
        result.append("\n")
        result.append("(waiting for PPO updates...)", style="dim")

        return result

    def _render_training(self) -> Text:
        """Render training mode DRL diagnostic metrics.

        Layout (5 lines):
            Entropy      Yield        Slots
            0.42         58%          62%
            → Healthy - policy converging
            Steps/Act    germ:12  prune:45  foss:89
            Trend        improving ↗
        """
        assert self._snapshot is not None  # Guarded by render()
        stats = self._snapshot.episode_stats
        result = Text()

        # Column width for metric display (13 chars each)
        COL_W = 13

        # Line 1: Column headers
        result.append("Entropy".ljust(COL_W), style="dim")
        result.append("Yield".ljust(COL_W), style="dim")
        result.append("Slots", style="dim")
        result.append("\n")

        # Line 2: Values in columns
        entropy = stats.action_entropy
        entropy_style = self._entropy_style(entropy)
        result.append(f"{entropy:.2f}".ljust(COL_W), style=entropy_style)

        yield_rate = stats.yield_rate
        yield_style = self._yield_style(yield_rate)
        result.append(f"{yield_rate:.0%}".ljust(COL_W), style=yield_style)

        slot_util = stats.slot_utilization
        slot_style = self._slot_util_style(slot_util)
        result.append(f"{slot_util:.0%}", style=slot_style)
        result.append("\n")

        # Line 3: Interpretation - human-readable diagnosis
        interpretation, interp_style = self._interpret_metrics(stats)
        result.append("→ ", style="dim")
        result.append(interpretation, style=interp_style)
        result.append("\n")

        # Line 4: Steps per action type
        result.append("Steps/Act".ljust(self.COL1), style="dim")
        result.append("germ:", style="dim")
        result.append(f"{stats.steps_per_germinate:.0f}".ljust(5), style="cyan")
        result.append("prune:", style="dim")
        result.append(f"{stats.steps_per_prune:.0f}".ljust(5), style="cyan")
        result.append("foss:", style="dim")
        result.append(f"{stats.steps_per_fossilize:.0f}", style="cyan")
        result.append("\n")

        # Line 5: Trend indicator
        result.append("Trend".ljust(self.COL1), style="dim")
        trend_map = {
            "improving": ("improving ↗", "green"),
            "stable": ("stable →", "dim"),
            "declining": ("declining ↘", "red"),
        }
        trend_text, trend_style = trend_map.get(
            stats.completion_trend, ("stable →", "dim")
        )
        result.append(trend_text, style=trend_style)

        return result

    def _entropy_style(self, entropy: float) -> str:
        """Style for entropy: 0.3-0.5 is healthy (green)."""
        if entropy > self.ENTROPY_HIGH:
            return "red"  # Random policy
        if entropy < self.ENTROPY_LOW:
            return "red bold"  # Collapsed
        if entropy < 0.3:
            return "yellow"  # Getting sharp (watch for collapse)
        return "green"  # Healthy range

    def _yield_style(self, yield_rate: float) -> str:
        """Style for yield rate: 0.4-0.7 is healthy."""
        if yield_rate < self.YIELD_LOW:
            return "red"  # Thrashing
        if yield_rate > self.YIELD_HIGH:
            return "yellow"  # Too conservative
        return "green"

    def _slot_util_style(self, slot_util: float) -> str:
        """Style for slot utilization: 0.4-0.8 is healthy."""
        if slot_util < self.SLOTS_LOW:
            return "red"  # WAIT spam
        if slot_util > self.SLOTS_HIGH:
            return "yellow"  # Germinate spam
        return "green"

    def _interpret_metrics(self, stats: "EpisodeStats") -> tuple[str, str]:
        """Generate human-readable interpretation from metric patterns.

        Returns (message, style) tuple.
        """
        entropy = stats.action_entropy
        yield_rate = stats.yield_rate
        slot_util = stats.slot_utilization

        # Priority 1: Detect degenerate policies
        if entropy < self.ENTROPY_LOW and slot_util < self.SLOTS_LOW:
            return "WAIT spam - policy collapsed", "red bold"

        if entropy < self.ENTROPY_LOW and slot_util > self.SLOTS_HIGH:
            return "Germinate spam - degenerate", "red bold"

        if entropy > self.ENTROPY_HIGH:
            return "Random policy - not learning", "yellow"

        # Priority 2: Efficiency issues
        if yield_rate < self.YIELD_LOW and slot_util > 0.3:
            return "Thrashing - seeds pruned before contributing", "red"

        if slot_util < self.SLOTS_LOW:
            return "Under-utilizing slots - too passive", "yellow"

        # Priority 3: Healthy patterns
        if entropy < 0.5 and yield_rate > 0.4 and 0.3 < slot_util < 0.9:
            return "Healthy - policy converging with good yield", "green"

        if entropy < 0.3 and yield_rate > 0.6:
            return "Converged - efficient policy", "green bold"

        # Default: still learning
        return "Learning - policy sharpening", "cyan"
