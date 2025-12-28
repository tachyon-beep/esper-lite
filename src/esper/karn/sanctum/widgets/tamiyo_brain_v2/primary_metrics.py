"""PrimaryMetrics - Episode Return and Entropy sparklines.

Displays the two most important RL metrics prominently at the top:
    Ep.Return  ▁▂▃▄▅▆▇█  -9.8 ↘    LR:1e-04  EntCoef:0.10
    Entropy    █▇▆▅▄▃▂▁   7.89 →
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

if TYPE_CHECKING:
    from collections import deque

    from esper.karn.sanctum.schema import SanctumSnapshot


# Sparkline block characters (8 levels)
SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"


def render_sparkline(
    values: list[float] | deque[float],
    width: int = 35,
    style: str = "bright_cyan",
) -> Text:
    """Render a sparkline using Unicode block characters.

    Args:
        values: Historical values to visualize
        width: Maximum width in characters
        style: Rich style for the blocks

    Returns:
        Text with sparkline or placeholder for empty data.
    """
    result = Text()

    if not values:
        result.append("─" * width, style="dim")
        return result

    data = list(values)[-width:]

    # Pad left if fewer values than width
    if len(data) < width:
        pad_count = width - len(data)
        result.append("─" * pad_count, style="dim")

    if not data:
        return result

    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val if max_val != min_val else 1

    for v in data:
        normalized = (v - min_val) / val_range
        idx = int(normalized * (len(SPARKLINE_BLOCKS) - 1))
        idx = max(0, min(len(SPARKLINE_BLOCKS) - 1, idx))
        result.append(SPARKLINE_BLOCKS[idx], style=style)

    return result


def detect_trend(values: list[float], window: int = 5) -> str:
    """Detect trend direction from recent values.

    Returns:
        "↗" (improving), "↘" (declining), or "→" (stable)
    """
    if len(values) < 2:
        return "→"

    recent = values[-window:] if len(values) >= window else values
    if len(recent) < 2:
        return "→"

    # Compare first half average to second half average
    mid = len(recent) // 2
    first_half = sum(recent[:mid]) / max(1, mid)
    second_half = sum(recent[mid:]) / max(1, len(recent) - mid)

    diff = second_half - first_half
    threshold = abs(first_half) * 0.05 if first_half != 0 else 0.01

    if diff > threshold:
        return "↗"
    elif diff < -threshold:
        return "↘"
    return "→"


def trend_style(trend: str, metric_type: str = "accuracy") -> str:
    """Get style for trend indicator.

    Args:
        trend: "↗", "↘", or "→"
        metric_type: "accuracy" (higher is better) or "loss" (lower is better)

    Returns:
        Rich style string
    """
    if metric_type == "accuracy":
        return {"↗": "green", "↘": "red", "→": "dim"}.get(trend, "dim")
    else:  # loss
        return {"↗": "red", "↘": "green", "→": "dim"}.get(trend, "dim")


class PrimaryMetrics(Container):
    """Primary metrics panel with Episode Return and Entropy sparklines."""

    SPARKLINE_WIDTH: ClassVar[int] = 35

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        """Compose the metrics display."""
        yield Static(id="ep-return-row", classes="metric-row")
        yield Static(id="entropy-row", classes="metric-row")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        tamiyo = snapshot.tamiyo

        # Episode Return row
        ep_return_widget = self.query_one("#ep-return-row", Static)
        ep_return_widget.update(self._render_episode_return(tamiyo))

        # Entropy row
        entropy_widget = self.query_one("#entropy-row", Static)
        entropy_widget.update(self._render_entropy(tamiyo))

    def _render_episode_return(self, tamiyo) -> Text:
        """Render Episode Return sparkline row."""
        result = Text()

        result.append("Ep.Return  ", style="bold cyan")

        if tamiyo.episode_return_history:
            sparkline = render_sparkline(
                tamiyo.episode_return_history,
                width=self.SPARKLINE_WIDTH,
            )
            result.append(sparkline)

            trend = detect_trend(list(tamiyo.episode_return_history))
            result.append(f"  {tamiyo.current_episode_return:>7.1f} ", style="white")
            result.append(trend, style=trend_style(trend, "accuracy"))

            # Hyperparameters
            if tamiyo.learning_rate:
                result.append(f"      LR:{tamiyo.learning_rate:.0e}", style="dim")
            result.append(f"  EntCoef:{tamiyo.entropy_coef:.2f}", style="dim")
        else:
            result.append("─" * self.SPARKLINE_WIDTH, style="dim")
            result.append("  (no data)", style="dim italic")

        return result

    def _render_entropy(self, tamiyo) -> Text:
        """Render Entropy sparkline row."""
        result = Text()

        result.append("Entropy    ", style="bold")

        if tamiyo.entropy_history:
            sparkline = render_sparkline(
                tamiyo.entropy_history,
                width=self.SPARKLINE_WIDTH,
            )
            result.append(sparkline)

            trend = detect_trend(list(tamiyo.entropy_history))
            result.append(f"  {tamiyo.entropy:>7.2f} ", style="white")
            # For entropy, stable is good, declining is bad (collapse)
            result.append(trend, style=trend_style(trend, "accuracy"))
        else:
            result.append("─" * self.SPARKLINE_WIDTH, style="dim")
            result.append("  (no data)", style="dim italic")

        return result
