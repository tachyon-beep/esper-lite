"""Sparkline utilities for TamiyoBrain panels.

Provides sparkline rendering used by Tamiyo subpanels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:
    from collections import deque


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
