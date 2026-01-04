"""Shared trend indicators for TamiyoBrain panels.

Centralizes the mapping from RL-aware trend labels to compact on-screen glyphs.
All Tamiyo subpanels should use this to avoid conflicting "arrow semantics".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from esper.karn.sanctum.schema import detect_trend

if TYPE_CHECKING:
    from collections import deque


def trend_arrow_for_history(
    history: "list[float] | deque[float] | None",
    *,
    metric_name: str,
    metric_type: str,
) -> tuple[str, str]:
    """Return (arrow, style) describing trend health for a metric.

    Semantics are *health-first*:
    - ↑ green: improving
    - → dim: stable
    - ~ yellow: volatile
    - ↓ red: warning

    Args:
        history: Recent metric values (oldest first).
        metric_name: Name for threshold lookup (e.g., "policy_loss", "episode_return").
        metric_type: "loss" (lower=better) or "accuracy" (higher=better).
    """
    if history is None or len(history) < 5:
        return "", "dim"

    trend = detect_trend(list(history), metric_name, metric_type)

    arrows: dict[str, tuple[str, str]] = {
        "improving": ("↑", "green"),
        "stable": ("→", "dim"),
        "volatile": ("~", "yellow"),
        "warning": ("↓", "red"),
    }
    return arrows[trend]

