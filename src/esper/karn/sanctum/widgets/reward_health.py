"""Reward health monitoring panel for Sanctum TUI.

Displays DRL Expert recommended metrics:
- PBRS fraction of total reward (healthy: 10-40%)
- Anti-gaming penalty frequency (healthy: <5%)
- Value function explained variance (healthy: >0.5)
- Hypervolume indicator (should increase over training)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.widgets import Static


@dataclass
class RewardHealthData:
    """Aggregated reward health metrics."""

    pbrs_fraction: float = 0.0  # |PBRS| / |total_reward|
    anti_gaming_trigger_rate: float = 0.0  # Fraction of steps with penalties
    ev_explained: float = 0.0  # Value function explained variance
    hypervolume: float = 0.0  # Pareto hypervolume indicator

    @property
    def is_pbrs_healthy(self) -> bool:
        """PBRS should be 10-40% of total reward (DRL Expert recommendation)."""
        return 0.1 <= self.pbrs_fraction <= 0.4

    @property
    def is_gaming_healthy(self) -> bool:
        """Anti-gaming penalties should trigger <5% of steps."""
        return self.anti_gaming_trigger_rate < 0.05

    @property
    def is_ev_healthy(self) -> bool:
        """Explained variance should be >0.5."""
        return self.ev_explained > 0.5


class RewardHealthPanel(Static):
    """Compact reward health display for Sanctum.

    Uses Textual CSS for border/title instead of Rich Panel to avoid
    double-nesting appearance.
    """

    def __init__(self, data: RewardHealthData | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._data = data or RewardHealthData()
        self.border_title = "REWARD HEALTH"

    def update_data(self, data: RewardHealthData) -> None:
        """Update health data and refresh display."""
        self._data = data
        self.refresh()

    def render(self) -> Text:
        """Render health indicators as plain Text."""
        result = Text()

        # PBRS fraction
        pbrs_color = "green" if self._data.is_pbrs_healthy else "red"
        result.append("PBRS: ", style="bold")
        result.append(f"{self._data.pbrs_fraction:.0%}", style=pbrs_color)
        result.append(" (10-40%)", style="dim")
        result.append("\n")

        # Anti-gaming
        gaming_color = "green" if self._data.is_gaming_healthy else "red"
        result.append("Gaming: ", style="bold")
        result.append(f"{self._data.anti_gaming_trigger_rate:.1%}", style=gaming_color)
        result.append(" (<5%)", style="dim")
        result.append("\n")

        # Explained variance
        ev_color = "green" if self._data.is_ev_healthy else "yellow"
        result.append("EV: ", style="bold")
        result.append(f"{self._data.ev_explained:.2f}", style=ev_color)
        result.append(" (>0.5)", style="dim")
        result.append("\n")

        # Hypervolume
        result.append("HV: ", style="bold")
        result.append(f"{self._data.hypervolume:.1f}", style="cyan")
        result.append("\n")  # Extra line for visual balance

        return result


__all__ = ["RewardHealthPanel", "RewardHealthData"]
