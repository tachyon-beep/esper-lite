"""Reward health monitoring panel for Sanctum TUI.

Displays DRL Expert recommended metrics:
- PBRS fraction of total reward (healthy: 10-40%)
- Anti-gaming penalty frequency (healthy: <5%)
- Value function explained variance (healthy: >0.5)
- Hypervolume indicator (should increase over training)
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.panel import Panel
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
    """Compact reward health display for Sanctum."""

    DEFAULT_CSS = """
    RewardHealthPanel {
        height: 6;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def __init__(self, data: RewardHealthData | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data or RewardHealthData()

    def update_data(self, data: RewardHealthData) -> None:
        """Update health data and refresh display."""
        self._data = data
        self.refresh()

    def render(self) -> Panel:
        """Render health indicators."""
        lines = []

        # PBRS fraction
        pbrs_color = "green" if self._data.is_pbrs_healthy else "red"
        lines.append(
            Text.assemble(
                ("PBRS: ", "bold"),
                (f"{self._data.pbrs_fraction:.0%}", pbrs_color),
                (" (10-40%)", "dim"),
            )
        )

        # Anti-gaming
        gaming_color = "green" if self._data.is_gaming_healthy else "red"
        lines.append(
            Text.assemble(
                ("Gaming: ", "bold"),
                (f"{self._data.anti_gaming_trigger_rate:.1%}", gaming_color),
                (" (<5%)", "dim"),
            )
        )

        # Explained variance
        ev_color = "green" if self._data.is_ev_healthy else "yellow"
        lines.append(
            Text.assemble(
                ("EV: ", "bold"),
                (f"{self._data.ev_explained:.2f}", ev_color),
                (" (>0.5)", "dim"),
            )
        )

        # Hypervolume
        lines.append(
            Text.assemble(
                ("HV: ", "bold"),
                (f"{self._data.hypervolume:.1f}", "cyan"),
            )
        )

        content = Text("\n").join(lines)
        return Panel(content, title="[bold]Reward Health[/]", border_style="blue")


__all__ = ["RewardHealthPanel", "RewardHealthData"]
