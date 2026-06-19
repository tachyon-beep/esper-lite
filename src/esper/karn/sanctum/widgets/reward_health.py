"""Reward health monitoring panel for Sanctum TUI.

Displays DRL Expert recommended metrics:
- PBRS fraction of total reward (healthy: 10-40%)
- Anti-gaming penalty frequency (healthy: <5%)
- Value-fit health via robust value_nrmse (healthy: NRMSE < 1.0)
- Hypervolume indicator (should increase over training)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.widgets import Static

# Robust value-fit health threshold. value_nrmse = RMSE(residual) / std(returns); below 1.0 the
# value head's residual RMSE is smaller than the return spread, i.e. it explains some variance.
# The artifact-prone explained_variance is NOT used for the health verdict (EV-telemetry-robustness):
# a low-return-variance batch can crater EV while value_nrmse stays healthy.
EV_VALUE_NRMSE_HEALTHY_THRESHOLD = 1.0


@dataclass
class RewardHealthData:
    """Aggregated reward health metrics."""

    pbrs_fraction: float = 0.0  # |PBRS| / |total_reward|
    anti_gaming_trigger_rate: float = 0.0  # Fraction of steps with penalties
    ev_explained: float = 0.0  # Value function explained variance (DIAGNOSTIC display only)
    # Robust value-fit signal (lower is better). Default inf = no data yet (unhealthy).
    value_nrmse: float = float("inf")
    ev_low_return_variance: bool = False  # EV denominator ill-conditioned (diagnostic flag)
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
        """Value-fit is healthy when the robust value_nrmse is below threshold.

        Keys on value_nrmse, the scale-stabilized fit signal, NOT the artifact-prone
        explained_variance: a low-return-variance batch (ev_low_return_variance) can crater EV
        while the value head is fitting fine (low NRMSE). EV is carried for display only.
        """
        return self.value_nrmse < EV_VALUE_NRMSE_HEALTHY_THRESHOLD


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
        """Render health indicators as plain Text (single-line compact format)."""
        result = Text()

        # All metrics on one line: PBRS:25% Game:2% EV:0.72 HV:1.2
        pbrs_color = "green" if self._data.is_pbrs_healthy else "red"
        gaming_color = "green" if self._data.is_gaming_healthy else "red"
        ev_color = "green" if self._data.is_ev_healthy else "yellow"

        result.append("PBRS:", style="dim")
        result.append(f"{self._data.pbrs_fraction:.0%}", style=pbrs_color)
        result.append(" Game:", style="dim")
        result.append(f"{self._data.anti_gaming_trigger_rate:.0%}", style=gaming_color)
        result.append(" EV:", style="dim")
        result.append(f"{self._data.ev_explained:.2f}", style=ev_color)
        result.append(" HV:", style="dim")
        result.append(f"{self._data.hypervolume:.1f}", style="cyan")

        return result


__all__ = ["RewardHealthPanel", "RewardHealthData"]
