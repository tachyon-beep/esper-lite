"""AnomalyStrip widget - surfaces problems automatically.

Shows a single-line summary of all anomalies detected across the system.
When everything is OK, displays "ALL CLEAR" in green.
When problems exist, displays counts with color-coded severity.

Layout:
  ANOMALIES: 2 envs stalled | 1 seed exploding | PPO entropy low | MEM 95%
  -- or --
  ALL CLEAR ✓
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class AnomalyStrip(Static):
    """Single-line anomaly summary widget.

    Surfaces problems automatically so operators don't need to scan.
    Red items are critical, yellow are warnings, green means OK.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        # Computed anomaly counts
        self.stalled_count: int = 0
        self.degraded_count: int = 0
        self.gradient_issues: int = 0
        self.ppo_issues: bool = False
        self.memory_alarm: bool = False

    @property
    def has_anomalies(self) -> bool:
        """True if any anomaly is detected."""
        return (
            self.stalled_count > 0
            or self.degraded_count > 0
            or self.gradient_issues > 0
            or self.ppo_issues
            or self.memory_alarm
        )

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self._compute_anomalies()
        self.refresh()

    def _compute_anomalies(self) -> None:
        """Compute all anomaly counts from snapshot."""
        if self._snapshot is None:
            return

        # Reset counts
        self.stalled_count = 0
        self.degraded_count = 0
        self.gradient_issues = 0
        self.ppo_issues = False
        self.memory_alarm = False

        # Count env status issues
        for env in self._snapshot.envs.values():
            if env.status == "stalled":
                self.stalled_count += 1
            elif env.status == "degraded":
                self.degraded_count += 1

            # Count gradient issues across all seeds
            for seed in env.seeds.values():
                if seed.has_exploding or seed.has_vanishing:
                    self.gradient_issues += 1

        # Check PPO health
        tamiyo = self._snapshot.tamiyo
        if tamiyo.entropy_collapsed:
            self.ppo_issues = True
        # High KL divergence (>0.05) is also a warning
        if tamiyo.kl_divergence > 0.05:
            self.ppo_issues = True

        # Check memory pressure
        self.memory_alarm = self._snapshot.vitals.has_memory_alarm

    def render(self) -> Text:
        """Render the anomaly strip."""
        if self._snapshot is None:
            return Text("Waiting for data...", style="dim")

        if not self.has_anomalies:
            return Text("ALL CLEAR ✓", style="bold green")

        # Build anomaly summary
        parts = []

        if self.stalled_count > 0:
            parts.append(("stalled", self.stalled_count, "yellow"))
        if self.degraded_count > 0:
            parts.append(("degraded", self.degraded_count, "red"))
        if self.gradient_issues > 0:
            label = "grad issue" if self.gradient_issues == 1 else "grad issues"
            parts.append((label, self.gradient_issues, "red"))
        if self.ppo_issues:
            parts.append(("PPO", None, "yellow"))
        if self.memory_alarm:
            parts.append(("MEM", None, "red"))

        result = Text()
        result.append("ANOMALIES: ", style="bold red")

        for i, (label, count, color) in enumerate(parts):
            if i > 0:
                result.append(" | ", style="dim")
            if count is not None:
                result.append(f"{count} {label}", style=color)
            else:
                result.append(f"{label} ⚠", style=color)

        return result
