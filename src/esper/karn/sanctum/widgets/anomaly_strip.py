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

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from esper.leyline import RETURN_VAR_CF_SHARE_GATE, RETURN_VAR_RESIDUAL_ALARM_SHARE

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class AnomalyStrip(Static):
    """Single-line anomaly summary widget.

    Surfaces problems automatically so operators don't need to scan.
    Red items are critical, yellow are warnings, green means OK.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        # Computed anomaly counts
        self.stalled_count: int = 0
        self.degraded_count: int = 0
        self.gradient_issues: int = 0  # Per-seed gradient issues (vanishing/exploding)
        self.ppo_issues: bool = False
        self.memory_alarm: bool = False
        # Network-level gradient health (from TamiyoState)
        self.dead_layers: int = 0
        self.exploding_layers: int = 0
        self.nan_grad_count: int = 0
        # EV-stab Stage-0: residual-share breach = the additend decomposition is
        # leaking untracked reward mass (a real anomaly). The cf-share gate trip is
        # a DIAGNOSIS (de-shaping justified), rendered as a persistent info chip
        # that must never trip the red strip.
        self.residual_breach: bool = False
        self.residual_share_value: float = 0.0
        self.gate_tripped: bool = False
        self.gate_cf_share: float = 0.0

    @property
    def has_anomalies(self) -> bool:
        """True if any anomaly is detected."""
        return (
            self.stalled_count > 0
            or self.degraded_count > 0
            or self.gradient_issues > 0
            or self.ppo_issues
            or self.memory_alarm
            or self.dead_layers > 0
            or self.exploding_layers > 0
            or self.nan_grad_count > 0
            or self.residual_breach
        )

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self._compute_anomalies()

        # Apply visual indicator when anomalies detected
        if self.has_anomalies:
            self.add_class("has-anomalies")
        else:
            self.remove_class("has-anomalies")

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
        self.dead_layers = 0
        self.exploding_layers = 0
        self.nan_grad_count = 0

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

        # Check network-level gradient health (from PPO update)
        self.dead_layers = tamiyo.dead_layers
        self.exploding_layers = tamiyo.exploding_layers
        self.nan_grad_count = tamiyo.nan_grad_count

        # Check memory pressure
        self.memory_alarm = self._snapshot.vitals.has_memory_alarm

        # EV-stab Stage-0 telemetry (None = leg off; never alarm on absence)
        self.residual_breach = False
        self.residual_share_value = 0.0
        residual = tamiyo.return_var_residual_share
        if residual is not None and abs(residual) > RETURN_VAR_RESIDUAL_ALARM_SHARE:
            self.residual_breach = True
            self.residual_share_value = residual

        self.gate_tripped = False
        self.gate_cf_share = 0.0
        cf_share = tamiyo.return_var_cf_share
        if cf_share is not None and cf_share > RETURN_VAR_CF_SHARE_GATE:
            self.gate_tripped = True
            self.gate_cf_share = cf_share

    def render(self) -> Text:
        """Render the anomaly strip."""
        if self._snapshot is None:
            return Text("Waiting for data...", style="dim")

        if not self.has_anomalies:
            result = Text("ALL CLEAR ✓", style="bold green")
            self._append_gate_chip(result)
            return result

        # Build anomaly summary
        parts: list[tuple[str, int | None, str]] = []

        if self.stalled_count > 0:
            parts.append(("stalled", self.stalled_count, "yellow"))
        if self.degraded_count > 0:
            parts.append(("degraded", self.degraded_count, "red"))
        if self.gradient_issues > 0:
            label = "seed grad" if self.gradient_issues == 1 else "seed grads"
            parts.append((label, self.gradient_issues, "red"))
        # Network-level gradient health (critical issues)
        if self.nan_grad_count > 0:
            parts.append(("NaN grads", self.nan_grad_count, "red"))
        if self.exploding_layers > 0:
            parts.append(("exploding", self.exploding_layers, "red"))
        if self.dead_layers > 0:
            parts.append(("dead layers", self.dead_layers, "yellow"))
        if self.ppo_issues:
            parts.append(("PPO", None, "yellow"))
        if self.memory_alarm:
            parts.append(("MEM", None, "red"))
        if self.residual_breach:
            parts.append(
                (f"decomp resid {self.residual_share_value:+.2f}", None, "red")
            )

        result = Text()
        result.append("ANOMALIES: ", style="bold red")

        for i, (label, count, color) in enumerate(parts):
            if i > 0:
                result.append(" | ", style="dim")
            if count is not None:
                result.append(f"{count} {label}", style=color)
            else:
                result.append(f"{label} ⚠", style=color)

        self._append_gate_chip(result)
        return result

    def _append_gate_chip(self, result: Text) -> None:
        """Append the Stage-0 gate info chip (diagnosis, not a failure state)."""
        if not self.gate_tripped:
            return
        result.append("  ·  ", style="dim")
        result.append(
            f"S0 gate: cf {self.gate_cf_share:.2f} ▲>{RETURN_VAR_CF_SHARE_GATE:.2f}",
            style="magenta",
        )
