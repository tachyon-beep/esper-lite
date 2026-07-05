"""AnomalyStrip widget - the always-on triage spine.

A single line under the run header that surfaces problems automatically, so an
operator focused on one tab is still pulled to a fire on another. Once the
layout became tabbed this strip is the ONLY global-alert surface, so it must
cover every critical condition the per-tab panels can show — not just env/PPO
counts, but the critic collapsing (EV<=0, value collapse/explosion), the trust
region exploding (joint ratio), and the loudest safety event of all, a governor
panic rollback.

Two severities drive both the text and the background:
  - CRITICAL paints the strip error-red (something is on fire).
  - WARNING gets a muted tint (worth a look, not an alarm) — keeping warnings
    off the error-red background is what stops alarm fatigue.

Layout:
  ANOMALIES: GOV ROLLBACK x2 [nan, divergence] | 1 seed grad | EV<=0 | MEM
  -- or --
  WARNINGS: 2 stalled | PPO
  -- or --
  ALL CLEAR ✓
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from esper.karn.sanctum.health import (
    classify_value_health,
    ev_is_critical,
    ratio_is_critical,
)
from esper.leyline import RETURN_VAR_CF_SHARE_GATE, RETURN_VAR_RESIDUAL_ALARM_SHARE

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState


class AnomalyStrip(Static):
    """Single-line anomaly summary widget with critical/warning severity."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        # Env/seed counts.
        self.stalled_count: int = 0
        self.degraded_count: int = 0
        self.gradient_issues: int = 0  # Per-seed gradient issues (vanishing/exploding)
        # PPO / network-level health.
        self.ppo_issues: bool = False
        self.memory_alarm: bool = False
        self.dead_layers: int = 0
        self.exploding_layers: int = 0
        self.nan_grad_count: int = 0
        # Critic / trust-region criticals (gated on ppo_data_received: a fresh
        # snapshot's zero-valued stats would otherwise read as EV<=0 + a value
        # collapse and paint the strip red before the first PPO update).
        self.ev_critical: bool = False
        self.value_critical: bool = False
        self.value_warning: bool = False
        self.value_label: str = "value"
        self.ratio_explosion: bool = False
        # Governor safety events (independent of the policy, so NOT gated on
        # ppo_data_received). rolled_back latches on rollback and clears on the
        # next epoch, so it is a real "currently rolled back" state.
        self.governor_rollback_count: int = 0
        self.governor_rollback_reasons: list[str] = []
        self.rollback_unattributed: bool = False
        # EV-stab Stage-0: residual-share breach = the additend decomposition is
        # leaking untracked reward mass (a real anomaly). The cf-share gate trip
        # is a DIAGNOSIS (de-shaping justified), rendered as a persistent info
        # chip that must never trip the red strip.
        self.residual_breach: bool = False
        self.residual_share_value: float = 0.0
        self.gate_tripped: bool = False
        self.gate_cf_share: float = 0.0

    @property
    def has_critical(self) -> bool:
        """True if any error-red condition is present."""
        return (
            self.degraded_count > 0
            or self.gradient_issues > 0
            or self.nan_grad_count > 0
            or self.exploding_layers > 0
            or self.memory_alarm
            or self.residual_breach
            or self.ev_critical
            or self.value_critical
            or self.ratio_explosion
            or self.governor_rollback_count > 0
        )

    @property
    def has_warning(self) -> bool:
        """True if any muted-warning condition is present."""
        return (
            self.stalled_count > 0
            or self.dead_layers > 0
            or self.ppo_issues
            or self.value_warning
            or self.rollback_unattributed
        )

    @property
    def has_anomalies(self) -> bool:
        """True if any anomaly (critical or warning) is detected."""
        return self.has_critical or self.has_warning

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self._compute_anomalies()

        # Background reflects the WORST severity present; critical wins so a
        # concurrent warning never downgrades the alarm.
        self.remove_class("has-critical")
        self.remove_class("has-warning")
        if self.has_critical:
            self.add_class("has-critical")
        elif self.has_warning:
            self.add_class("has-warning")

        self.refresh()

    def _compute_anomalies(self) -> None:
        """Compute all anomaly state from the snapshot."""
        if self._snapshot is None:
            return

        # Reset.
        self.stalled_count = 0
        self.degraded_count = 0
        self.gradient_issues = 0
        self.ppo_issues = False
        self.memory_alarm = False
        self.dead_layers = 0
        self.exploding_layers = 0
        self.nan_grad_count = 0
        self.ev_critical = False
        self.value_critical = False
        self.value_warning = False
        self.value_label = "value"
        self.ratio_explosion = False
        self.governor_rollback_count = 0
        self.governor_rollback_reasons = []
        self.rollback_unattributed = False

        # Env status + per-seed gradient issues + governor rollbacks.
        seen_reasons: list[str] = []
        for env in self._snapshot.envs.values():
            if env.status == "stalled":
                self.stalled_count += 1
            elif env.status == "degraded":
                self.degraded_count += 1

            for seed in env.seeds.values():
                if seed.has_exploding or seed.has_vanishing:
                    self.gradient_issues += 1

            if env.rolled_back:
                self.governor_rollback_count += 1
                reason = env.rollback_reason or "?"
                if reason not in seen_reasons:
                    seen_reasons.append(reason)
        self.governor_rollback_reasons = seen_reasons

        tamiyo = self._snapshot.tamiyo

        # PPO health (entropy_collapsed / KL are presence-safe defaults, so they
        # need no data gate — entropy_collapsed is a latched bool).
        if tamiyo.entropy_collapsed:
            self.ppo_issues = True
        if tamiyo.kl_divergence > 0.05:
            self.ppo_issues = True

        # Network-level gradient health (from the PPO update).
        self.dead_layers = tamiyo.dead_layers
        self.exploding_layers = tamiyo.exploding_layers
        self.nan_grad_count = tamiyo.nan_grad_count

        # Memory pressure.
        self.memory_alarm = self._snapshot.vitals.has_memory_alarm

        # Critic / trust-region criticals — gated on real PPO data.
        if tamiyo.ppo_data_received:
            self.ev_critical = ev_is_critical(tamiyo)
            self.ratio_explosion = ratio_is_critical(tamiyo)
            verdict = classify_value_health(tamiyo)
            self.value_critical = verdict == "Critical"
            self.value_warning = verdict == "Warning"
            if verdict != "OK":
                self.value_label = self._describe_value(tamiyo)

        # Governor rollback-attribution starvation (per-rollout aggregate; a
        # presence-safe default of 0, so it needs no data gate).
        self.rollback_unattributed = tamiyo.rollback_unattributed_count > 0

        # EV-stab Stage-0 telemetry (None = leg off; never alarm on absence).
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

    @staticmethod
    def _describe_value(tamiyo: "TamiyoState") -> str:
        """Name the value-health failure mode for the strip label."""
        v_range = abs(tamiyo.value_max - tamiyo.value_min)
        if v_range < 0.1:
            return "value collapse"
        if v_range > 500:
            return "value explosion"
        return "value unstable"

    def render(self) -> Text:
        """Render the anomaly strip."""
        if self._snapshot is None:
            return Text("Waiting for data...", style="dim")

        if not self.has_anomalies:
            result = Text("ALL CLEAR ✓", style="bold green")
            self._append_gate_chip(result)
            return result

        segments: list[Text] = []
        # Governor rollback leads — the loudest safety event, never buried
        # behind env counts or truncated off the end of a full line.
        if self.governor_rollback_count > 0:
            reasons = ", ".join(self.governor_rollback_reasons) or "?"
            segments.append(
                Text(
                    f"GOV ROLLBACK ×{self.governor_rollback_count} [{reasons}]",
                    style="bold red",
                )
            )
        for label, count, color in self._anomaly_parts():
            text = f"{count} {label}" if count is not None else f"{label} ⚠"
            segments.append(Text(text, style=color))

        result = Text()
        if self.has_critical:
            result.append("ANOMALIES: ", style="bold red")
        else:
            result.append("WARNINGS: ", style="bold yellow")
        for i, segment in enumerate(segments):
            if i > 0:
                result.append(" | ", style="dim")
            result.append_text(segment)

        self._append_gate_chip(result)
        return result

    def _anomaly_parts(self) -> list[tuple[str, int | None, str]]:
        """Ordered (label, count, color) parts — criticals first, then warnings."""
        parts: list[tuple[str, int | None, str]] = []

        # Criticals (red).
        if self.degraded_count > 0:
            parts.append(("degraded", self.degraded_count, "red"))
        if self.gradient_issues > 0:
            label = "seed grad" if self.gradient_issues == 1 else "seed grads"
            parts.append((label, self.gradient_issues, "red"))
        if self.nan_grad_count > 0:
            parts.append(("NaN grads", self.nan_grad_count, "red"))
        if self.exploding_layers > 0:
            parts.append(("exploding", self.exploding_layers, "red"))
        if self.ev_critical:
            parts.append(("EV≤0", None, "red"))
        if self.value_critical:
            parts.append((self.value_label, None, "red"))
        if self.ratio_explosion:
            parts.append(("RatioJnt explosion", None, "red"))
        if self.memory_alarm:
            parts.append(("MEM", None, "red"))
        if self.residual_breach:
            parts.append(
                (f"decomp resid {self.residual_share_value:+.2f}", None, "red")
            )

        # Warnings (yellow).
        if self.stalled_count > 0:
            parts.append(("stalled", self.stalled_count, "yellow"))
        if self.dead_layers > 0:
            parts.append(("dead layers", self.dead_layers, "yellow"))
        if self.ppo_issues:
            parts.append(("PPO", None, "yellow"))
        if self.value_warning:
            parts.append((self.value_label, None, "yellow"))
        if self.rollback_unattributed:
            parts.append(("rollback unattr", None, "yellow"))

        return parts

    def _append_gate_chip(self, result: Text) -> None:
        """Append the Stage-0 gate info chip (diagnosis, not a failure state)."""
        if not self.gate_tripped:
            return
        result.append("  ·  ", style="dim")
        result.append(
            f"S0 gate: cf {self.gate_cf_share:.2f} ▲>{RETURN_VAR_CF_SHARE_GATE:.2f}",
            style="magenta",
        )
