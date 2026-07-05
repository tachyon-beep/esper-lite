"""AnomalyStrip widget - the always-on triage spine.

A single line under the run header that surfaces problems automatically, so an
operator focused on one tab is still pulled to a fire on another. Once the
layout became tabbed this strip is the ONLY global-alert surface, so it must
cover every critical condition the per-tab panels can show — not just env/PPO
counts, but the critic collapsing (EV<=0, value collapse/explosion), the trust
region exploding (joint ratio), and the loudest safety event of all, a governor
panic rollback.

It also aggregates across ALL policy legs (A/B), not just the primary one the
other tabs show: a critical unique to leg B must still fire the strip. When two
or more legs are present each anomaly is attributed to its leg, e.g. "EV≤0 (B)".

Two severities drive both the text and the background:
  - CRITICAL paints the strip error-red (something is on fire).
  - WARNING gets a muted tint (worth a look, not an alarm) — keeping warnings
    off the error-red background is what stops alarm fatigue.

Layout:
  ANOMALIES: GOV ROLLBACK x2 [nan, divergence] (A) | EV≤0 (B) | 1 seed grad (A)
  -- or --
  WARNINGS: 2 stalled | PPO
  -- or --
  ALL CLEAR ✓
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class _LegAnomalies:
    """One policy leg's computed anomaly state (a pure per-snapshot verdict)."""

    group_id: str | None = None
    stalled_count: int = 0
    degraded_count: int = 0
    gradient_issues: int = 0
    ppo_issues: bool = False
    memory_alarm: bool = False
    dead_layers: int = 0
    exploding_layers: int = 0
    nan_grad_count: int = 0
    ev_critical: bool = False
    value_critical: bool = False
    value_warning: bool = False
    value_label: str = "value"
    ratio_explosion: bool = False
    governor_rollback_count: int = 0
    governor_rollback_reasons: list[str] = field(default_factory=list)
    rollback_unattributed: bool = False
    residual_breach: bool = False
    residual_share_value: float = 0.0
    gate_tripped: bool = False
    gate_cf_share: float = 0.0

    @property
    def critical(self) -> bool:
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
    def warning(self) -> bool:
        return (
            self.stalled_count > 0
            or self.dead_layers > 0
            or self.ppo_issues
            or self.value_warning
            or self.rollback_unattributed
        )

    def parts(self) -> list[tuple[str, int | None, str]]:
        """Non-governor (label, count, color) parts — criticals then warnings."""
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


def _compute_leg(snapshot: "SanctumSnapshot", group_id: str | None) -> _LegAnomalies:
    """Compute one leg's anomaly state from its snapshot (pure)."""
    leg = _LegAnomalies(group_id=group_id)

    # Env status + per-seed gradient issues + governor rollbacks.
    seen_reasons: list[str] = []
    for env in snapshot.envs.values():
        if env.status == "stalled":
            leg.stalled_count += 1
        elif env.status == "degraded":
            leg.degraded_count += 1

        for seed in env.seeds.values():
            if seed.has_exploding or seed.has_vanishing:
                leg.gradient_issues += 1

        if env.rolled_back:
            leg.governor_rollback_count += 1
            reason = env.rollback_reason or "?"
            if reason not in seen_reasons:
                seen_reasons.append(reason)
    leg.governor_rollback_reasons = seen_reasons

    tamiyo = snapshot.tamiyo

    # PPO health (entropy_collapsed / KL are presence-safe defaults; no data gate
    # needed — entropy_collapsed is a latched bool).
    if tamiyo.entropy_collapsed:
        leg.ppo_issues = True
    if tamiyo.kl_divergence > 0.05:
        leg.ppo_issues = True

    # Network-level gradient health (from the PPO update).
    leg.dead_layers = tamiyo.dead_layers
    leg.exploding_layers = tamiyo.exploding_layers
    leg.nan_grad_count = tamiyo.nan_grad_count

    # Memory pressure.
    leg.memory_alarm = snapshot.vitals.has_memory_alarm

    # Critic / trust-region criticals — gated on real PPO data so a fresh
    # snapshot's zero-valued stats (which read as EV<=0 + a value collapse) do
    # not paint the strip red before the first update.
    if tamiyo.ppo_data_received:
        leg.ev_critical = ev_is_critical(tamiyo)
        leg.ratio_explosion = ratio_is_critical(tamiyo)
        verdict = classify_value_health(tamiyo)
        leg.value_critical = verdict == "Critical"
        leg.value_warning = verdict == "Warning"
        if verdict != "OK":
            leg.value_label = _describe_value(tamiyo)

    # Governor rollback-attribution starvation (per-rollout aggregate; a
    # presence-safe default of 0, so no data gate).
    leg.rollback_unattributed = tamiyo.rollback_unattributed_count > 0

    # EV-stab Stage-0 telemetry (None = leg off; never alarm on absence).
    residual = tamiyo.return_var_residual_share
    if residual is not None and abs(residual) > RETURN_VAR_RESIDUAL_ALARM_SHARE:
        leg.residual_breach = True
        leg.residual_share_value = residual

    cf_share = tamiyo.return_var_cf_share
    if cf_share is not None and cf_share > RETURN_VAR_CF_SHARE_GATE:
        leg.gate_tripped = True
        leg.gate_cf_share = cf_share

    return leg


def _describe_value(tamiyo: "TamiyoState") -> str:
    """Name the value-health failure mode for the strip label."""
    v_range = abs(tamiyo.value_max - tamiyo.value_min)
    if v_range < 0.1:
        return "value collapse"
    if v_range > 500:
        return "value explosion"
    return "value unstable"


class AnomalyStrip(Static):
    """Single-line anomaly summary widget with critical/warning severity."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._legs: list[_LegAnomalies] = []
        self._primary_group_id: str | None = None

    # --- aggregate views (sum counts / any bools across legs) ---------------
    # Kept as the widget's public introspection surface; with a single leg they
    # equal that leg's values.

    @property
    def stalled_count(self) -> int:
        return sum(leg.stalled_count for leg in self._legs)

    @property
    def degraded_count(self) -> int:
        return sum(leg.degraded_count for leg in self._legs)

    @property
    def gradient_issues(self) -> int:
        return sum(leg.gradient_issues for leg in self._legs)

    @property
    def nan_grad_count(self) -> int:
        return sum(leg.nan_grad_count for leg in self._legs)

    @property
    def exploding_layers(self) -> int:
        return sum(leg.exploding_layers for leg in self._legs)

    @property
    def dead_layers(self) -> int:
        return sum(leg.dead_layers for leg in self._legs)

    @property
    def governor_rollback_count(self) -> int:
        return sum(leg.governor_rollback_count for leg in self._legs)

    @property
    def ppo_issues(self) -> bool:
        return any(leg.ppo_issues for leg in self._legs)

    @property
    def memory_alarm(self) -> bool:
        return any(leg.memory_alarm for leg in self._legs)

    @property
    def ev_critical(self) -> bool:
        return any(leg.ev_critical for leg in self._legs)

    @property
    def value_critical(self) -> bool:
        return any(leg.value_critical for leg in self._legs)

    @property
    def value_warning(self) -> bool:
        return any(leg.value_warning for leg in self._legs)

    @property
    def ratio_explosion(self) -> bool:
        return any(leg.ratio_explosion for leg in self._legs)

    @property
    def rollback_unattributed(self) -> bool:
        return any(leg.rollback_unattributed for leg in self._legs)

    @property
    def residual_breach(self) -> bool:
        return any(leg.residual_breach for leg in self._legs)

    @property
    def has_critical(self) -> bool:
        return any(leg.critical for leg in self._legs)

    @property
    def has_warning(self) -> bool:
        return any(leg.warning for leg in self._legs)

    @property
    def has_anomalies(self) -> bool:
        return self.has_critical or self.has_warning

    # --- entry points -------------------------------------------------------

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update from a single snapshot (no leg attribution)."""
        self._legs = [_compute_leg(snapshot, group_id=None)]
        self._primary_group_id = None
        self._apply_severity_class()
        self.refresh()

    def update_snapshots(
        self,
        snapshots_by_group: dict[str, "SanctumSnapshot"],
        primary_group_id: str | None = None,
    ) -> None:
        """Update from every policy leg, aggregating anomalies across all of them."""
        ordered = self._order_groups(snapshots_by_group, primary_group_id)
        self._legs = [
            _compute_leg(snapshots_by_group[gid], group_id=gid) for gid in ordered
        ]
        self._primary_group_id = primary_group_id
        self._apply_severity_class()
        self.refresh()

    @staticmethod
    def _order_groups(
        snapshots_by_group: dict[str, "SanctumSnapshot"],
        primary_group_id: str | None,
    ) -> list[str]:
        """Stable leg order: the primary leg first, then the rest sorted."""
        ordered = sorted(snapshots_by_group.keys())
        if primary_group_id in ordered:
            ordered.remove(primary_group_id)
            ordered.insert(0, primary_group_id)
        return ordered

    def _apply_severity_class(self) -> None:
        """Background reflects the worst severity across all legs; critical wins."""
        self.remove_class("has-critical")
        self.remove_class("has-warning")
        if self.has_critical:
            self.add_class("has-critical")
        elif self.has_warning:
            self.add_class("has-warning")

    # --- render -------------------------------------------------------------

    def render(self) -> Text:
        """Render the anomaly strip."""
        if not self._legs:
            return Text("Waiting for data...", style="dim")

        if not self.has_anomalies:
            result = Text("ALL CLEAR ✓", style="bold green")
            self._append_gate_chip(result)
            return result

        multi = len(self._legs) > 1
        segments: list[Text] = []

        # Governor rollbacks across ALL legs lead — the loudest safety event,
        # never buried behind env counts or truncated off a full line.
        for leg in self._legs:
            if leg.governor_rollback_count > 0:
                reasons = ", ".join(leg.governor_rollback_reasons) or "?"
                tag = f" ({leg.group_id})" if multi and leg.group_id else ""
                segments.append(
                    Text(
                        f"GOV ROLLBACK ×{leg.governor_rollback_count} [{reasons}]{tag}",
                        style="bold red",
                    )
                )

        for leg in self._legs:
            tag = f" ({leg.group_id})" if multi and leg.group_id else ""
            for label, count, color in leg.parts():
                if count is not None:
                    text = f"{count} {label}{tag}"
                else:
                    text = f"{label}{tag} ⚠"
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

    def _primary_leg(self) -> _LegAnomalies | None:
        """The leg driving the other tabs (for the informational gate chip)."""
        if not self._legs:
            return None
        for leg in self._legs:
            if leg.group_id == self._primary_group_id:
                return leg
        return self._legs[0]

    def _append_gate_chip(self, result: Text) -> None:
        """Append the primary leg's Stage-0 gate info chip (diagnosis, not failure)."""
        leg = self._primary_leg()
        if leg is None or not leg.gate_tripped:
            return
        result.append("  ·  ", style="dim")
        result.append(
            f"S0 gate: cf {leg.gate_cf_share:.2f} ▲>{RETURN_VAR_CF_SHARE_GATE:.2f}",
            style="magenta",
        )
