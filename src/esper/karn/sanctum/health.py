"""Widget-agnostic health classifiers shared across Sanctum surfaces.

Single source of truth for "is this metric critical?" so the always-on
AnomalyStrip and the per-tab health panels can never disagree. A panel
rendering "Critical" while the strip stays silent is exactly the off-tab
blind spot the 2026-07-05 tri-domain design review flagged: once the layout
became tabbed, the strip is the only surface that pulls the operator to a
fire on a tab they are not watching, so it must classify a condition the
same way the owning panel does.

The classifiers are PURE verdicts over whatever state they are handed. The
caller decides *whether* to consult one — e.g. the strip gates the
metric-derived criticals on ``tamiyo.ppo_data_received`` so a fresh
snapshot's all-zero value stats (which read as a collapse) do not paint the
strip red before the first PPO update. Gating lives in the caller, never in
the classifier, so the panel delegation stays behaviour-preserving.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from esper.leyline import DEFAULT_RATIO_EXPLOSION_THRESHOLD

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import TamiyoState

# --- Value-function health thresholds ---------------------------------------
# Verbatim from the original HealthStatusPanel._get_value_status; promoted to
# named module constants so the strip and the panel share one set of values.
_VALUE_COLLAPSE_RANGE = 0.1
_VALUE_COLLAPSE_STD = 0.01
_VALUE_MEAN_COV_FLOOR = 0.1
_VALUE_COV_CRITICAL = 3.0
_VALUE_COV_WARNING = 2.0
_VALUE_INITIAL_SPREAD_FLOOR = 0.1
_VALUE_RELATIVE_CRITICAL = 10.0
_VALUE_RELATIVE_WARNING = 5.0
_VALUE_RANGE_EXPLOSION_CRITICAL = 1000.0
_VALUE_ABS_EXPLOSION_CRITICAL = 10000.0
_VALUE_RANGE_EXPLOSION_WARNING = 500.0
_VALUE_ABS_EXPLOSION_WARNING = 5000.0


def classify_value_health(tamiyo: "TamiyoState") -> str:
    """Return ``"OK"`` | ``"Warning"`` | ``"Critical"`` for the critic's value range.

    Catches value collapse (range ~= 0 held constant), relative instability
    (coefficient of variation), a large excursion relative to the warmup
    spread, and outright value explosion. Pure verdict: an empty snapshot's
    all-zero stats return "Critical" (range 0, std 0), so the caller must gate
    on ``ppo_data_received`` before treating that as a real collapse.
    """
    v_range = tamiyo.value_max - tamiyo.value_min
    v_mean = tamiyo.value_mean
    v_std = tamiyo.value_std
    initial = tamiyo.initial_value_spread

    # Collapse detection: values stuck at a constant.
    if v_range < _VALUE_COLLAPSE_RANGE and v_std < _VALUE_COLLAPSE_STD:
        return "Critical"

    # Coefficient of variation check (relative instability).
    if abs(v_mean) > _VALUE_MEAN_COV_FLOOR:
        cov = v_std / abs(v_mean)
        if cov > _VALUE_COV_CRITICAL:
            return "Critical"
        if cov > _VALUE_COV_WARNING:
            return "Warning"

    # Relative threshold (if the warmup spread is known).
    if initial is not None and initial > _VALUE_INITIAL_SPREAD_FLOOR:
        ratio = v_range / initial
        if ratio > _VALUE_RELATIVE_CRITICAL:
            return "Critical"
        if ratio > _VALUE_RELATIVE_WARNING:
            return "Warning"
        return "OK"

    # Absolute fallback (during warmup or if the initial spread is unknown).
    if v_range > _VALUE_RANGE_EXPLOSION_CRITICAL or abs(tamiyo.value_max) > _VALUE_ABS_EXPLOSION_CRITICAL:
        return "Critical"
    if v_range > _VALUE_RANGE_EXPLOSION_WARNING or abs(tamiyo.value_max) > _VALUE_ABS_EXPLOSION_WARNING:
        return "Warning"

    return "OK"


def ev_is_critical(tamiyo: "TamiyoState") -> bool:
    """Explained variance <= 0 means the critic explains nothing about returns.

    Suppressed when ``ev_low_return_variance`` is set: a floored EV denominator
    makes the ratio meaningless (an ill-conditioned denominator, not a critic
    failure), so surfacing it as a critic collapse would be a false alarm.
    Pure predicate; the caller gates on ``ppo_data_received`` (a fresh
    snapshot's EV defaults to 0.0, which would otherwise read as critical).
    """
    if tamiyo.ev_low_return_variance:
        return False
    return tamiyo.explained_variance <= 0.0


def ratio_is_critical(tamiyo: "TamiyoState") -> bool:
    """Joint PPO ratio (product across the factored heads) exceeds the explosion
    threshold — an oversized update even when each per-head ratio looks fine."""
    return tamiyo.joint_ratio_max > DEFAULT_RATIO_EXPLOSION_THRESHOLD
