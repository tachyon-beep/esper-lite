"""Non-selectable settlement: the pure computational core (Phase 2, DEFAULT-OFF).

Spec: docs/analysis/2026-07-14-permanence-visible-preregistration.md §2.2–§2.4 +
docs/plans/ready/2026-07-14-phase2-settlement-implementation.md §0/§3. Everything
here is pure math over explicit inputs so the executable replay can drive it
directly; trainer integration is a thin adapter (separate increment).

Key invariants owned here:
- The boundary is SCHEDULE-DETERMINED (fixed cadence, extend-on-scarce
  measurements) — the policy chooses whether, never when (READ-1).
- The boundary package is DISCOUNT-NEUTRAL both signs (÷γ^d): its present value
  at the request instant equals the shipping prior exactly (PDR-0083 #1).
- The premium is the EXPRESSION evaluated from the shipping constants, never a
  hand-rounded literal (PDR-0084 #2).
- T_annuity ≡ T_provisional at boundary-frozen arguments, ALL branches —
  including both ratio_penalty branches with the WINDOWED cf denominator
  (PDR-0096/0097: ransomware is a persistent, selectable property; the ratio
  check is the law's only counterfactual cross-check and stays in on both
  sides of the boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from esper.leyline import DEFAULT_GAMMA
from esper.leyline.fossil_settlement import FossilSettlementConfig

from .contribution import (
    ContributionRewardConfig,
    _compute_attributed_value,
    _compute_ratio_penalty,
    _compute_timing_discount,
)

SettlementBranch = Literal["prior", "noncontributing"]


def next_boundary(request_epoch: int, config: FossilSettlementConfig) -> int:
    """First fixed-cadence boundary >= request + min_window (pre-reg §2.2)."""
    earliest = request_epoch + config.min_window
    return math.ceil(earliest / config.w_settle) * config.w_settle


def boundary_with_extension(
    request_epoch: int,
    measured_epochs: frozenset[int] | set[int],
    config: FossilSettlementConfig,
    max_epochs: int,
) -> int | None:
    """The settling boundary under the zero-slack rule (pre-reg §2.3).

    Extends boundary-by-boundary until the window [request+1, B] contains
    >= min_measurements valid measurements. Returns None when no boundary
    <= max_epochs qualifies — the caller must treat that as a late request
    that should have been masked (fail loud, never settle-on-fewer).
    """
    boundary = next_boundary(request_epoch, config)
    while boundary <= max_epochs:
        n_valid = sum(1 for e in measured_epochs if request_epoch < e <= boundary)
        if n_valid >= config.min_measurements:
            return boundary
        boundary += config.w_settle
    return None


def ewma(values: list[float], span: int) -> float:
    """The frozen smoother: adjust=False recursion seeded at the first value."""
    if not values:
        raise ValueError("ewma of an empty series")
    alpha = 2.0 / (span + 1.0)
    state = values[0]
    for value in values[1:]:
        state = alpha * value + (1.0 - alpha) * state
    return state


def q_settle(
    window_measurements: list[float], config: FossilSettlementConfig
) -> float:
    """The payout-basis quote: EWMA over post-request measurements (§2.3)."""
    if len(window_measurements) < config.min_measurements:
        raise ValueError(
            f"q_settle requires >= min_measurements ({config.min_measurements}) "
            f"valid measurements, got {len(window_measurements)} — the zero-slack "
            "rule extends the boundary instead of settling on fewer"
        )
    return ewma(window_measurements, config.ewma_span)


def settlement_gate(
    window_measurements: list[float], config: FossilSettlementConfig
) -> bool:
    """The boundary qualification (F3: raw threshold or LCB form, per config)."""
    quote = q_settle(window_measurements, config)
    if config.gate_form == "raw":
        return quote >= config.q_threshold
    n = len(window_measurements)
    mean = sum(window_measurements) / n
    var = sum((m - mean) ** 2 for m in window_measurements) / (n - 1)
    lcb = quote - config.lcb_z * math.sqrt(var / n)
    return lcb >= config.q_threshold


def settlement_premium(reward_config: ContributionRewardConfig) -> float:
    """The matched-control flat prior, evaluated from the SHIPPING constants.

    Consolidates both shipping flat FOSSILIZE terms (PDR-0082/0083):
    fossilize_base_bonus + fossilize_terminal_scale * tanh(1/quality_ceiling)
    — with defaults, 0.5 + 3·tanh(1/3) ≈ 1.464538. Never retuned from results.
    """
    return reward_config.fossilize_base_bonus + (
        reward_config.fossilize_terminal_scale
        * math.tanh(1.0 / reward_config.fossilize_quality_ceiling)
    )


@dataclass(frozen=True)
class BoundaryPackage:
    """What the boundary epoch pays (before the annuity stream), pre-reg §2.4."""

    branch: SettlementBranch
    prior_paid: float  # discount-neutralized: value / gamma^d, either sign
    q_settle: float
    cf_windowed: float
    d: int  # request -> boundary delay (reward-phase to reward-phase)


def boundary_package(
    request_epoch: int,
    boundary_epoch: int,
    legitimacy_request: float,
    window_measurements: list[float],
    window_cf_measurements: list[float],
    reward_config: ContributionRewardConfig,
    settlement_config: FossilSettlementConfig,
    gamma: float = DEFAULT_GAMMA,
) -> BoundaryPackage:
    """The fixed boundary payment: prior (qualified) or the −0.2 branch, ÷γ^d both signs."""
    d = boundary_epoch - request_epoch
    if d < 1:
        raise ValueError("boundary must follow the request")
    quote = q_settle(window_measurements, settlement_config)
    cf_windowed = ewma(window_cf_measurements, settlement_config.ewma_span)
    if settlement_gate(window_measurements, settlement_config):
        branch: SettlementBranch = "prior"
        raw_value = settlement_premium(reward_config) * legitimacy_request
    else:
        branch = "noncontributing"
        raw_value = reward_config.fossilize_noncontributing_penalty
    return BoundaryPackage(
        branch=branch,
        prior_paid=raw_value / gamma**d,
        q_settle=quote,
        cf_windowed=cf_windowed,
        d=d,
    )


def annuity_per_epoch(
    q_settle_value: float,
    cf_windowed: float,
    progress_at_boundary: float | None,
    germination_epoch: int,
    reward_config: ContributionRewardConfig,
) -> float:
    """T_provisional at boundary-frozen arguments — ALL branches (PDR-0096).

    Mirrors the non-escrow attribution law in compute_contribution_reward at
    (c := q_settle, cf := cf_windowed, progress := progress_at_boundary):
    negative passthrough; progress gating; the attributed() cap; the
    attribution-discount sigmoid; the timing discount (positive path only);
    plus BOTH ratio_penalty branches via the extracted production function.
    Computed once at the boundary; paid per epoch to horizon.
    """
    attribution_discount = 1.0
    if cf_windowed < 0:
        exp_arg = min(
            -reward_config.attribution_sigmoid_steepness * cf_windowed, 700.0
        )
        attribution_discount = 1.0 / (1.0 + math.exp(exp_arg))

    if q_settle_value < 0:
        bounded = reward_config.contribution_weight * q_settle_value
    else:
        if progress_at_boundary is None:
            attributed = q_settle_value * 0.5
        elif progress_at_boundary <= 0:
            attributed = 0.0
        elif q_settle_value >= progress_at_boundary:
            attributed = _compute_attributed_value(
                progress=progress_at_boundary,
                seed_contribution=q_settle_value,
                formula=reward_config.attribution_formula,
            )
        else:
            attributed = q_settle_value
        attributed *= attribution_discount
        bounded = reward_config.contribution_weight * attributed
        if not reward_config.disable_timing_discount:
            bounded *= _compute_timing_discount(
                germination_epoch,
                reward_config.germination_warmup_epochs,
                reward_config.germination_discount_floor,
            )

    return bounded + _compute_ratio_penalty(
        q_settle_value, cf_windowed, attribution_discount, reward_config
    )


__all__ = [
    "BoundaryPackage",
    "annuity_per_epoch",
    "boundary_package",
    "boundary_with_extension",
    "ewma",
    "next_boundary",
    "q_settle",
    "settlement_gate",
    "settlement_premium",
]
