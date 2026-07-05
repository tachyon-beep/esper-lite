"""Value-free per-RETURN reward-variance decomposition (EV-stab Stage-0 gate).

``compute_return_variance_shares`` accumulates each per-step signed reward component
into a discounted return-to-go (value-free, ``lambda=1`` within segments, raw scale),
then returns the per-term return-variance share ``share_i = Cov(R_i, R) / Var(R)``.
By covariance linearity ``sum_i Cov(R_i, R) = Var(R)``, so the shares sum to 1 exactly
whenever the per-step components sum to the total reward.

This is deliberately DISTINCT from the per-STEP ``compute_variance_shares`` (which lives
on the reward-composition side): return variance is ``sum_t sum_s gamma^{t+s} Cov(r_t, r_s)``,
so temporally persistent streams contribute more than their per-step share suggests. The
Stage-0 GATE (``Cov(R_cf, R)/Var(R) > 0.40``) reads THIS return-level version.

It is also VALUE-FREE: unlike the Stage-2 ON-leg metric (which decomposes GAE lambda-returns
built from the cf value head's bootstrapped predictions), this decomposes the raw discounted
reward return and so is identical on both legs — the control-run diagnostic Stage-0 requires.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _discounted_return_to_go(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    """Value-free discounted return-to-go, reset at episode boundaries.

    ``R[t] = rewards[t] + gamma * (0 if dones[t] else R[t + 1])`` — a terminal step
    (``dones[t]``) contributes only its own reward (no bootstrap into the next episode).
    """
    n = rewards.shape[0]
    returns = np.empty(n, dtype=np.float64)
    running = 0.0
    for t in range(n - 1, -1, -1):
        if bool(dones[t]):
            running = 0.0
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


def compute_return_variance_shares(
    component_rewards: dict[str, Sequence[float]],
    dones: Sequence[bool],
    gamma: float,
    *,
    cf_term: str = "bounded_attribution",
    residual_term: str = "residual",
) -> dict[str, Any]:
    """Per-term return-variance shares over a batch of finalized steps.

    Args:
        component_rewards: ``{term: per-step signed reward}`` — the exhaustive signed
            additend decomposition (columns sum to ``reward_raw`` per step).
        dones: per-step episode-termination flags (return-to-go resets at ``True``).
        gamma: discount factor for the return accumulation.
        cf_term: the counterfactual stream key (``R_cf``); its share is the epic gate
            (``> 0.40``). ``R_main := R_total - R_cf``.
        residual_term: the telemetry-less catch-all key; its share is the ``~0``
            reconciliation diagnostic.

    Returns:
        ``{"shares": {term: Cov(R_i, R)/Var(R)}, "share_attribution": float,
        "residual_share": float, "r_main_var_share": float, "shares_sum": float,
        "n_steps": int}``. ``r_main_var_share = Var(R_main)/Var(R)`` is the smoothness
        leg (variance-share, not the fragile CoV).
    """
    terms = list(component_rewards.keys())
    dones_arr = np.asarray(dones, dtype=bool)
    n = int(dones_arr.shape[0])
    if n == 0:
        raise ValueError("compute_return_variance_shares requires a non-empty batch")

    component_returns: dict[str, np.ndarray] = {}
    total_returns = np.zeros(n, dtype=np.float64)
    for term in terms:
        rewards = np.asarray(component_rewards[term], dtype=np.float64)
        returns = _discounted_return_to_go(rewards, dones_arr, gamma)
        component_returns[term] = returns
        total_returns += returns

    total_centered = total_returns - total_returns.mean()
    var_total = float(np.mean(total_centered * total_centered))  # population Var(R)

    # Degenerate batch (constant total return, e.g. an all-terminal single-step update):
    # every share is 0.0 by convention — no signal, not a bug. Guards the div-by-zero.
    if var_total <= 0.0:
        zero_shares = dict.fromkeys(terms, 0.0)
        return {
            "shares": zero_shares,
            "share_attribution": 0.0,
            "residual_share": 0.0,
            "r_main_var_share": 0.0,
            "shares_sum": 0.0,
            "n_steps": n,
        }

    shares: dict[str, float] = {}
    for term in terms:
        centered = component_returns[term] - component_returns[term].mean()
        cov = float(np.mean(centered * total_centered))
        shares[term] = cov / var_total

    # Gate-specific outputs. The `in` checks are genuine optionality — the same helper
    # serves a generic decomposition (arbitrary term set) and the named-term Stage-0 gate;
    # an absent cf term means R_cf = 0 (so R_main == R_total, share 0), not a hidden bug.
    r_cf_returns = component_returns[cf_term] if cf_term in component_returns else np.zeros(n)
    r_main_returns = total_returns - r_cf_returns
    r_main_centered = r_main_returns - r_main_returns.mean()
    var_main = float(np.mean(r_main_centered * r_main_centered))
    r_main_var_share = var_main / var_total

    return {
        "shares": shares,
        "share_attribution": shares[cf_term] if cf_term in shares else 0.0,
        "residual_share": shares[residual_term] if residual_term in shares else 0.0,
        "r_main_var_share": r_main_var_share,
        "shares_sum": float(sum(shares.values())),
        "n_steps": n,
    }
