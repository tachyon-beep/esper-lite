"""Phase 0 (Stage-0): batch per-term reward variance-share decomposition.

Given a batch of finalized SHAPED/ESCROW steps ``(reward_raw, RewardComponentsTelemetry)``,
``compute_variance_shares`` returns, per additend (``partition.ADDITEND_SIGN_MAP`` keys
plus ``residual``):

- ``share_i = Cov(R_i, R) / Var(R)`` — the per-term return-variance share. POPULATION
  convention (``ddof=0``), matching ``cov_rcf_return_share`` in ``ppo_agent`` so the shares
  sum to 1 exactly (a consistent correction is required for the identity).
- ``cov_i = std(R_i) / (|mean(R_i)| + eps)`` — the per-term coefficient of variation.

Top-line outputs:

- ``share_attribution`` — the ``bounded_attribution`` (``R_cf``) share, exactly what GATE 1's
  variance leg reads (``> 0.40``) and the term Phase −1 perturbs.
- ``residual_share`` — the reconciliation diagnostic. The GATE-0 reconciliation test is
  ``residual_share ≈ 0`` (a live term hidden in ``residual`` inflates it), NOT the
  tautological ``shares_sum == 1`` (which holds by covariance linearity for ANY classification
  once ``residual := reward_raw − Σ(named)`` closes the sum). ``shares_sum`` is reported only
  as a floating-point sanity check.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.rewards.partition import (
    ADDITEND_SIGN_MAP,
    RESIDUAL_KEY,
    decompose_additends,
)

_COV_EPS = 1e-8


def compute_variance_shares(
    steps: Sequence[tuple[float, RewardComponentsTelemetry]],
) -> dict[str, Any]:
    """Compute per-term return-variance shares + CoV over a batch of finalized steps.

    Args:
        steps: a sequence of ``(reward_raw, components)`` for the batch (one per PPO step).

    Returns:
        ``{"shares": {term: Cov(R_i,R)/Var(R)}, "cov": {term: CoV}, "share_attribution":
        float, "residual_share": float, "shares_sum": float, "n_steps": int}``. Includes
        the ``residual`` term in ``shares``.

    Raises:
        ValueError: if ``steps`` is empty (no batch to decompose).
    """
    if len(steps) == 0:
        raise ValueError("compute_variance_shares requires a non-empty batch")

    terms = [*ADDITEND_SIGN_MAP.keys(), RESIDUAL_KEY]
    # Per-term signed-contribution arrays + the reward_raw array, built from the
    # exhaustive-by-construction decomposition so the columns sum to reward_raw per row.
    columns: dict[str, list[float]] = {term: [] for term in terms}
    reward_raw_list: list[float] = []
    for reward_raw, components in steps:
        decomposed = decompose_additends(reward_raw, components)
        reward_raw_list.append(float(reward_raw))
        for term in terms:
            columns[term].append(decomposed[term])

    reward = np.asarray(reward_raw_list, dtype=np.float64)
    reward_centered = reward - reward.mean()
    var_reward = float(np.mean(reward_centered * reward_centered))  # population Var(R)

    shares: dict[str, float] = {}
    cov: dict[str, float] = {}
    for term in terms:
        col = np.asarray(columns[term], dtype=np.float64)
        col_mean = float(col.mean())
        col_centered = col - col_mean
        # Cov(R_i, R) / Var(R), population; 0 when the batch return is degenerate.
        if var_reward > 0.0:
            cov_term_reward = float(np.mean(col_centered * reward_centered))
            shares[term] = cov_term_reward / var_reward
        else:
            shares[term] = 0.0
        col_std = float(np.sqrt(np.mean(col_centered * col_centered)))
        cov[term] = col_std / (abs(col_mean) + _COV_EPS)

    return {
        "shares": shares,
        "cov": cov,
        "share_attribution": shares["bounded_attribution"],
        "residual_share": shares[RESIDUAL_KEY],
        "shares_sum": float(sum(shares.values())),
        "n_steps": len(steps),
    }


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


def compute_return_variance_metrics(
    buffer: Any, gamma: float, *, cf_term: str = "bounded_attribution"
) -> dict[str, float]:
    """Read the value-free Stage-0 gate off a finished rollout buffer.

    Pools the per-component signed additend SoA across envs (with env-boundary resets) and
    returns the flat gate metrics:

    - ``return_var_cf_share``       = ``Cov(R_cf, R)/Var(R)`` — THE Stage-0 gate (``> 0.40``).
    - ``return_var_main_share``     = ``Var(R_main)/Var(R)`` — the smoothness leg.
    - ``return_var_residual_share`` = the ``~0`` reconciliation diagnostic.

    Returns ``{}`` when the buffer collected no steps: ``compute_return_variance_shares``
    raises on an empty batch, so a flag-on update with no stored transitions must not crash.
    ``buffer`` is duck-typed on ``collect_component_rewards()`` (no import — avoids a cycle).
    """
    component_rewards, dones = buffer.collect_component_rewards()
    if not dones:
        return {}
    shares = compute_return_variance_shares(component_rewards, dones, gamma, cf_term=cf_term)
    return {
        "return_var_cf_share": shares["share_attribution"],
        "return_var_main_share": shares["r_main_var_share"],
        "return_var_residual_share": shares["residual_share"],
    }
