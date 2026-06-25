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
