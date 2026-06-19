"""Value function metrics computation for telemetry contracts.

Computes TELE-220 to TELE-228 metrics from rollout buffer data.
"""

from __future__ import annotations

import math
from typing import TypedDict

import torch


class ValueFunctionMetricsDict(TypedDict):
    """Computed value function metrics for telemetry."""

    # TELE-220: V-Return Correlation
    v_return_correlation: float

    # TELE-221/222/223: TD Error Statistics
    td_error_mean: float
    td_error_std: float
    bellman_error: float

    # TELE-224/225/226: Return Percentiles
    return_p10: float
    return_p50: float
    return_p90: float

    # TELE-227/228: Return Distribution
    return_variance: float
    return_skewness: float


@torch.inference_mode()
def compute_value_function_metrics(
    td_errors: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
) -> ValueFunctionMetricsDict:
    """Compute value function quality metrics from buffer data.

    Args:
        td_errors: TD errors from GAE computation [N]
        values: Value predictions V(s) [N]
        returns: Computed returns [N]

    Returns:
        Dictionary with all 9 value function metrics.

    PERF: Uses single .tolist() call at end to minimize GPU-CPU syncs.
    """
    if td_errors.numel() == 0:
        return ValueFunctionMetricsDict(
            v_return_correlation=0.0,
            td_error_mean=0.0,
            td_error_std=0.0,
            bellman_error=0.0,
            return_p10=0.0,
            return_p50=0.0,
            return_p90=0.0,
            return_variance=0.0,
            return_skewness=0.0,
        )

    # TD Error Statistics (TELE-221/222/223)
    td_mean = td_errors.mean()
    td_std = td_errors.std(correction=0)
    bellman = td_errors.abs().mean()  # Mean absolute TD error

    # V-Return Correlation (TELE-220)
    # Pearson: cov(X,Y) / (std(X) * std(Y))
    v_mean = values.mean()
    r_mean = returns.mean()
    v_std = values.std(correction=0)
    r_std = returns.std(correction=0)

    if v_std > 1e-8 and r_std > 1e-8:
        covariance = ((values - v_mean) * (returns - r_mean)).mean()
        correlation = covariance / (v_std * r_std)
        correlation = correlation.clamp(-1.0, 1.0)  # Numerical stability
    else:
        correlation = torch.tensor(0.0, device=td_errors.device)

    # Return Percentiles (TELE-224/225/226)
    # quantile() requires float tensor on same device
    p10 = torch.quantile(returns.float(), 0.1)
    p50 = torch.quantile(returns.float(), 0.5)
    p90 = torch.quantile(returns.float(), 0.9)

    # Return Variance (TELE-227)
    ret_var = returns.var(correction=0)

    # Return Skewness (TELE-228)
    # skewness = E[(X-mu)^3] / sigma^3
    if r_std > 1e-8:
        centered = returns - r_mean
        skewness = (centered**3).mean() / (r_std**3)
    else:
        skewness = torch.tensor(0.0, device=td_errors.device)

    # PERF: Single GPU->CPU sync via stacking all scalars
    results = torch.stack([
        correlation,
        td_mean,
        td_std,
        bellman,
        p10,
        p50,
        p90,
        ret_var,
        skewness,
    ]).tolist()

    return ValueFunctionMetricsDict(
        v_return_correlation=results[0],
        td_error_mean=results[1],
        td_error_std=results[2],
        bellman_error=results[3],
        return_p10=results[4],
        return_p50=results[5],
        return_p90=results[6],
        return_variance=results[7],
        return_skewness=results[8],
    )


def compute_floored_explained_variance(
    raw_values: torch.Tensor,
    valid_returns: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Variance-floored explained variance + floor-stabilized companion (W7 helper seam).

    Pure helper extracted from ``PPOAgent.update()`` so the EV/value-fit computation is
    unit-testable. EV = ``1 - Var(returns - values) / max(Var(returns), floor)``.

    Numerical conventions:
    - Bessel correction=1 (PyTorch ``Tensor.var()`` / ``.std()`` default) — this is the
      SHIPPED EV behavior. The sibling helpers in this module (``v_return_correlation``,
      ``bellman_error``, buffer-wide ``return_variance``) use ``correction=0``; the EV family
      deliberately keeps ``correction=1`` and the two are documented as distinct.
    - The ``torch.clamp(var_returns, min=floor)`` is a HARD clamp: a no-op above the floor (so a
      healthy batch's EV is numerically unchanged) and a denominator substitution only below it.
    - ``value_nrmse`` floors its denominator on the STD scale (additive ``sqrt(floor)``) so it
      cannot blow up at low variance.

    B3 / no-bug-hiding: there is NO numel guard here. A ``numel<=1`` mask yields NaN
    ``var()``/``std()`` and falls through to the caller's non-finite return-stat raise — it must
    NEVER be swallowed into an ``ev_low_return_variance=True``-on-degenerate convention or a
    NaN-by-convention EV. The floor is for the legitimate low (but finite, ``numel>=2``)
    variance regime only, never as a ``clamp(NaN)`` sanitizer.

    Args:
        raw_values: Denormalized value predictions on the raw return scale [N].
        valid_returns: Returns on the raw scale [N].
        floor: Absolute return-variance floor (raw scale; e.g. 1.0).

    Returns:
        ``(explained_variance, value_nrmse, ev_low_return_variance, ev_return_variance)``.
    """
    residual_var = (valid_returns - raw_values).var()  # Bessel (correction=1)
    var_returns = valid_returns.var()  # Bessel (correction=1) -- EV denominator
    denom = torch.clamp(var_returns, min=floor)
    ev_low_return_variance = bool(var_returns < floor)
    explained_variance = 1.0 - residual_var / denom
    value_nrmse = torch.sqrt(residual_var) / (valid_returns.std() + math.sqrt(floor))
    ev_return_variance = var_returns
    return explained_variance, value_nrmse, ev_low_return_variance, ev_return_variance


def compute_floored_aux_explained_variance(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
    floor_fraction: float,
    floor_min: float,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Variance-floored aux (contribution-target) explained variance + companions.

    Aux mirror of :func:`compute_floored_explained_variance`, operating on the
    contribution-target scale rather than the raw-return scale. EV =
    ``1 - Var(target - pred) / max(Var(target), floor)``.

    The contribution-target scale is NOT calibratable from persisted telemetry, so the floor
    is **data-relative, recomputed each update**: ``floor = max(floor_min, floor_fraction *
    Var0(target))``. The ``0.05`` fraction mirrors the main EV floor being ~3-5% of the healthy
    return-variance median; ``floor_min`` is the absolute minimum (the old bare
    ``clamp(min=0.01)`` value) so a genuinely tiny-but-real spread still floors sanely.

    Numerical conventions (deliberately mixed, matching the calibration recipe):
    - The EV DENOMINATOR clamp uses ``Tensor.var()`` (Bessel correction=1, the shipped EV
      convention — identical to the main helper).
    - The data-relative FLOOR MAGNITUDE uses ``Var0`` (population variance, ``correction=0``):
      a stable estimate of the data scale, per the recipe's ``tgt.var(unbiased=False)``.
    - The ``ev_low_return_variance`` flag compares the true (correction=1) variance against the
      computed floor.
    - ``aux_value_nrmse`` floors its denominator on the STD scale (additive ``sqrt(floor)``) so
      it cannot blow up at low variance.

    B3 / no-bug-hiding: there is NO numel guard here. The caller MUST require ``numel>=2``
    before calling (a ``numel<=1`` flatten yields NaN ``var()``/``std()``); the aux block must
    SKIP this update rather than feed a degenerate batch here. This helper never sanitizes NaN.

    Args:
        pred_flat: Flattened contribution predictions on the contribution-target scale [N].
        target_flat: Flattened contribution targets [N] (same selection as ``pred_flat``).
        floor_fraction: Data-relative floor fraction (e.g. 0.05).
        floor_min: Absolute minimum floor (e.g. 0.01).

    Returns:
        ``(aux_explained_variance, aux_value_nrmse, aux_ev_low_return_variance, aux_ev_return_variance)``.
    """
    var_targets = target_flat.var()  # Bessel (correction=1) -- EV denominator + flag basis
    floor = max(floor_min, floor_fraction * float(target_flat.var(correction=0)))
    denom = torch.clamp(var_targets, min=floor)
    aux_ev_low_return_variance = bool(var_targets < floor)
    residual_var = (pred_flat - target_flat).var()  # Bessel (correction=1)
    aux_explained_variance = 1.0 - residual_var / denom
    aux_value_nrmse = torch.sqrt(residual_var) / (target_flat.std() + math.sqrt(floor))
    aux_ev_return_variance = var_targets
    return (
        aux_explained_variance,
        aux_value_nrmse,
        aux_ev_low_return_variance,
        aux_ev_return_variance,
    )


__all__ = [
    "ValueFunctionMetricsDict",
    "compute_value_function_metrics",
    "compute_floored_explained_variance",
    "compute_floored_aux_explained_variance",
]
