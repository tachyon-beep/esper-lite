"""Value function metrics computation for telemetry.

Computes TELE-220 to TELE-228 metrics from rollout buffer data.
"""

from __future__ import annotations

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
    bellman = (td_errors**2).mean()  # Mean squared TD error

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
