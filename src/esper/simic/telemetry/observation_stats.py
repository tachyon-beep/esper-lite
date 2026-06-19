"""Observation Statistics Computation.

Captures per-step observation space health metrics for early detection
of input distribution issues that precede NaN gradients.

Per DRL Expert: Observation space drift and outliers are often the
root cause of training instability, catching them early prevents
debugging cascading failures in gradients/losses.

ObservationStatsTelemetry (the wire-format contract) lives in
esper.leyline.telemetry_contracts.  This module provides the heavy
compute function that needs torch and OBS_V3 layout constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from esper.leyline import OBS_V3_BASE_FEATURE_SIZE, OBS_V3_SLOT_FEATURE_SIZE
from esper.leyline.telemetry_contracts import ObservationStatsTelemetry

if TYPE_CHECKING:
    import torch


_OBS_V3_HOST_FEATURE_SIZE: int = 3 + 5 + 5  # epoch/loss/acc + loss_hist(5) + acc_hist(5)
_OBS_V3_CONTEXT_FEATURE_SIZE: int = OBS_V3_BASE_FEATURE_SIZE - _OBS_V3_HOST_FEATURE_SIZE


def compute_observation_stats(
    obs_tensor: "torch.Tensor",
    *,
    normalized_obs_tensor: "torch.Tensor",
    normalizer_mean: "torch.Tensor | None" = None,
    normalizer_var: "torch.Tensor | None" = None,
    initial_normalizer_mean: "torch.Tensor | None" = None,
    clip: float = 10.0,
) -> ObservationStatsTelemetry:
    """Compute observation statistics from raw observation tensor.

    Args:
        obs_tensor: Raw observation tensor [batch_size, obs_dim]
        normalized_obs_tensor: Normalized+clipped tensor fed to policy [batch_size, obs_dim]
        normalizer_mean: Current running mean from normalizer (optional)
        normalizer_var: Current running variance from normalizer (optional).
            Compared against the normalizer's initial variance of 1.0.
        initial_normalizer_mean: Initial running mean for drift calculation (optional)
        clip: Normalizer clip value (used for saturation/clipping indicators)

    Returns:
        ObservationStatsTelemetry with computed statistics.

    Performance:
        Uses PyTorch ops to stay on GPU, only converts final scalars to Python.
        Should add ~0.1ms overhead per step (negligible vs forward/backward pass).
    """
    import torch

    batch_size = obs_tensor.shape[0]
    obs_dim = obs_tensor.shape[1]

    if obs_dim < OBS_V3_BASE_FEATURE_SIZE:
        raise ValueError(
            f"Obs dim {obs_dim} < OBS_V3_BASE_FEATURE_SIZE={OBS_V3_BASE_FEATURE_SIZE}."
        )
    if _OBS_V3_CONTEXT_FEATURE_SIZE <= 0:
        raise RuntimeError(
            "Invalid Obs V3 layout: context feature size must be positive."
        )
    slot_dim = obs_dim - OBS_V3_BASE_FEATURE_SIZE
    if slot_dim <= 0:
        raise ValueError(
            f"Obs dim {obs_dim} has no slot features; expected Obs V3 layout."
        )
    if slot_dim % OBS_V3_SLOT_FEATURE_SIZE != 0:
        raise ValueError(
            f"Obs V3 slot_dim {slot_dim} is not divisible by OBS_V3_SLOT_FEATURE_SIZE="
            f"{OBS_V3_SLOT_FEATURE_SIZE}."
        )
    if normalized_obs_tensor.shape != obs_tensor.shape:
        raise ValueError(
            "normalized_obs_tensor shape must match obs_tensor shape: "
            f"{tuple(normalized_obs_tensor.shape)} != {tuple(obs_tensor.shape)}"
        )

    # Check for NaN/Inf
    nan_mask = torch.isnan(obs_tensor)
    inf_mask = torch.isinf(obs_tensor)
    nan_count_t = nan_mask.sum()
    inf_count_t = inf_mask.sum()
    total_elements = batch_size * obs_dim

    # Replace NaN/Inf with 0 for stats computation
    # Keep the decision tensor-native; branching on nan_mask.any() / inf_mask.any()
    # would synchronize CUDA tensors with the host on every telemetry sample.
    zero = torch.zeros((), device=obs_tensor.device, dtype=obs_tensor.dtype)
    clean_obs = torch.where(nan_mask | inf_mask, zero, obs_tensor)

    # Group stats (Obs V3 layout)
    host = clean_obs[:, :_OBS_V3_HOST_FEATURE_SIZE]
    context = clean_obs[:, _OBS_V3_HOST_FEATURE_SIZE:OBS_V3_BASE_FEATURE_SIZE]
    slots = clean_obs[:, OBS_V3_BASE_FEATURE_SIZE:]

    # C2 FIX: Batch all scalar GPU tensors to single sync.
    # Before: 12 individual .item() calls = 12 GPU sync barriers (~0.6ms).
    # After: Single .tolist() call = 1 GPU sync barrier (~0.05ms).
    host_mean_t = host.mean()
    host_std_t = host.std(unbiased=False)
    context_mean_t = context.mean()
    context_std_t = context.std(unbiased=False)
    slot_mean_t = slots.mean()
    slot_std_t = slots.std(unbiased=False)

    # Outlier detection: count values outside 3-sigma
    # Use per-feature mean/std for outlier detection
    feature_mean = clean_obs.mean(dim=0, keepdim=True)
    feature_std = (
        clean_obs.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    )  # Avoid div by zero
    z_scores = torch.abs((clean_obs - feature_mean) / feature_std)
    outlier_count_t = (z_scores > 3.0).sum()

    # Saturation / clipping indicators on normalized observations
    abs_norm = normalized_obs_tensor.abs()
    near_clip_pct_t = (abs_norm >= (0.9 * clip)).float().mean()
    clip_pct_t = (abs_norm >= (clip - 1e-6)).float().mean()

    # Normalization drift (how much the running mean/std has shifted)
    drift_parts = []
    if (
        normalizer_mean is not None
        and initial_normalizer_mean is not None
        and normalizer_mean.shape == initial_normalizer_mean.shape
    ):
        drift_parts.append((normalizer_mean - initial_normalizer_mean).abs().mean())
    if normalizer_var is not None:
        current_std = torch.sqrt(normalizer_var.clamp_min(0.0))
        initial_std = torch.ones_like(current_std)
        drift_parts.append((current_std - initial_std).abs().mean())
    if drift_parts:
        drift_t = torch.stack(drift_parts).sum()
    else:
        drift_t = torch.tensor(0.0, device=obs_tensor.device)

    # Single GPU→CPU sync: stack all scalar tensors and transfer at once
    # This reduces 12 sync barriers to 1, saving ~0.5ms per step
    stacked = torch.stack([
        nan_count_t.float(),       # 0
        inf_count_t.float(),       # 1
        host_mean_t,               # 2
        host_std_t,                # 3
        context_mean_t,            # 4
        context_std_t,             # 5
        slot_mean_t,               # 6
        slot_std_t,                # 7
        outlier_count_t.float(),   # 8
        near_clip_pct_t,           # 9
        clip_pct_t,                # 10
        drift_t,                   # 11
    ])
    values = stacked.tolist()  # Single GPU sync

    # Unpack values
    nan_count = int(values[0])
    inf_count = int(values[1])
    host_mean = float(values[2])
    host_std = float(values[3])
    context_mean = float(values[4])
    context_std = float(values[5])
    slot_mean = float(values[6])
    slot_std = float(values[7])
    outlier_count = int(values[8])
    near_clip_pct = float(values[9])
    clip_pct_val = float(values[10])
    normalization_drift = float(values[11])

    # Compute percentages from counts
    nan_pct = (nan_count / total_elements) if total_elements > 0 else 0.0
    inf_pct = (inf_count / total_elements) if total_elements > 0 else 0.0
    outlier_pct = (outlier_count / total_elements) if total_elements > 0 else 0.0

    return ObservationStatsTelemetry(
        slot_features_mean=slot_mean,
        slot_features_std=slot_std,
        host_features_mean=host_mean,
        host_features_std=host_std,
        context_features_mean=context_mean,
        context_features_std=context_std,
        outlier_pct=outlier_pct,
        near_clip_pct=near_clip_pct,
        clip_pct=clip_pct_val,
        nan_count=nan_count,
        inf_count=inf_count,
        nan_pct=nan_pct,
        inf_pct=inf_pct,
        normalization_drift=normalization_drift,
        batch_size=batch_size,
    )


__all__ = ["ObservationStatsTelemetry", "compute_observation_stats"]
# ObservationStatsTelemetry is re-exported here for simic-internal callers;
# the canonical home is esper.leyline.telemetry_contracts.
