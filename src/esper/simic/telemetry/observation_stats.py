"""Observation Statistics Telemetry.

Captures per-step observation space health metrics for early detection
of input distribution issues that precede NaN gradients.

Per DRL Expert: Observation space drift and outliers are often the
root cause of training instability, catching them early prevents
debugging cascading failures in gradients/losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from esper.leyline import OBS_V3_BASE_FEATURE_SIZE, OBS_V3_SLOT_FEATURE_SIZE

if TYPE_CHECKING:
    import torch


_OBS_V3_HOST_FEATURE_SIZE: int = 3 + 5 + 5  # epoch/loss/acc + loss_hist(5) + acc_hist(5)
_OBS_V3_CONTEXT_FEATURE_SIZE: int = OBS_V3_BASE_FEATURE_SIZE - _OBS_V3_HOST_FEATURE_SIZE


@dataclass(slots=True)
class ObservationStatsTelemetry:
    """Observation space health metrics for debugging.

    Tracks feature statistics to catch input distribution issues
    before they propagate to NaN gradients.
    """

    # Per-feature-group statistics (computed over batch dimension)
    # Obs V3 has: slot features (per-slot), host features, context features
    slot_features_mean: float = 0.0
    slot_features_std: float = 0.0
    host_features_mean: float = 0.0
    host_features_std: float = 0.0
    context_features_mean: float = 0.0
    context_features_std: float = 0.0

    # Outlier detection (observations outside 3-sigma)
    outlier_pct: float = 0.0  # Fraction in [0, 1] (rendered as X.X%)

    # Saturation / clipping indicators (computed on NORMALIZED obs)
    near_clip_pct: float = 0.0  # Fraction with |x_norm| >= 0.9*clip
    clip_pct: float = 0.0  # Fraction clamped at |x_norm| == clip

    # Numerical health
    nan_count: int = 0
    inf_count: int = 0
    nan_pct: float = 0.0  # Fraction of NaNs in raw obs tensor
    inf_pct: float = 0.0  # Fraction of Infs in raw obs tensor

    # Normalization drift (how much running mean/std has shifted since epoch 0)
    normalization_drift: float = 0.0

    # Batch size (for context)
    batch_size: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dict for TelemetryEvent data field.

        Uses explicit dict construction for performance (PyTorch Expert pattern).
        """
        return {
            "slot_features_mean": self.slot_features_mean,
            "slot_features_std": self.slot_features_std,
            "host_features_mean": self.host_features_mean,
            "host_features_std": self.host_features_std,
            "context_features_mean": self.context_features_mean,
            "context_features_std": self.context_features_std,
            "outlier_pct": self.outlier_pct,
            "near_clip_pct": self.near_clip_pct,
            "clip_pct": self.clip_pct,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "nan_pct": self.nan_pct,
            "inf_pct": self.inf_pct,
            "normalization_drift": self.normalization_drift,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> "ObservationStatsTelemetry":
        """Reconstruct from dict (inverse of to_dict).

        Uses direct key access (not .get()) per CLAUDE.md defensive programming
        prohibition - if a field is missing, that's a bug to surface, not hide.
        """
        return cls(
            slot_features_mean=float(data["slot_features_mean"]),
            slot_features_std=float(data["slot_features_std"]),
            host_features_mean=float(data["host_features_mean"]),
            host_features_std=float(data["host_features_std"]),
            context_features_mean=float(data["context_features_mean"]),
            context_features_std=float(data["context_features_std"]),
            outlier_pct=float(data["outlier_pct"]),
            near_clip_pct=float(data["near_clip_pct"]),
            clip_pct=float(data["clip_pct"]),
            nan_count=int(data["nan_count"]),
            inf_count=int(data["inf_count"]),
            nan_pct=float(data["nan_pct"]),
            inf_pct=float(data["inf_pct"]),
            normalization_drift=float(data["normalization_drift"]),
            batch_size=int(data["batch_size"]),
        )


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
        normalizer_var: Current running variance from normalizer (optional)
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
    # Check via .any() which is faster than sum > 0 for sparse masks
    has_bad_values = nan_mask.any() or inf_mask.any()
    if has_bad_values:
        clean_obs = obs_tensor.clone()
        clean_obs[nan_mask | inf_mask] = 0.0
    else:
        clean_obs = obs_tensor

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

    # Normalization drift (how much the running mean has shifted)
    has_drift = (
        normalizer_mean is not None
        and initial_normalizer_mean is not None
        and normalizer_mean.shape == initial_normalizer_mean.shape
    )
    if has_drift:
        drift_t = (normalizer_mean - initial_normalizer_mean).abs().mean()
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
