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

if TYPE_CHECKING:
    import torch


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
    outlier_pct: float = 0.0

    # Numerical health
    nan_count: int = 0
    inf_count: int = 0

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
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
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
            nan_count=int(data["nan_count"]),
            inf_count=int(data["inf_count"]),
            normalization_drift=float(data["normalization_drift"]),
            batch_size=int(data["batch_size"]),
        )


def compute_observation_stats(
    obs_tensor: "torch.Tensor",
    normalizer_mean: "torch.Tensor | None" = None,
    normalizer_var: "torch.Tensor | None" = None,
    initial_normalizer_mean: "torch.Tensor | None" = None,
) -> ObservationStatsTelemetry:
    """Compute observation statistics from raw observation tensor.

    Args:
        obs_tensor: Raw observation tensor [batch_size, obs_dim]
        normalizer_mean: Current running mean from normalizer (optional)
        normalizer_var: Current running variance from normalizer (optional)
        initial_normalizer_mean: Initial running mean for drift calculation (optional)

    Returns:
        ObservationStatsTelemetry with computed statistics.

    Performance:
        Uses PyTorch ops to stay on GPU, only converts final scalars to Python.
        Should add ~0.1ms overhead per step (negligible vs forward/backward pass).
    """
    import torch

    batch_size = obs_tensor.shape[0]
    obs_dim = obs_tensor.shape[1]

    # Check for NaN/Inf
    nan_mask = torch.isnan(obs_tensor)
    inf_mask = torch.isinf(obs_tensor)
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())

    # Replace NaN/Inf with 0 for stats computation
    # Only clone if bad values exist (PyTorch Expert: avoids ~0.02ms allocation in 99.9% case)
    has_bad_values = nan_count > 0 or inf_count > 0
    if has_bad_values:
        clean_obs = obs_tensor.clone()
        clean_obs[nan_mask | inf_mask] = 0.0
    else:
        clean_obs = obs_tensor

    # Overall statistics (for simplicity, compute across all features)
    # For per-group stats, would need to know feature layout
    overall_mean = float(clean_obs.mean().item())
    overall_std = float(clean_obs.std().item())

    # Outlier detection: count values outside 3-sigma
    # Use per-feature mean/std for outlier detection
    feature_mean = clean_obs.mean(dim=0, keepdim=True)
    feature_std = clean_obs.std(dim=0, keepdim=True) + 1e-8  # Avoid div by zero
    z_scores = torch.abs((clean_obs - feature_mean) / feature_std)
    outlier_count = int((z_scores > 3.0).sum().item())
    total_elements = batch_size * obs_dim
    outlier_pct = (outlier_count / total_elements) * 100.0 if total_elements > 0 else 0.0

    # Normalization drift (how much the running mean has shifted)
    normalization_drift = 0.0
    if (
        normalizer_mean is not None
        and initial_normalizer_mean is not None
        and normalizer_mean.shape == initial_normalizer_mean.shape
    ):
        drift = (normalizer_mean - initial_normalizer_mean).abs().mean()
        normalization_drift = float(drift.item())

    return ObservationStatsTelemetry(
        # Use overall stats for slot/host/context (simplification)
        # Real per-group stats would require knowing the feature layout
        slot_features_mean=overall_mean,
        slot_features_std=overall_std,
        host_features_mean=overall_mean,
        host_features_std=overall_std,
        context_features_mean=overall_mean,
        context_features_std=overall_std,
        outlier_pct=outlier_pct,
        nan_count=nan_count,
        inf_count=inf_count,
        normalization_drift=normalization_drift,
        batch_size=batch_size,
    )


__all__ = ["ObservationStatsTelemetry", "compute_observation_stats"]
