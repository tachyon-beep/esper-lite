"""Debug-Level Telemetry for Simic Training.

These functions are expensive (5-50ms) and should only be called
when anomalies are detected or debug mode is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class LayerGradientStats:
    """Per-layer gradient statistics for debugging."""

    layer_name: str
    param_count: int

    # Distribution statistics
    grad_norm: float
    grad_mean: float
    grad_std: float
    grad_min: float
    grad_max: float

    # Health indicators
    zero_fraction: float
    small_fraction: float  # < 1e-6
    large_fraction: float  # > 10.0
    nan_count: int
    inf_count: int

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)


def collect_per_layer_gradients(
    model: nn.Module,
    small_threshold: float = 1e-6,
    large_threshold: float = 10.0,
) -> list[LayerGradientStats]:
    """Collect detailed per-layer gradient statistics.

    WARNING: This is expensive (~10-50ms for large models).
    Only use in debug mode.

    Args:
        model: PyTorch model with gradients computed
        small_threshold: Threshold for "small" gradient count
        large_threshold: Threshold for "large" gradient count

    Returns:
        List of LayerGradientStats, one per parameter
    """
    stats = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        flat = grad.view(-1)

        # Batch all stats into single GPU sync per layer (11 .item() → 1 .tolist())
        # Use correction=0 for std to avoid division by zero on single-element tensors
        if flat.numel() > 0:
            batch_stats = torch.stack([
                grad.norm(),
                flat.mean(),
                flat.std(correction=0),
                flat.min(),
                flat.max(),
                (flat == 0).float().mean(),
                (flat.abs() < small_threshold).float().mean(),
                (flat.abs() > large_threshold).float().mean(),
                torch.isnan(flat).sum().float(),
                torch.isinf(flat).sum().float(),
            ])
            (grad_norm, grad_mean, grad_std, grad_min, grad_max,
             zero_frac, small_frac, large_frac, nan_cnt, inf_cnt) = batch_stats.tolist()
        else:
            grad_norm = grad_mean = grad_std = grad_min = grad_max = 0.0
            zero_frac = small_frac = large_frac = 0.0
            nan_cnt = inf_cnt = 0.0

        layer_stats = LayerGradientStats(
            layer_name=name,
            param_count=param.numel(),
            grad_norm=grad_norm,
            grad_mean=grad_mean,
            grad_std=grad_std,
            grad_min=grad_min,
            grad_max=grad_max,
            zero_fraction=zero_frac,
            small_fraction=small_frac,
            large_fraction=large_frac,
            nan_count=int(nan_cnt),
            inf_count=int(inf_cnt),
        )
        stats.append(layer_stats)

    return stats


@dataclass(slots=True)
class NumericalStabilityReport:
    """Detailed numerical stability analysis for debugging."""

    # NaN/Inf locations
    nan_in_weights: list[str] = field(default_factory=list)
    nan_in_gradients: list[str] = field(default_factory=list)
    inf_in_weights: list[str] = field(default_factory=list)
    inf_in_gradients: list[str] = field(default_factory=list)

    # Value ranges
    max_weight: float = 0.0
    max_gradient: float = 0.0

    # Loss stability
    loss_value: float = 0.0
    loss_is_finite: bool = True

    def has_issues(self) -> bool:
        """Check if any numerical issues detected."""
        return (
            len(self.nan_in_weights) > 0
            or len(self.nan_in_gradients) > 0
            or len(self.inf_in_weights) > 0
            or len(self.inf_in_gradients) > 0
            or not self.loss_is_finite
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)


def check_numerical_stability(
    model: nn.Module,
    loss: torch.Tensor | None = None,
) -> NumericalStabilityReport:
    """Check for numerical stability issues.

    Call after backward() but before optimizer.step() to catch issues
    before they propagate.

    Args:
        model: Model to check
        loss: Optional loss tensor to check

    Returns:
        NumericalStabilityReport
    """
    nan_weights = []
    nan_grads = []
    inf_weights = []
    inf_grads = []

    # Collect max values as tensors, then sync once at the end
    weight_maxes = []
    grad_maxes = []

    for name, param in model.named_parameters():
        # Check weights (these are boolean checks, no sync needed)
        if torch.isnan(param.data).any():
            nan_weights.append(name)
        if torch.isinf(param.data).any():
            inf_weights.append(name)
        weight_maxes.append(param.data.abs().max())

        # Check gradients
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)
            grad_maxes.append(param.grad.abs().max())

    # Single sync for all max values
    max_weight = torch.stack(weight_maxes).max().item() if weight_maxes else 0.0
    max_grad = torch.stack(grad_maxes).max().item() if grad_maxes else 0.0

    # Check loss
    loss_val = 0.0
    loss_finite = True
    if loss is not None:
        loss_val = loss.item()
        loss_finite = not (math.isnan(loss_val) or math.isinf(loss_val))

    return NumericalStabilityReport(
        nan_in_weights=nan_weights,
        nan_in_gradients=nan_grads,
        inf_in_weights=inf_weights,
        inf_in_gradients=inf_grads,
        max_weight=max_weight,
        max_gradient=max_grad,
        loss_value=loss_val,
        loss_is_finite=loss_finite,
    )


@dataclass(slots=True)
class RatioExplosionDiagnostic:
    """Diagnostic data when PPO ratios explode.

    Captures the specific transitions that caused ratio explosion
    for post-hoc debugging.
    """

    # Indices of problematic transitions
    worst_ratio_indices: list[int] = field(default_factory=list)
    worst_ratio_values: list[float] = field(default_factory=list)
    worst_ratio_actions: list[int] = field(default_factory=list)

    # Log prob divergence
    logit_diff_mean: float = 0.0
    logit_diff_max: float = 0.0

    @classmethod
    def from_batch(
        cls,
        ratio: "torch.Tensor",
        old_log_probs: "torch.Tensor",
        new_log_probs: "torch.Tensor",
        states: "torch.Tensor",
        actions: "torch.Tensor",
        action_masks: "torch.Tensor",
        max_threshold: float = 5.0,
        min_threshold: float = 0.1,
    ) -> "RatioExplosionDiagnostic":
        """Create diagnostic from batch tensors.

        Args:
            ratio: PPO ratio tensor [N]
            old_log_probs: Old log probabilities [N]
            new_log_probs: New log probabilities [N]
            states: State observations [N, state_dim]
            actions: Actions taken [N]
            action_masks: Valid action masks [N, action_dim]
            max_threshold: Ratio above this is problematic
            min_threshold: Ratio below this is problematic

        Returns:
            RatioExplosionDiagnostic
        """
        # Find problematic indices
        bad_mask = (ratio > max_threshold) | (ratio < min_threshold)
        bad_indices = bad_mask.nonzero(as_tuple=True)[0].tolist()

        # Extract worst values
        worst_values = ratio[bad_indices].tolist() if bad_indices else []
        worst_actions = actions[bad_indices].tolist() if bad_indices else []

        # Compute log prob divergence (batch into single sync)
        logit_diff = (new_log_probs - old_log_probs).abs()
        diff_stats = torch.stack([logit_diff.mean(), logit_diff.max()])
        logit_diff_mean, logit_diff_max = diff_stats.tolist()

        return cls(
            worst_ratio_indices=bad_indices,
            worst_ratio_values=worst_values,
            worst_ratio_actions=worst_actions,
            logit_diff_mean=logit_diff_mean,
            logit_diff_max=logit_diff_max,
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)


__all__ = [
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
    "RatioExplosionDiagnostic",
]
