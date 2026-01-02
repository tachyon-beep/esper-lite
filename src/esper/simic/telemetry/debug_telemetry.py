"""Debug-Level Telemetry for Simic Training.

These functions are expensive (5-50ms) and should only be called
when anomalies are detected or debug mode is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any

import torch
import torch.nn as nn


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

    def to_dict(self) -> dict[str, Any]:
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
    # Collect all layer stats tensors first, then sync once at the end
    layer_names: list[str] = []
    param_counts: list[int] = []
    stat_tensors: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        # Use flatten() instead of view(-1) to handle non-contiguous tensors
        # (e.g., transposed parameters, tied weights)
        flat = grad.flatten()

        layer_names.append(name)
        param_counts.append(param.numel())

        # Batch all stats into a single tensor (10 values per layer)
        # Use correction=0 for std to avoid division by zero on single-element tensors
        if flat.numel() > 0:
            layer_stats = torch.stack([
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
        else:
            # Empty tensor: all zeros
            layer_stats = torch.zeros(10, device=grad.device)

        stat_tensors.append(layer_stats)

    # Single GPU sync: stack all layers and transfer to CPU
    if not stat_tensors:
        return []

    all_stats = torch.stack(stat_tensors).tolist()  # [num_layers, 10]

    # Build LayerGradientStats objects from synced data
    results = []
    for i, (name, param_count) in enumerate(zip(layer_names, param_counts)):
        (grad_norm, grad_mean, grad_std, grad_min, grad_max,
         zero_frac, small_frac, large_frac, nan_cnt, inf_cnt) = all_stats[i]

        results.append(LayerGradientStats(
            layer_name=name,
            param_count=param_count,
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
        ))

    return results


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)


def check_numerical_stability(
    model: nn.Module,
    loss: torch.Tensor | None = None,
) -> NumericalStabilityReport:
    """Check for numerical stability issues.

    Call after backward() but before optimizer.step() to catch issues
    before they propagate.

    PERF NOTE: This function intentionally does O(num_params) GPU syncs via
    torch.isnan().any() and torch.isinf().any() per parameter. This is
    acceptable because:
    1. This is a DEBUG function, not hot-path production code
    2. We need to identify WHICH specific parameters have issues
    3. Batching would require stacking all params (memory-expensive)

    When debugging is disabled (the normal case), this function is not called.

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
        # Check weights (.any() triggers GPU sync per param - acceptable for debug)
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

    # Single sync for all max values (assumes all params on same device)
    # No defensive pattern - empty lists indicate model has no parameters (bug)
    max_weight = torch.stack(weight_maxes).max().item()
    max_grad = torch.stack(grad_maxes).max().item()

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


# Maximum number of exemplars to report in diagnostics (prevents payload explosion)
_MAX_DIAGNOSTIC_EXEMPLARS = 100


@dataclass(slots=True)
class RatioExplosionDiagnostic:
    """Diagnostic data when PPO ratios explode.

    Captures the specific transitions that caused ratio explosion
    for post-hoc debugging. Lists are capped to _MAX_DIAGNOSTIC_EXEMPLARS
    to prevent memory/telemetry explosion during widespread instability.
    """

    # Indices of problematic transitions (capped to _MAX_DIAGNOSTIC_EXEMPLARS)
    worst_ratio_indices: list[int] = field(default_factory=list)
    worst_ratio_values: list[float] = field(default_factory=list)
    worst_ratio_actions: list[int] = field(default_factory=list)

    # Summary counts (always accurate, unlike capped lists)
    num_bad_total: int = 0
    num_nonfinite: int = 0

    # Log prob divergence
    logit_diff_mean: float = 0.0
    logit_diff_max: float = 0.0

    @classmethod
    def from_batch(
        cls,
        ratio: "torch.Tensor",
        old_log_probs: "torch.Tensor",
        new_log_probs: "torch.Tensor",
        actions: "torch.Tensor",
        max_threshold: float = 5.0,
        min_threshold: float = 0.1,
        states: "torch.Tensor | None" = None,
        action_masks: "torch.Tensor | None" = None,
    ) -> "RatioExplosionDiagnostic":
        """Create diagnostic from batch tensors.

        Args:
            ratio: PPO ratio tensor [N]
            old_log_probs: Old log probabilities [N]
            new_log_probs: New log probabilities [N]
            actions: Actions taken [N]
            max_threshold: Ratio above this is problematic
            min_threshold: Ratio below this is problematic
            states: State observations [N, state_dim] (unused, reserved for future)
            action_masks: Valid action masks [N, action_dim] (unused, reserved for future)

        Returns:
            RatioExplosionDiagnostic
        """
        # Handle empty tensors gracefully (edge case: no valid transitions)
        # Empty tensors cause mean() to return nan and max() to raise RuntimeError
        if ratio.numel() == 0:
            return cls(
                worst_ratio_indices=[],
                worst_ratio_values=[],
                worst_ratio_actions=[],
                num_bad_total=0,
                num_nonfinite=0,
                logit_diff_mean=0.0,
                logit_diff_max=0.0,
            )

        # Find problematic indices - include NaN/Inf as "bad" since they indicate
        # numerical instability that must be debugged. NaN comparisons are always
        # False, so we must explicitly check with isfinite().
        nonfinite_mask = ~torch.isfinite(ratio)
        threshold_mask = (ratio > max_threshold) | (ratio < min_threshold)
        bad_mask = nonfinite_mask | threshold_mask

        # Count totals before capping
        num_nonfinite = int(nonfinite_mask.sum().item())
        num_bad_total = int(bad_mask.sum().item())

        # Compute log prob divergence (for diff_stats)
        # Handle NaN in log probs gracefully
        logit_diff = (new_log_probs - old_log_probs).abs()
        finite_diff = logit_diff[torch.isfinite(logit_diff)]
        if finite_diff.numel() > 0:
            diff_stats = torch.stack([finite_diff.mean(), finite_diff.max()])
        else:
            diff_stats = torch.zeros(2, device=ratio.device)

        # PERF: Batch GPU→CPU transfers before calling .tolist()
        # Move all tensors to CPU together, then extract Python values.
        bad_indices_tensor = bad_mask.nonzero(as_tuple=True)[0]

        # Cap to _MAX_DIAGNOSTIC_EXEMPLARS to prevent payload explosion during instability
        if bad_indices_tensor.numel() > _MAX_DIAGNOSTIC_EXEMPLARS:
            # Keep first _MAX_DIAGNOSTIC_EXEMPLARS (could also sample randomly, but first-N
            # is deterministic and often captures the onset of instability)
            bad_indices_tensor = bad_indices_tensor[:_MAX_DIAGNOSTIC_EXEMPLARS]

        bad_indices_cpu = bad_indices_tensor.cpu()
        diff_stats_cpu = diff_stats.cpu()

        bad_indices = bad_indices_cpu.tolist()
        logit_diff_mean, logit_diff_max = diff_stats_cpu.tolist()

        # Extract worst values (already indexed, just need CPU transfer)
        if bad_indices:
            worst_values_cpu = ratio[bad_indices_tensor].cpu()
            worst_actions_cpu = actions[bad_indices_tensor].cpu()
            worst_values = worst_values_cpu.tolist()
            worst_actions = worst_actions_cpu.tolist()
        else:
            worst_values = []
            worst_actions = []

        return cls(
            worst_ratio_indices=bad_indices,
            worst_ratio_values=worst_values,
            worst_ratio_actions=worst_actions,
            num_bad_total=num_bad_total,
            num_nonfinite=num_nonfinite,
            logit_diff_mean=logit_diff_mean,
            logit_diff_max=logit_diff_max,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)


__all__ = [
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
    "RatioExplosionDiagnostic",
]
