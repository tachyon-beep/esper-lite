"""Lightweight Gradient Collector for Seed Telemetry.

Collects gradient statistics for seed parameters without the full
overhead of DiagnosticTracker. Designed for per-epoch collection
during comparison and training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator

import torch
import torch.nn as nn


@dataclass(slots=True)
class GradientHealthMetrics:
    """Enhanced gradient health metrics for Ops Normal monitoring.

    Extends basic gradient stats with layer-wise information
    and numerical stability indicators.
    """

    # Basic stats (existing)
    gradient_norm: float
    gradient_health: float
    has_vanishing: bool
    has_exploding: bool

    # Layer-wise summary
    min_layer_norm: float
    max_layer_norm: float
    norm_ratio: float  # max/min - high ratio indicates imbalance

    # Quality indicators
    zero_grad_fraction: float
    nan_count: int
    inf_count: int

    def is_healthy(
        self,
        max_norm_ratio: float = 1000.0,
        max_zero_fraction: float = 0.5,
    ) -> bool:
        """Check if gradients indicate healthy training.

        Returns:
            True if gradients are healthy
        """
        return (
            self.nan_count == 0
            and self.inf_count == 0
            and self.norm_ratio < max_norm_ratio
            and self.zero_grad_fraction < max_zero_fraction
            and not self.has_exploding
        )

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


class SeedGradientCollector:
    """Lightweight gradient statistics collector for seed telemetry.

    Unlike DiagnosticTracker, this collector:
    - Does not use hooks (called explicitly after backward)
    - Computes only essential stats (norm, health, vanishing, exploding)
    - Is stateless (no history)
    - Uses vectorized operations for performance
    """

    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
    ):
        """Initialize collector with detection thresholds.

        Args:
            vanishing_threshold: Gradient norm below this is considered vanishing
            exploding_threshold: Gradient norm above this is considered exploding
        """
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

    def collect(self, parameters: Iterator[nn.Parameter]) -> dict:
        """Collect gradient statistics from parameters (sync version).

        Call this after loss.backward() to gather gradient stats.
        WARNING: Calls .item() which forces CPU-GPU sync. Use collect_async()
        inside CUDA stream contexts to avoid blocking.

        Args:
            parameters: Iterator of parameters (e.g., model.parameters())

        Returns:
            Dict with keys: gradient_norm, gradient_health, has_vanishing, has_exploding
        """
        async_stats = self.collect_async(parameters)
        return materialize_grad_stats(async_stats)

    def collect_async(self, parameters: Iterator[nn.Parameter]) -> dict:
        """Collect gradient statistics as tensors (async-safe version).

        Returns tensors instead of floats to avoid .item() sync inside CUDA streams.
        Call materialize_grad_stats() AFTER stream.synchronize() to get final values.

        Args:
            parameters: Iterator of parameters (e.g., model.parameters())

        Returns:
            Dict with tensor values (call materialize_grad_stats to convert to Python types)
        """
        # Filter params with grads
        grads = [p.grad for p in parameters if p.grad is not None]

        if not grads:
            return {
                '_empty': True,
                'gradient_norm': 0.0,
                'gradient_health': 1.0,
                'has_vanishing': False,
                'has_exploding': False,
            }

        # [PyTorch 2.9] Use _foreach_norm for efficient multi-tensor norm computation
        # This is a fused CUDA kernel that computes all norms in a single kernel launch,
        # avoiding Python iteration overhead. Used internally by clip_grad_norm_.
        per_param_norms = torch._foreach_norm(grads, ord=2)

        # Stack for vectorized comparisons
        all_norms = torch.stack(per_param_norms)
        n_grads = len(grads)

        # Total norm via Pythagorean theorem (avoids large tensor allocation)
        total_squared_norm = (all_norms ** 2).sum()

        return {
            '_empty': False,
            '_n_grads': n_grads,
            '_total_squared_norm': total_squared_norm,
            '_all_norms': all_norms,
            '_n_vanishing': (all_norms < self.vanishing_threshold).sum(),
            '_n_exploding': (all_norms > self.exploding_threshold).sum(),
        }


def materialize_grad_stats(async_stats: dict) -> dict:
    """Convert async gradient stats tensors to Python values.

    Call this AFTER stream.synchronize() to safely extract .item() values.

    Args:
        async_stats: Dict from collect_async() with tensor values

    Returns:
        Dict with Python float/bool values ready for telemetry
    """
    if async_stats.get('_empty', False):
        # Already materialized (empty case returns Python values directly)
        return {
            'gradient_norm': async_stats['gradient_norm'],
            'gradient_health': async_stats['gradient_health'],
            'has_vanishing': async_stats['has_vanishing'],
            'has_exploding': async_stats['has_exploding'],
        }

    # Extract tensor values - safe to call .item() after stream sync
    n_grads = async_stats['_n_grads']
    total_squared_norm = async_stats['_total_squared_norm'].item()
    n_vanishing = async_stats['_n_vanishing'].item()
    n_exploding = async_stats['_n_exploding'].item()

    # Compute derived statistics
    gradient_norm = (total_squared_norm ** 0.5) / n_grads

    # Compute health score (0-1, higher is healthier)
    vanishing_ratio = n_vanishing / n_grads
    exploding_ratio = n_exploding / n_grads

    health = 1.0
    health -= vanishing_ratio * 0.5  # Penalize vanishing
    health -= exploding_ratio * 0.8  # Penalize exploding more
    health = max(0.0, min(1.0, health))

    return {
        'gradient_norm': gradient_norm,
        'gradient_health': health,
        'has_vanishing': n_vanishing > 0,
        'has_exploding': n_exploding > 0,
    }


def collect_seed_gradients(
    seed_parameters: Iterator[nn.Parameter],
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 100.0,
    return_enhanced: bool = False,
) -> dict | GradientHealthMetrics:
    """Convenience function to collect gradient stats.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection
        return_enhanced: If True, return GradientHealthMetrics dataclass

    Returns:
        Dict with gradient statistics, or GradientHealthMetrics if return_enhanced
    """
    # Convert to list to allow multiple passes
    params = list(seed_parameters)
    grads = [p.grad for p in params if p.grad is not None]

    if not grads:
        if return_enhanced:
            return GradientHealthMetrics(
                gradient_norm=0.0,
                gradient_health=1.0,
                has_vanishing=False,
                has_exploding=False,
                min_layer_norm=0.0,
                max_layer_norm=0.0,
                norm_ratio=1.0,
                zero_grad_fraction=0.0,
                nan_count=0,
                inf_count=0,
            )
        return {
            'gradient_norm': 0.0,
            'gradient_health': 1.0,
            'has_vanishing': False,
            'has_exploding': False,
        }

    # Compute per-layer norms
    per_layer_norms = [g.norm(2).item() for g in grads]
    n_grads = len(grads)

    # Aggregate stats
    total_norm = sum(n**2 for n in per_layer_norms) ** 0.5
    avg_norm = total_norm / n_grads

    min_norm = min(per_layer_norms)
    max_norm = max(per_layer_norms)
    norm_ratio = max_norm / max(min_norm, 1e-10)

    # Count vanishing/exploding
    n_vanishing = sum(1 for n in per_layer_norms if n < vanishing_threshold)
    n_exploding = sum(1 for n in per_layer_norms if n > exploding_threshold)

    # Quality checks
    all_grads = torch.cat([g.view(-1) for g in grads])
    zero_fraction = (all_grads == 0).float().mean().item()
    nan_count = torch.isnan(all_grads).sum().item()
    inf_count = torch.isinf(all_grads).sum().item()

    # Health score
    vanishing_ratio = n_vanishing / n_grads
    exploding_ratio = n_exploding / n_grads
    health = 1.0
    health -= vanishing_ratio * 0.5
    health -= exploding_ratio * 0.8
    health = max(0.0, min(1.0, health))

    if return_enhanced:
        return GradientHealthMetrics(
            gradient_norm=avg_norm,
            gradient_health=health,
            has_vanishing=n_vanishing > 0,
            has_exploding=n_exploding > 0,
            min_layer_norm=min_norm,
            max_layer_norm=max_norm,
            norm_ratio=norm_ratio,
            zero_grad_fraction=zero_fraction,
            nan_count=int(nan_count),
            inf_count=int(inf_count),
        )

    return {
        'gradient_norm': avg_norm,
        'gradient_health': health,
        'has_vanishing': n_vanishing > 0,
        'has_exploding': n_exploding > 0,
    }


def collect_seed_gradients_async(
    seed_parameters: Iterator[nn.Parameter],
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 100.0,
) -> dict:
    """Collect gradient stats as tensors (async-safe version).

    Returns tensors instead of floats to avoid .item() sync inside CUDA streams.
    Call materialize_grad_stats() AFTER stream.synchronize() to get final values.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection

    Returns:
        Dict with tensor values (call materialize_grad_stats to convert to Python types)
    """
    collector = SeedGradientCollector(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )
    return collector.collect_async(seed_parameters)


# =============================================================================
# Dual Gradient Collection (Host + Seed) for G2 Gate
# =============================================================================

@dataclass(slots=True)
class DualGradientStats:
    """Gradient statistics for both host and seed networks.

    Used to compute the seed_gradient_norm_ratio metric for the G2 gate,
    which detects if a seed is actively learning vs. riding host improvements.
    """
    host_grad_norm: float
    host_param_count: int
    seed_grad_norm: float
    seed_param_count: int

    @property
    def normalized_ratio(self) -> float:
        """Parameter-normalized gradient ratio (seed intensity / host intensity).

        Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)

        This normalizes by sqrt(param_count) because for i.i.d. gradient elements
        with variance σ², the expected L2 norm is σ√n. Normalizing removes the
        parameter count bias so small seeds aren't penalized for having fewer params.

        Returns:
            Ratio >= 0. Values around 0.05-0.2 indicate healthy seed activity.
            Values < 0.01 suggest the seed is dormant or riding host improvements.
        """
        eps = 1e-8
        if self.host_grad_norm < eps or self.seed_grad_norm < eps:
            return 0.0
        if self.host_param_count == 0 or self.seed_param_count == 0:
            return 0.0

        # Normalize by sqrt(param_count) to remove scale bias
        seed_intensity = self.seed_grad_norm / (self.seed_param_count ** 0.5)
        host_intensity = self.host_grad_norm / (self.host_param_count ** 0.5)

        return seed_intensity / (host_intensity + eps)


def collect_dual_gradients_async(
    host_parameters: Iterator[nn.Parameter],
    seed_parameters: Iterator[nn.Parameter],
) -> dict:
    """Collect gradient norms for both host and seed (async-safe).

    Call this after loss.backward() but before optimizer.step() to capture
    gradient norms from both networks in the same training step.

    Args:
        host_parameters: Iterator of host network parameters
        seed_parameters: Iterator of seed module parameters

    Returns:
        Dict with tensor values; call materialize_dual_grad_stats() after
        stream.synchronize() to get DualGradientStats.
    """
    host_grads = [p.grad for p in host_parameters if p.grad is not None]
    seed_grads = [p.grad for p in seed_parameters if p.grad is not None]

    result = {'_dual': True}

    # Host gradients
    if host_grads:
        host_norms = torch._foreach_norm(host_grads, ord=2)
        # Sum of squared norms for total norm via Pythagorean theorem
        # Use torch.stack to keep in tensor domain - avoids Python loop serialization
        result['_host_squared_sum'] = torch.stack(host_norms).pow(2).sum()
        result['_host_param_count'] = sum(g.numel() for g in host_grads)
    else:
        result['_host_squared_sum'] = 0.0
        result['_host_param_count'] = 0

    # Seed gradients
    if seed_grads:
        seed_norms = torch._foreach_norm(seed_grads, ord=2)
        # Use torch.stack to keep in tensor domain - avoids Python loop serialization
        result['_seed_squared_sum'] = torch.stack(seed_norms).pow(2).sum()
        result['_seed_param_count'] = sum(g.numel() for g in seed_grads)
    else:
        result['_seed_squared_sum'] = 0.0
        result['_seed_param_count'] = 0

    return result


def materialize_dual_grad_stats(async_stats: dict) -> DualGradientStats:
    """Convert async dual gradient stats to DualGradientStats.

    Call this AFTER stream.synchronize() to safely extract .item() values.

    Args:
        async_stats: Dict from collect_dual_gradients_async()

    Returns:
        DualGradientStats with computed norms and param counts
    """
    host_squared = async_stats['_host_squared_sum']
    seed_squared = async_stats['_seed_squared_sum']

    # Handle tensor vs float (depends on whether collected in stream)
    if isinstance(host_squared, torch.Tensor):
        host_squared = host_squared.item()
    if isinstance(seed_squared, torch.Tensor):
        seed_squared = seed_squared.item()

    return DualGradientStats(
        host_grad_norm=host_squared ** 0.5,
        host_param_count=async_stats['_host_param_count'],
        seed_grad_norm=seed_squared ** 0.5,
        seed_param_count=async_stats['_seed_param_count'],
    )
