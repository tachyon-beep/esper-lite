"""Lightweight Gradient Collector for Seed Telemetry.

Collects gradient statistics for seed parameters without the full
overhead of DiagnosticTracker. Designed for per-epoch collection
during comparison and training loops.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn


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

        # Compute L2 norm for each gradient tensor
        # Using public API (torch.norm) instead of private torch._foreach_norm
        per_param_norms = [g.norm(2) for g in grads]

        # Stack to compute stats efficiently on GPU/CPU
        all_norms = torch.stack(per_param_norms)
        n_grads = len(grads)

        # Compute Statistics - keep as tensors!
        total_squared_norm = torch.sum(all_norms ** 2)

        # Health Checks (vectorized) - keep as tensors!
        n_vanishing = torch.sum(all_norms < self.vanishing_threshold)
        n_exploding = torch.sum(all_norms > self.exploding_threshold)

        return {
            '_empty': False,
            '_n_grads': n_grads,
            '_total_squared_norm': total_squared_norm,
            '_n_vanishing': n_vanishing,
            '_n_exploding': n_exploding,
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
) -> dict:
    """Convenience function to collect gradient stats (sync version).

    WARNING: Calls .item() which forces CPU-GPU sync. Use collect_seed_gradients_async()
    inside CUDA stream contexts to avoid blocking.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection

    Returns:
        Dict with gradient statistics
    """
    collector = SeedGradientCollector(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )
    return collector.collect(seed_parameters)


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
