"""Lightweight Gradient Collector for Seed Telemetry.

Collects gradient statistics for seed parameters without the full
overhead of DiagnosticTracker. Designed for per-epoch collection
during comparison and training loops.

Uses vectorized operations (torch._foreach_norm) for performance
in multi-seed scenarios.
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
        """Collect gradient statistics from parameters.

        Call this after loss.backward() to gather gradient stats.

        Args:
            parameters: Iterator of parameters (e.g., model.parameters())

        Returns:
            Dict with keys: gradient_norm, gradient_health, has_vanishing, has_exploding
        """
        # Filter params with grads
        grads = [p.grad for p in parameters if p.grad is not None]

        if not grads:
            return {
                'gradient_norm': 0.0,
                'gradient_health': 1.0,
                'has_vanishing': False,
                'has_exploding': False,
            }

        # Vectorized Norm Calculation using _foreach_norm
        # Computes L2 norm for all tensors in the list at once
        # Note: _foreach_norm returns a list of scalar tensors
        per_param_norms = torch._foreach_norm(grads, 2.0)

        # Stack to compute stats efficiently on GPU/CPU
        all_norms = torch.stack(per_param_norms)

        # Compute Statistics
        total_squared_norm = torch.sum(all_norms ** 2).item()
        gradient_norm = (total_squared_norm ** 0.5) / len(grads)

        # Health Checks (vectorized)
        n_vanishing = torch.sum(all_norms < self.vanishing_threshold).item()
        n_exploding = torch.sum(all_norms > self.exploding_threshold).item()

        # Compute health score (0-1, higher is healthier)
        # Penalize vanishing/exploding gradients
        vanishing_ratio = n_vanishing / len(grads)
        exploding_ratio = n_exploding / len(grads)

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
    """Convenience function to collect gradient stats.

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
