"""Kasmina Isolation - Gradient isolation and blending.

Ensures seed modules don't destabilize the host network during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# =============================================================================
# Alpha Blending
# =============================================================================

class AlphaSchedule:
    """Sigmoid-based alpha schedule for smooth blending."""

    def __init__(
        self,
        total_steps: int,
        start: float = 0.0,
        end: float = 1.0,
        temperature: float = 1.0,
    ):
        self.total_steps = max(1, total_steps)
        self.start = start
        self.end = end
        self.temperature = max(temperature, 1e-6)

    def __call__(self, step: int) -> float:
        """Get alpha value at given step."""
        if step <= 0:
            return self.start
        if step >= self.total_steps:
            return self.end

        midpoint = self.total_steps / 2
        scaled = (step - midpoint) / self.temperature
        sigmoid = 0.5 * (1.0 + math.tanh(scaled * 0.5))

        return self.start + (self.end - self.start) * sigmoid


def blend_with_isolation(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend host and seed features with proper gradient flow.

    Gradient attribution:
        d_output/d_host_features = (1 - alpha)
        d_output/d_seed_features = alpha

    Both host and seed receive gradients proportional to their contribution.
    This enables the host backbone to continue learning during BLENDING+.

    Gradient flow diagram:
        Loss
          │
          ▼
        Output = lerp(host, seed, α)
          │                    │
          │ (1-α) gradient     │ α gradient
          ▼                    ▼
        Host Features         Seed Features
          ▲                    │
          │ (blocked if        │
          │  isolate_gradients)│
          └────────────────────┘

    To control seed→host gradient flow (the indirect path), use
    isolate_gradients at the SEED INPUT, not here. This function
    should always allow gradients to both direct inputs.
    """
    # torch.lerp is a fused operation: lerp(a, b, w) = a + w * (b - a)
    # Clamp alpha to [0, 1] for safety
    return torch.lerp(host_features, seed_features, max(0.0, min(1.0, alpha)))


def ste_forward(host_features: torch.Tensor, seed_features: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator forward pass.

    Forward: returns host_features (seed contribution cancels out)
    Backward: gradients flow to both host and seed parameters

    This is torch.compile friendly - pure tensor operations, no control flow.
    """
    return host_features + (seed_features - seed_features.detach())


# =============================================================================
# Gradient Isolation Monitor
# =============================================================================

class GradientIsolationMonitor:
    """Monitors gradient flow to verify isolation between host and seed."""

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.host_grad_norm: float = 0.0
        self.seed_grad_norm: float = 0.0
        self.violations: int = 0
        self._host_params: list[nn.Parameter] = []
        self._seed_params: list[nn.Parameter] = []

    def register(self, host: nn.Module, seed: nn.Module) -> None:
        """Register host and seed modules for monitoring."""
        self._host_params = [p for p in host.parameters() if p.requires_grad]
        self._seed_params = [p for p in seed.parameters() if p.requires_grad]

    @torch.no_grad()
    def check_isolation(self) -> tuple[bool, dict]:
        """Check if gradient isolation is maintained (sync version).

        WARNING: Calls .item() which forces CPU-GPU sync. Use check_isolation_async()
        inside CUDA stream contexts to avoid blocking.
        """
        async_stats = self.check_isolation_async()
        return self.materialize_isolation_stats(async_stats)

    @torch.no_grad()
    def check_isolation_async(self) -> dict:
        """Check isolation returning tensors (async-safe, no .item() sync).

        Uses torch._foreach_norm for batched norm computation - O(1) CUDA kernel
        launches instead of O(n_params). Returns tensors to avoid .item() sync.

        Call materialize_isolation_stats() AFTER stream.synchronize() to get
        final values.

        Returns:
            Dict with tensor values; use materialize_isolation_stats() to convert.
        """
        host_grads = [p.grad for p in self._host_params if p.grad is not None]
        seed_grads = [p.grad for p in self._seed_params if p.grad is not None]

        result = {'_async': True}

        if host_grads:
            # torch._foreach_norm returns list of norms per tensor
            norms = torch._foreach_norm(host_grads)
            # Sum of squared norms for total norm via Pythagorean theorem
            result['_host_norm_sq'] = torch.stack(norms).pow(2).sum()
        else:
            result['_host_norm_sq'] = None  # Distinguishes "no grads" from "zero grads"

        if seed_grads:
            norms = torch._foreach_norm(seed_grads)
            result['_seed_norm_sq'] = torch.stack(norms).pow(2).sum()
        else:
            result['_seed_norm_sq'] = None

        return result

    def materialize_isolation_stats(self, async_stats: dict) -> tuple[bool, dict]:
        """Convert async isolation stats to final values.

        Call this AFTER stream.synchronize() to safely extract .item() values.

        Args:
            async_stats: Dict from check_isolation_async()

        Returns:
            Tuple of (is_isolated, stats_dict) matching check_isolation() signature
        """
        host_sq = async_stats['_host_norm_sq']
        seed_sq = async_stats['_seed_norm_sq']

        # Convert tensors to Python floats
        if host_sq is None:
            host_norm = 0.0
        elif isinstance(host_sq, torch.Tensor):
            host_norm = host_sq.item() ** 0.5
        else:
            host_norm = host_sq ** 0.5

        if seed_sq is None:
            seed_norm = 0.0
        elif isinstance(seed_sq, torch.Tensor):
            seed_norm = seed_sq.item() ** 0.5
        else:
            seed_norm = seed_sq ** 0.5

        self.host_grad_norm = host_norm
        self.seed_grad_norm = seed_norm

        is_isolated = host_norm < self.threshold

        if not is_isolated:
            self.violations += 1

        # Compute gradient ratio: seed_norm / (host_norm + eps)
        ratio = seed_norm / (host_norm + 1e-8) if host_norm > 0 else 0.0

        return is_isolated, {
            "host_grad_norm": host_norm,
            "seed_grad_norm": seed_norm,
            "seed_gradient_ratio": ratio,
            "isolated": is_isolated,
            "violations": self.violations,
        }

    def reset(self) -> None:
        """Reset violation counter and clear parameter references."""
        self.violations = 0
        self.host_grad_norm = 0.0
        self.seed_grad_norm = 0.0
        self._host_params.clear()
        self._seed_params.clear()


__all__ = [
    "AlphaSchedule",
    "blend_with_isolation",
    "ste_forward",
    "GradientIsolationMonitor",
]
