"""Kasmina Isolation - Gradient isolation and health monitoring.

Ensures seed modules don't destabilize the host network during training.
Provides gradient health metrics for G2 gate decisions.
"""

from __future__ import annotations


import torch
import torch.nn as nn


# Numerical stability constant for gradient ratio computation
GRAD_RATIO_EPSILON: float = 1e-8


# =============================================================================
# Alpha Blending
# =============================================================================

def blend_with_isolation(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: torch.Tensor,
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

    Args:
        host_features: Host network features
        seed_features: Seed module features
        alpha: Blending weight as tensor (must match device/dtype of features).
            MUST be a tensor (not scalar) for torch.compile compatibility.
    """
    # torch.lerp is a fused operation: lerp(a, b, w) = a + w * (b - a)
    # Clamp alpha to [0, 1] for safety
    # Use Tensor.clamp() method for torch.compile compatibility with 0-dim tensors
    alpha = alpha.clamp(0.0, 1.0)
    return torch.lerp(host_features, seed_features, alpha)


def ste_forward(host_features: torch.Tensor, seed_features: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator forward pass.

    Forward: returns host_features (seed contribution cancels out)
    Backward: gradients flow to both host and seed parameters

    This is torch.compile friendly - pure tensor operations, no control flow.

    Note:
        When using channels_last memory format, callers must ensure host_features
        is contiguous BEFORE calling this function. See BUG-005 and the workaround
        in SeedSlot.forward(). The issue is a PyTorch segfault when backward()
        encounters channels_last tensors with the STE + detach combination.
    """
    return host_features + (seed_features - seed_features.detach())


# =============================================================================
# Gradient Health Monitor
# =============================================================================

class GradientHealthMonitor:
    """Monitors gradient health for G2 gate decisions.

    NOTE: Gradient isolation is enforced STRUCTURALLY via detach() at the
    seed input boundary, not via numeric threshold detection. This monitor
    reports gradient norms for health assessment (G2 gate) and debugging,
    NOT for isolation violation detection.

    The isolation guarantee is:
        "No gradients from seed path flow into host parameters"

    This is achieved by detaching host_features before passing to seed:
        seed_input = host_features.detach()  # Structural guarantee

    Host gradients from the direct loss path (host → loss) are EXPECTED
    and do NOT indicate isolation failures.
    """

    def __init__(self):
        self.host_grad_norm: float = 0.0
        self.seed_grad_norm: float = 0.0
        self._host_params: list[nn.Parameter] = []
        self._seed_params: list[nn.Parameter] = []

    def register(self, host: nn.Module, seed: nn.Module) -> None:
        """Register host and seed modules for monitoring."""
        self._host_params = [p for p in host.parameters() if p.requires_grad]
        self._seed_params = [p for p in seed.parameters() if p.requires_grad]

    @torch.no_grad()
    def compute_gradient_health(self) -> dict:
        """Compute gradient health metrics (sync version).

        WARNING: Calls .item() which forces CPU-GPU sync. Use compute_gradient_health_async()
        inside CUDA stream contexts to avoid blocking.

        Returns:
            Dict with gradient norms and ratio for G2 gate decisions.
        """
        async_stats = self.compute_gradient_health_async()
        return self.materialize_gradient_stats(async_stats)

    @torch.no_grad()
    def compute_gradient_health_async(self) -> dict:
        """Compute gradient health returning tensors (async-safe, no .item() sync).

        Uses torch._foreach_norm for batched norm computation - O(1) CUDA kernel
        launches instead of O(n_params). Returns tensors to avoid .item() sync.

        Call materialize_gradient_stats() AFTER stream.synchronize() to get
        final values.

        Returns:
            Dict with tensor values; use materialize_gradient_stats() to convert.
        """
        host_grads = [p.grad for p in self._host_params if p.grad is not None]
        seed_grads = [p.grad for p in self._seed_params if p.grad is not None]

        result = {'_async': True}

        if host_grads:
            # torch._foreach_norm returns list of norms per tensor
            # NOTE: torch._foreach_norm is a private API but stable since PyTorch 1.9.
            # If removed in future PyTorch, fallback: torch.stack([p.norm() for p in grads])
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

    def materialize_gradient_stats(self, async_stats: dict) -> dict:
        """Convert async gradient stats to final values.

        Call this AFTER stream.synchronize() to safely extract .item() values.

        Args:
            async_stats: Dict from compute_gradient_health_async()

        Returns:
            Dict with gradient norms and ratio for G2 gate decisions.
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

        # Compute gradient ratio: seed_norm / (host_norm + eps)
        # This ratio is used by G2 gate to assess seed gradient health
        ratio = seed_norm / (host_norm + GRAD_RATIO_EPSILON) if host_norm > 0 else 0.0

        return {
            "host_grad_norm": host_norm,
            "seed_grad_norm": seed_norm,
            "seed_gradient_ratio": ratio,
        }

    def reset(self) -> None:
        """Clear cached values and parameter references."""
        self.host_grad_norm = 0.0
        self.seed_grad_norm = 0.0
        self._host_params.clear()
        self._seed_params.clear()


__all__ = [
    "blend_with_isolation",
    "ste_forward",
    "GradientHealthMonitor",
]
