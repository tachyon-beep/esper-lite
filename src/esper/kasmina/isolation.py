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
    detach_host: bool = True,
) -> torch.Tensor:
    """Blend host and seed features with optional gradient isolation on host path."""
    host_path = host_features.detach() if detach_host else host_features
    # torch.lerp is a fused operation: lerp(a, b, w) = a + w * (b - a)
    # Clamp alpha to [0, 1] for safety
    return torch.lerp(host_path, seed_features, max(0.0, min(1.0, alpha)))


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

    def check_isolation(self) -> tuple[bool, dict]:
        """Check if gradient isolation is maintained."""
        host_norm = 0.0
        seed_norm = 0.0

        for p in self._host_params:
            if p.grad is not None:
                host_norm += p.grad.norm().item() ** 2

        for p in self._seed_params:
            if p.grad is not None:
                seed_norm += p.grad.norm().item() ** 2

        host_norm = host_norm ** 0.5
        seed_norm = seed_norm ** 0.5

        self.host_grad_norm = host_norm
        self.seed_grad_norm = seed_norm

        is_isolated = host_norm < self.threshold

        if not is_isolated:
            self.violations += 1

        return is_isolated, {
            "host_grad_norm": host_norm,
            "seed_grad_norm": seed_norm,
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
    "GradientIsolationMonitor",
]
