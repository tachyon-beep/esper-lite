"""Kasmina Blending Algorithms - Tamiyo's blending library.

Each algorithm defines how a seed's influence ramps from 0 to 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn


class BlendAlgorithm(ABC):
    """Base class for blending algorithms."""

    algorithm_id: str = "base"

    @abstractmethod
    def get_alpha(self, step: int) -> float:
        """Get alpha value for a given step."""
        pass

    def get_alpha_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Get alpha as tensor (for gated blends). Default: scalar."""
        raise NotImplementedError("Use get_alpha() for non-gated blends")


class LinearBlend(BlendAlgorithm):
    """Linear ramp from 0 to 1 over total_steps."""

    algorithm_id = "linear"

    def __init__(self, total_steps: int = 5):
        self.total_steps = max(1, total_steps)

    def get_alpha(self, step: int) -> float:
        return min(1.0, max(0.0, step / self.total_steps))


class SigmoidBlend(BlendAlgorithm):
    """Sigmoid curve for smooth transitions."""

    algorithm_id = "sigmoid"

    def __init__(self, total_steps: int = 10, steepness: float = 1.0):
        self.total_steps = max(1, total_steps)
        self.steepness = steepness

    def get_alpha(self, step: int) -> float:
        # Map step to [-6, 6] range for sigmoid
        x = (step / self.total_steps - 0.5) * 12 * self.steepness
        return 1.0 / (1.0 + math.exp(-x))


class GatedBlend(BlendAlgorithm, nn.Module):
    """Learned gating mechanism for adaptive blending."""

    algorithm_id = "gated"

    def __init__(self, channels: int):
        BlendAlgorithm.__init__(self)
        nn.Module.__init__(self)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )
        self._step = 0

    def get_alpha(self, step: int) -> float:
        self._step = step
        return 0.5  # Default; actual alpha comes from get_alpha_tensor

    def get_alpha_tensor(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(x)  # (B, 1)
        return alpha.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting


class BlendCatalog:
    """Registry of blending algorithms."""

    _algorithms: dict[str, type] = {
        "linear": LinearBlend,
        "sigmoid": SigmoidBlend,
        "gated": GatedBlend,
    }

    @classmethod
    def list_algorithms(cls) -> list[str]:
        return list(cls._algorithms.keys())

    @classmethod
    def create(cls, algorithm_id: str, **kwargs) -> BlendAlgorithm:
        if algorithm_id not in cls._algorithms:
            raise ValueError(f"Unknown blend algorithm: {algorithm_id}")
        return cls._algorithms[algorithm_id](**kwargs)


__all__ = [
    "BlendAlgorithm",
    "LinearBlend",
    "SigmoidBlend",
    "GatedBlend",
    "BlendCatalog",
]
