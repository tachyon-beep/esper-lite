"""Kasmina blending / gating primitives.

Phase 2+ contract:
- Alpha *amplitude* scheduling is owned by `AlphaController` (scalar, time-based).
- `SeedSlot.alpha_schedule` is reserved for per-sample gating only (currently: `GatedBlend`).

`LinearBlend` and `SigmoidBlend` are retained as curve utilities for tests and
numerical characterization; SeedSlot does not instantiate them for runtime
blending.

TODO: [MAINTENANCE] - If we fully commit to AlphaController-only scheduling,
consider removing schedule-based BlendAlgorithm types and keeping only gating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import threading

import torch
import torch.nn as nn


class BlendAlgorithm(nn.Module, ABC):
    """Base class for blending algorithms.

    All implementations must provide `get_alpha_for_blend()` which returns
    a tensor suitable for broadcasting in the blend operation. This unified
    interface eliminates runtime type-checking at blend time.

    Inherits from nn.Module so all blend algorithms can be registered as
    submodules of SeedSlot (required for consistent PyTorch module handling).

    Note:
        The internal alpha tensor cache is not serialized. After loading
        a checkpoint, the cache will be repopulated on the first forward pass.
        This has no correctness impact, only a single-allocation performance cost.
    """

    algorithm_id: str = "base"
    _current_step: int = 0  # Tracked internally for schedule-based blends

    def __init__(self):
        # nn.Module must initialize first in the MRO so submodules register correctly.
        super().__init__()
        # Thread-local cache for alpha tensor to avoid per-forward allocation
        # Uses thread-local storage for multi-GPU DataParallel safety
        self._alpha_cache_local = threading.local()

    def _get_cached_alpha_tensor(self, value: float, x: torch.Tensor) -> torch.Tensor:
        """Get alpha tensor, using cache if possible.

        Cache is invalidated when device, dtype, or alpha value changes.
        This eliminates thousands of tensor allocations per episode.

        All three checks (device, dtype, value) are necessary because:
        - Device: DataParallel may use different GPUs across threads
        - Dtype: Mixed precision training may change dtypes dynamically
        - Value: Schedule-based blends change alpha each step
        """
        cache = getattr(self._alpha_cache_local, 'cache', None)
        if cache is not None:
            cached_device, cached_dtype, cached_value, cached_tensor = cache
            if cached_device == x.device and cached_dtype == x.dtype and cached_value == value:
                return cached_tensor

        # Create new tensor and cache it
        tensor = torch.tensor(value, device=x.device, dtype=x.dtype)
        self._alpha_cache_local.cache = (x.device, x.dtype, value, tensor)
        return tensor

    def step(self, step: int) -> None:
        """Update the current step for schedule-based algorithms."""
        self._current_step = step

    @abstractmethod
    def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
        """Get alpha tensor for blending.

        Args:
            x: Host features tensor. Shape depends on topology:
               - CNN: (B, C, H, W)
               - Transformer: (B, T, C)

        Returns:
            Alpha tensor broadcastable to x's shape. May be:
            - Scalar tensor for schedule-based blends (broadcasts to all)
            - Per-sample tensor for learned gated blends

        Note:
            Schedule-based algorithms (linear, sigmoid) ignore x and use
            the step set via step(). Gated algorithms use x to compute
            per-sample alpha.
        """
        pass

    def get_alpha(self, step: int) -> float:
        """Get scalar alpha for inspection/testing. Override in subclasses."""
        self._current_step = step
        return 0.5


class LinearBlend(BlendAlgorithm):
    """Linear ramp from 0 to 1 over total_steps."""

    algorithm_id = "linear"

    def __init__(self, total_steps: int = 5):
        super().__init__()
        self.total_steps = max(1, total_steps)

    def get_alpha(self, step: int) -> float:
        self._current_step = step
        return min(1.0, max(0.0, step / self.total_steps))

    def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar alpha as 0-dim tensor (broadcasts to any shape)."""
        alpha = min(1.0, max(0.0, self._current_step / self.total_steps))
        return self._get_cached_alpha_tensor(alpha, x)


class SigmoidBlend(BlendAlgorithm):
    """Sigmoid curve for smooth transitions."""

    algorithm_id = "sigmoid"

    def __init__(self, total_steps: int = 10, steepness: float = 1.0):
        super().__init__()
        self.total_steps = max(1, total_steps)
        self.steepness = steepness

    def _compute_alpha(self, step: int) -> float:
        """Compute sigmoid alpha for given step."""
        x = (step / self.total_steps - 0.5) * 12 * self.steepness
        return 1.0 / (1.0 + math.exp(-x))

    def get_alpha(self, step: int) -> float:
        self._current_step = step
        return self._compute_alpha(step)

    def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar alpha as 0-dim tensor (broadcasts to any shape)."""
        alpha = self._compute_alpha(self._current_step)
        return self._get_cached_alpha_tensor(alpha, x)


class GatedBlend(BlendAlgorithm):
    """Learned gating mechanism for adaptive blending.

    Supports both CNN (B, C, H, W) and transformer (B, T, C) inputs.
    The gate network operates on pooled (B, C) features.
    """

    algorithm_id = "gated"

    def __init__(self, channels: int, topology: str = "cnn", total_steps: int = 10):
        super().__init__()
        if channels <= 0:
            raise ValueError("GatedBlend requires channels > 0")
        if topology not in ("cnn", "transformer"):
            raise ValueError(f"Unknown topology '{topology}' for GatedBlend")
        self.topology = topology
        self.total_steps = max(1, total_steps)
        hidden_dim = max(1, channels // 4)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool to (B, C) regardless of topology."""
        if self.topology == "cnn":
            # (B, C, H, W) -> (B, C)
            return x.mean(dim=[2, 3])
        else:
            # (B, T, C) -> (B, C)
            return x.mean(dim=1)

    def get_alpha(self, step: int | None = None) -> float:
        """Return blending progress for lifecycle tracking.

        Unlike schedule-based blends, gated blending uses learned gates
        during forward(). For lifecycle/G3 gate compatibility, we report
        step-based progress: step / total_steps.

        This ensures G3 gate can pass naturally when blending completes.
        """
        if step is not None:
            self._current_step = step

        current = step if step is not None else self._current_step
        return min(1.0, current / self.total_steps)

    def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample alpha from input features."""
        pooled = self._pool_features(x)  # (B, C)
        alpha = self.gate(pooled)  # (B, 1)
        # Expand for broadcasting to match input shape
        if self.topology == "cnn":
            return alpha.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        else:
            return alpha.view(-1, 1, 1)  # (B, 1, 1)


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
