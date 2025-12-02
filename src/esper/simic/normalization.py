"""Observation normalization for RL training.

Provides running mean/std normalization using Welford's numerically stable
online algorithm. Used by vectorized PPO for observation preprocessing.

GPU-native: All operations stay on the same device as input tensors,
avoiding costly CPU synchronization in the training loop.
"""

from __future__ import annotations

import torch


class RunningMeanStd:
    """Running mean and std for observation normalization.

    Uses Welford's online algorithm for numerical stability.
    GPU-native: automatically moves stats to match input device.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4, device: str = "cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        # Use tensor for count to keep all ops on device (avoids CPU sync)
        self.count = torch.tensor(epsilon, device=device)
        self.epsilon = epsilon
        self._device = device

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running stats with new batch of observations.

        GPU-native: operates entirely on the input tensor's device,
        avoiding CPU synchronization overhead.
        """
        # Auto-migrate stats to input device (one-time cost)
        if self.mean.device != x.device:
            self.to(x.device)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor,
                             batch_count: int) -> None:
        """Update using batch moments (Welford's algorithm)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize observation using running stats.

        Note: After auto-migration in update(), mean/var are on correct device.
        """
        return torch.clamp(
            (x - self.mean) / torch.sqrt(self.var + self.epsilon),
            -clip, clip
        )

    def to(self, device: str | torch.device) -> "RunningMeanStd":
        """Move stats to device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        self._device = str(device)
        return self

    @property
    def device(self) -> torch.device:
        """Current device of the stats."""
        return self.mean.device


__all__ = ["RunningMeanStd"]
