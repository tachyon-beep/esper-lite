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

    Uses Welford's online algorithm for numerical stability by default.
    Optionally uses EMA (exponential moving average) for slower adaptation
    during long training runs to prevent distribution shift.

    GPU-native: automatically moves stats to match input device.

    Args:
        shape: Shape of observations to normalize
        epsilon: Small constant for numerical stability
        device: Device to store stats on
        momentum: If None, uses Welford's algorithm (full history weighting).
            If set (e.g., 0.99), uses EMA for slower adaptation:
            new_stat = momentum * old_stat + (1 - momentum) * batch_stat
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        epsilon: float = 1e-4,
        device: str = "cpu",
        momentum: float | None = None,
    ):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        # Use tensor for count to keep all ops on device (avoids CPU sync)
        self.count = torch.tensor(epsilon, device=device)
        self.epsilon = epsilon
        self._device = device
        self.momentum = momentum

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
        """Update using batch moments.

        Uses either Welford's algorithm (momentum=None) or EMA (momentum set).
        EMA provides slower adaptation for long training stability.
        """
        if self.momentum is not None:
            # EMA update: slower adaptation to prevent distribution shift
            # during long training runs
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            # Count still tracked for diagnostics but not used in EMA
            self.count = self.count + batch_count
        else:
            # Welford's online algorithm: full history weighting
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

        GPU-native: auto-migrates stats to input device if needed.
        """
        # Auto-migrate stats to input device (one-time cost)
        if self.mean.device != x.device:
            self.to(x.device)

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


class RewardNormalizer:
    """Running reward normalization for critic stability.

    Normalizes rewards by dividing by running std (NOT subtracting mean).
    Essential when reward magnitudes can vary wildly (e.g., ransomware fix).

    Why std-only (no mean subtraction):
    - The critic learns E[R] through its value function target
    - Subtracting running mean from rewards creates non-stationary targets
    - When mean shifts, the critic must constantly recalibrate
    - Dividing by std only preserves reward semantics while stabilizing magnitudes

    Uses Welford's online algorithm where:
    - mean: running mean (tracked for variance computation only)
    - m2: sum of squared deviations from the current mean (NOT variance)
    - variance = m2 / (count - 1) for sample variance

    Usage:
        normalizer = RewardNormalizer(clip=10.0)
        normalized_reward = normalizer.update_and_normalize(raw_reward)
    """

    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared deviations (Welford's M2)
        self.count = 0  # Start at 0, not epsilon
        self.clip = clip
        self.epsilon = epsilon

    def update_and_normalize(self, reward: float) -> float:
        """Update running stats and return normalized reward.

        Uses Welford's online algorithm for numerical stability.
        Returns reward / std (no mean subtraction for critic stability).

        For the first sample, returns clipped raw reward since variance
        cannot be computed from a single sample.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2

        # Need at least 2 samples to compute sample variance
        if self.count < 2:
            return max(-self.clip, min(self.clip, reward))

        # Normalize by std only (no mean subtraction)
        # variance = m2 / (count - 1) for sample variance
        std = max(self.epsilon, (self.m2 / (self.count - 1)) ** 0.5)
        normalized = reward / std
        return max(-self.clip, min(self.clip, normalized))

    def normalize_only(self, reward: float) -> float:
        """Normalize without updating stats (for evaluation)."""
        if self.count < 2:
            return max(-self.clip, min(self.clip, reward))
        std = max(self.epsilon, (self.m2 / (self.count - 1)) ** 0.5)
        normalized = reward / std
        return max(-self.clip, min(self.clip, normalized))


__all__ = ["RunningMeanStd", "RewardNormalizer"]
