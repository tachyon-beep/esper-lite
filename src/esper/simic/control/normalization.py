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

    Thread Safety (H13):
        This class is NOT thread-safe. The update() and normalize() methods
        modify internal state (mean, var, count) without locks. Concurrent
        calls from multiple threads will cause data races.

        In vectorized training (simic/training/vectorized.py), this is safe
        because each ParallelEnvState has its own RunningMeanStd instance.
        Per-env normalizers avoid cross-env state sharing.

        If you need to share a normalizer across threads:
        - Use external locking (threading.Lock) around update() calls
        - Or use torch.nn.SyncBatchNorm for distributed training

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

    @torch.inference_mode()
    def update(self, x: torch.Tensor) -> None:
        """Update running stats with new batch of observations.

        GPU-native: operates entirely on the input tensor's device,
        avoiding CPU synchronization overhead.
        """
        # Auto-migrate stats to input device (one-time synchronous cost).
        # WARNING: This triggers a GPU sync when stats move CPU→GPU or between GPUs.
        # For best performance, initialize RunningMeanStd on the target device.
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
            # during long training runs.
            #
            # Law of total variance: when combining distributions with different means,
            # the combined variance includes a cross-term for between-group variance:
            #   Var_combined = w1*Var1 + w2*Var2 + w1*w2*(Mean1 - Mean2)^2
            #
            # For EMA with weights (momentum, 1-momentum):
            #   new_var = m*old_var + (1-m)*batch_var + m*(1-m)*(old_mean - batch_mean)^2
            #
            # The cross-term must be computed BEFORE updating the mean.
            delta = batch_mean - self.mean
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = (
                self.momentum * self.var
                + (1 - self.momentum) * batch_var
                + self.momentum * (1 - self.momentum) * delta ** 2
            )
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
        # Auto-migrate stats to input device (one-time synchronous cost).
        # WARNING: This triggers a GPU sync when stats move CPU→GPU or between GPUs.
        # For best performance, initialize RunningMeanStd on the target device.
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

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary for checkpointing."""
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from dictionary."""
        self.mean = state["mean"].to(self._device)
        self.var = state["var"].to(self._device)
        self.count = state["count"].to(self._device)

    def reset(self) -> None:
        """Reset running statistics to initial state.

        Call this after changing input transforms (e.g., adding symlog) to avoid
        stale statistics causing LSTM saturation or other normalization issues.

        The normalizer will re-learn statistics from scratch on subsequent updates.
        """
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.fill_(self.epsilon)


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
            return float(max(-self.clip, min(self.clip, reward)))

        # Normalize by std only (no mean subtraction)
        # variance = m2 / (count - 1) for sample variance
        std = max(self.epsilon, (self.m2 / (self.count - 1)) ** 0.5)
        normalized = reward / std
        return float(max(-self.clip, min(self.clip, normalized)))

    def normalize_only(self, reward: float) -> float:
        """Normalize without updating stats (for evaluation)."""
        if self.count < 2:
            return float(max(-self.clip, min(self.clip, reward)))
        std = max(self.epsilon, (self.m2 / (self.count - 1)) ** 0.5)
        normalized = reward / std
        return float(max(-self.clip, min(self.clip, normalized)))

    def state_dict(self) -> dict[str, float | int]:
        """Return state dictionary for checkpointing."""
        return {
            "mean": self.mean,
            "m2": self.m2,
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        """Load state from dictionary."""
        self.mean = state["mean"]
        self.m2 = state["m2"]
        self.count = int(state["count"])


class ValueNormalizer:
    """Running normalization for value function targets (PopArt-lite).

    Solves the scale mismatch bug in PPO where:
    - Critic is trained on normalized returns (std~1)
    - GAE uses raw critic outputs mixed with raw rewards

    This normalizer tracks running statistics and provides:
    - normalize(): Scale returns for critic training
    - denormalize(): Unscale critic outputs for GAE computation

    Key difference from full PopArt:
    - Full PopArt rescales the value head's final layer weights when stats change
    - This simplified version just tracks stats; the critic learns normalized outputs
    - We denormalize outputs at GAE time rather than rescaling weights

    This is sufficient when:
    - Return scale is relatively stable within training
    - Episodes are fixed-length (less distribution shift)

    For environments with wildly varying reward scales, consider full PopArt.

    GPU-native: All operations stay on the same device as input tensors.

    Args:
        epsilon: Small constant for numerical stability (default 1e-4)
        device: Device to store stats on (default "cpu")
        momentum: EMA momentum for stat updates (default 0.99).
            Higher values = slower adaptation = more stability.
            Use None for Welford's algorithm (full history weighting).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        device: str | torch.device = "cpu",
        momentum: float = 0.99,
    ):
        device = torch.device(device) if isinstance(device, str) else device
        # Track running mean and std of returns
        # Mean is tracked for proper variance computation but NOT subtracted
        # (same rationale as RewardNormalizer: preserve value semantics)
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.count = torch.tensor(0.0, device=device)
        self.epsilon = epsilon
        self._device = device
        self.momentum = momentum
        # Track if we've seen enough samples for reliable stats
        self._min_samples = 32  # Need this many samples before using running stats

    @property
    def std(self) -> torch.Tensor:
        """Current running std (clamped for stability)."""
        return torch.sqrt(self.var + self.epsilon)

    @property
    def has_valid_stats(self) -> bool:
        """Whether we have enough samples for reliable normalization."""
        return self.count.item() >= self._min_samples

    @torch.inference_mode()
    def update(self, returns: torch.Tensor) -> None:
        """Update running stats with a batch of returns.

        Args:
            returns: Tensor of return values [batch] or [batch, seq]
        """
        # Flatten and filter non-finite values
        flat = returns.flatten()
        finite_mask = torch.isfinite(flat)
        if not finite_mask.any():
            return

        valid = flat[finite_mask]
        batch_count = valid.numel()
        if batch_count == 0:
            return

        # Move stats to input device if needed
        if self.mean.device != valid.device:
            self.to(valid.device)

        batch_mean = valid.mean()
        batch_var = valid.var(correction=0)  # Population variance

        if self.momentum is not None and self.count.item() >= self._min_samples:
            # EMA update (after warmup)
            delta = batch_mean - self.mean
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = (
                self.momentum * self.var
                + (1 - self.momentum) * batch_var
                + self.momentum * (1 - self.momentum) * delta ** 2
            )
        else:
            # Welford's algorithm (during warmup or if momentum=None)
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
            new_var = m2 / tot_count.clamp(min=1)
            self.mean = new_mean
            self.var = new_var

        self.count = self.count + batch_count

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns for critic training.

        Returns returns / std (no mean subtraction to preserve semantics).
        During warmup, returns values unchanged.

        Args:
            returns: Return values to normalize

        Returns:
            Normalized returns (std~1) if stats valid, else unchanged
        """
        if not self.has_valid_stats:
            return returns

        if self.mean.device != returns.device:
            self.to(returns.device)

        return returns / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize critic outputs for GAE computation.

        Converts normalized-scale values back to raw reward scale.
        During warmup, returns values unchanged.

        Args:
            values: Critic outputs (on normalized scale)

        Returns:
            Values on raw reward scale
        """
        if not self.has_valid_stats:
            return values

        if self.mean.device != values.device:
            self.to(values.device)

        return values * self.std

    def get_scale(self) -> float:
        """Get current normalization scale (for telemetry).

        Returns 1.0 during warmup.
        """
        if not self.has_valid_stats:
            return 1.0
        return self.std.item()

    def to(self, device: str | torch.device) -> "ValueNormalizer":
        """Move stats to device."""
        device = torch.device(device) if isinstance(device, str) else device
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        self._device = device
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary for checkpointing."""
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from dictionary."""
        self.mean = state["mean"].to(self._device)
        self.var = state["var"].to(self._device)
        self.count = state["count"].to(self._device)


__all__ = ["RunningMeanStd", "RewardNormalizer", "ValueNormalizer"]
