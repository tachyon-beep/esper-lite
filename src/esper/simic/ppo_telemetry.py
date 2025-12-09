"""PPO Telemetry Dataclasses.

Structured telemetry for PPO training diagnostics, covering:
- Policy health (ratio statistics, KL, entropy)
- Value function health (explained variance)
- Action distribution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class PPOHealthTelemetry:
    """Per-update PPO health metrics.

    These metrics are collected every PPO update (Ops Normal level).
    """

    # Loss components
    policy_loss: float
    value_loss: float
    entropy: float

    # KL and clipping
    approx_kl: float
    clip_fraction: float

    # Ratio statistics (CRITICAL for detecting explosions)
    ratio_mean: float
    ratio_std: float
    ratio_max: float
    ratio_min: float

    # Optional: early stopping info
    early_stopped: bool = False
    early_stop_epoch: int | None = None

    def is_ratio_healthy(
        self,
        max_ratio_threshold: float = 5.0,
        min_ratio_threshold: float = 0.1,
    ) -> bool:
        """Check if ratio statistics indicate healthy training.

        Args:
            max_ratio_threshold: Ratio above this is concerning
            min_ratio_threshold: Ratio below this is concerning

        Returns:
            True if ratios are within healthy bounds
        """
        return (
            self.ratio_max < max_ratio_threshold
            and self.ratio_min > min_ratio_threshold
        )

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


@dataclass(slots=True)
class ValueFunctionTelemetry:
    """Value function health metrics.

    Tracks how well the critic is predicting returns.
    """

    explained_variance: float
    value_mean: float
    value_std: float
    return_mean: float
    return_std: float
    advantage_mean: float
    advantage_std: float

    @classmethod
    def from_tensors(
        cls,
        returns: "torch.Tensor",
        values: "torch.Tensor",
        advantages: "torch.Tensor | None" = None,
    ) -> "ValueFunctionTelemetry":
        """Create from PyTorch tensors.

        Args:
            returns: Computed returns [N]
            values: Predicted values [N]
            advantages: Computed advantages [N] (optional)

        Returns:
            ValueFunctionTelemetry instance
        """
        import torch

        # Explained variance: 1 - Var(returns - values) / Var(returns)
        var_returns = returns.var()
        if var_returns > 1e-8:
            explained_var_tensor = 1.0 - (returns - values).var() / var_returns
        else:
            explained_var_tensor = torch.tensor(0.0, device=returns.device)

        # Batch all stats into single GPU sync (8 .item() calls → 1 .tolist())
        if advantages is not None:
            stats = torch.stack([
                explained_var_tensor,
                values.mean(), values.std(),
                returns.mean(), returns.std(),
                advantages.mean(), advantages.std(),
            ])
            ev, v_mean, v_std, r_mean, r_std, adv_mean, adv_std = stats.tolist()
        else:
            stats = torch.stack([
                explained_var_tensor,
                values.mean(), values.std(),
                returns.mean(), returns.std(),
            ])
            ev, v_mean, v_std, r_mean, r_std = stats.tolist()
            adv_mean, adv_std = 0.0, 1.0

        return cls(
            explained_variance=ev,
            value_mean=v_mean,
            value_std=v_std,
            return_mean=r_mean,
            return_std=r_std,
            advantage_mean=adv_mean,
            advantage_std=adv_std,
        )

    def is_healthy(self, min_explained_variance: float = 0.1) -> bool:
        """Check if value function is healthy.

        Args:
            min_explained_variance: Below this indicates broken critic

        Returns:
            True if value function appears healthy
        """
        return self.explained_variance >= min_explained_variance

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


__all__ = ["PPOHealthTelemetry", "ValueFunctionTelemetry"]
