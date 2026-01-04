"""Reward configuration contracts for Esper training.

These configuration dataclasses define the hyperparameters for reward computation.
They live in leyline (not simic) because they're shared across domain boundaries:
- runtime/tasks.py uses LossRewardConfig in TaskSpec
- simic/rewards uses them for actual reward computation

Moving these here breaks the import cycle:
    runtime -> simic.rewards -> simic.training -> tolaria -> runtime
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LossRewardConfig:
    """Configuration for loss-primary reward computation.

    All weights are tunable hyperparameters optimized for
    cross-task comparability using normalized loss delta.
    """

    # Loss delta scaling
    loss_delta_weight: float = 5.0
    max_loss_delta: float = 5.0  # After normalization
    regression_penalty_scale: float = 0.5  # Asymmetric clipping
    typical_loss_delta_std: float = 0.1  # Task-specific normalization

    # Compute rent (logarithmic scaling)
    compute_rent_weight: float = 0.05
    max_rent_penalty: float = 5.0
    grace_epochs: int = 3  # Rent-free grace period for new seeds

    # Stage bonuses (PBRS-compatible)
    stage_potential_weight: float = 0.1

    # Terminal bonus
    baseline_loss: float = 2.3  # Task-specific (random init loss)
    target_loss: float = 0.3  # Task-specific (achievable loss)
    terminal_loss_weight: float = 1.0

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss

    @staticmethod
    def default() -> "LossRewardConfig":
        return LossRewardConfig()

    @staticmethod
    def for_cifar10() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=2.3,  # ln(10)
            target_loss=0.3,
            typical_loss_delta_std=0.05,
        )

    @staticmethod
    def for_tinystories() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,
            typical_loss_delta_std=0.15,
            compute_rent_weight=0.01,
        )
