"""Epoch execution state extracted from VectorizedPPOTrainer.

This module provides dataclasses for managing per-batch and per-epoch state,
reducing the cognitive load of the main training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from esper.simic.rewards import ContributionRewardInputs, LossRewardInputs
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    RewardSummaryAccumulator,
)

if TYPE_CHECKING:
    pass


@dataclass
class EpochState:
    """Mutable state that changes during epoch execution.

    These are the per-batch accumulators and trackers, extracted from
    VectorizedPPOTrainer.run() to reduce variable count and improve clarity.
    """

    # Per-env accumulators for final metrics
    env_final_accs: list[float]
    env_total_rewards: list[float]
    reward_summary_accum: list[RewardSummaryAccumulator]

    # Action state (pre-allocated for reuse across epochs)
    action_specs: list[ActionSpec]
    action_outcomes: list[ActionOutcome]
    action_mask_flags: list[ActionMaskFlags]
    contribution_reward_inputs: list[ContributionRewardInputs]
    loss_reward_inputs: list[LossRewardInputs]

    # Rollback tracking
    env_rollback_occurred: list[bool]

    # For normalizer update (collected raw states from all epochs)
    raw_states_for_normalizer_update: list[Any] = field(default_factory=list)

    # Timing accumulators for telemetry
    throughput_step_time_ms_sum: float = 0.0
    throughput_dataloader_wait_ms_sum: float = 0.0

    @classmethod
    def create(
        cls,
        n_envs: int,
        max_epochs: int,
        env_reward_configs: list[Any],
        loss_reward_config: Any,
    ) -> "EpochState":
        """Create a new EpochState for a batch of environments.

        Args:
            n_envs: Number of parallel environments in this batch.
            max_epochs: Maximum epochs per episode.
            env_reward_configs: Per-environment reward configurations.
            loss_reward_config: Shared loss-based reward configuration.

        Returns:
            Initialized EpochState with pre-allocated data structures.
        """
        from esper.leyline import LifecycleOp

        return cls(
            env_final_accs=[0.0] * n_envs,
            env_total_rewards=[0.0] * n_envs,
            reward_summary_accum=[RewardSummaryAccumulator() for _ in range(n_envs)],
            action_specs=[ActionSpec() for _ in range(n_envs)],
            action_outcomes=[ActionOutcome() for _ in range(n_envs)],
            action_mask_flags=[ActionMaskFlags() for _ in range(n_envs)],
            contribution_reward_inputs=[
                ContributionRewardInputs(
                    action=LifecycleOp.WAIT,
                    seed_contribution=None,
                    val_acc=0.0,
                    seed_info=None,
                    epoch=0,
                    max_epochs=max_epochs,
                    total_params=0,
                    host_params=1,
                    acc_at_germination=None,
                    acc_delta=0.0,
                    config=env_reward_configs[env_idx],
                )
                for env_idx in range(n_envs)
            ],
            loss_reward_inputs=[
                LossRewardInputs(
                    action=LifecycleOp.WAIT,
                    loss_delta=0.0,
                    val_loss=0.0,
                    seed_info=None,
                    epoch=0,
                    max_epochs=max_epochs,
                    total_params=0,
                    host_params=1,
                    config=loss_reward_config,
                )
                for _ in range(n_envs)
            ],
            env_rollback_occurred=[False] * n_envs,
            raw_states_for_normalizer_update=[],
            throughput_step_time_ms_sum=0.0,
            throughput_dataloader_wait_ms_sum=0.0,
        )

    def reset_for_new_batch(self, n_envs: int) -> None:
        """Reset state for a new batch while preserving allocations.

        Args:
            n_envs: Number of environments in the new batch (may differ from previous).
        """
        # Resize lists if needed
        while len(self.env_final_accs) < n_envs:
            self.env_final_accs.append(0.0)
            self.env_total_rewards.append(0.0)
            self.reward_summary_accum.append(RewardSummaryAccumulator())
            self.action_specs.append(ActionSpec())
            self.action_outcomes.append(ActionOutcome())
            self.action_mask_flags.append(ActionMaskFlags())
            self.env_rollback_occurred.append(False)

        # Reset values
        for i in range(n_envs):
            self.env_final_accs[i] = 0.0
            self.env_total_rewards[i] = 0.0
            self.reward_summary_accum[i] = RewardSummaryAccumulator()
            self.env_rollback_occurred[i] = False

        self.raw_states_for_normalizer_update.clear()
        self.throughput_step_time_ms_sum = 0.0
        self.throughput_dataloader_wait_ms_sum = 0.0


__all__ = ["EpochState"]
