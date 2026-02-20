from __future__ import annotations

"""PolicyGroup: Independent policy with dedicated environments for A/B testing.

This module provides the core abstraction for dual-policy A/B testing, where
each PolicyGroup encapsulates one policy with its own environments, reward
configuration, and training history.

Architecture:
    - One PolicyGroup per GPU/device
    - Each group trains independently with its own reward mode
    - Groups run in lockstep for fair comparison
    - Final results show which reward mode produces the better policy

Example:
    >>> from esper.simic.rewards import RewardMode
    >>> group_a = PolicyGroup(
    ...     group_id="A",
    ...     device=torch.device("cuda:0"),
    ...     agent=PPOAgent(...),
    ...     reward_config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
    ... )
    >>> group_a.reward_mode  # Derived from reward_config
    RewardMode.SHAPED
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from esper.simic.agent import PPOAgent
from esper.simic.rewards import ContributionRewardConfig, RewardMode
from esper.simic.vectorized_types import EpisodeRecord

if TYPE_CHECKING:
    from esper.simic.training.parallel_env_state import ParallelEnvState

@dataclass
class PolicyGroup:
    """Independent policy with dedicated environments for A/B testing.

    Each PolicyGroup represents one experimental condition in a dual-policy
    A/B test. It contains a separate policy, dedicated environments, and
    its own reward configuration.

    Key Design:
        - One policy per group (no shared parameters)
        - One device per group (GPU isolation)
        - Independent training history (no cross-contamination)
        - Comparable metrics (lockstep episode execution)
        - Single source of truth: reward_mode is derived from reward_config

    Attributes:
        group_id: Identifier for this group (e.g., "A", "B")
        device: PyTorch device for this group's policy and environments
        agent: PPO agent for this group (independent policy)
        envs: List of parallel environment states for this group
        reward_config: Reward computation configuration (contains authoritative reward_mode)
        episode_history: Per-episode metrics for this group (typed via EpisodeRecord)
        total_episodes: Total number of episodes completed by this group
        total_steps: Total environment steps taken by this group
        best_accuracy: Best final accuracy achieved by this group

    Properties:
        reward_mode: Derived from reward_config.reward_mode (no duplication)

    Usage:
        Groups are created at the start of dual-policy training and stepped
        in parallel. After training, their episode_history lists are compared
        to determine which reward mode performed better.
    """

    group_id: str
    device: torch.device
    agent: PPOAgent
    # NOTE: envs field is for future parallel implementation - currently unused
    # because we delegate to train_ppo_vectorized which manages its own environments.
    # This field will be populated when we implement true parallel lockstep training.
    envs: list["ParallelEnvState"] = field(default_factory=list)
    reward_config: ContributionRewardConfig = field(default_factory=ContributionRewardConfig)
    episode_history: list[EpisodeRecord] = field(default_factory=list)

    @property
    def reward_mode(self) -> RewardMode:
        """Reward mode derived from reward_config (single source of truth)."""
        return self.reward_config.reward_mode

    # Per-group metrics
    total_episodes: int = 0
    total_steps: int = 0
    best_accuracy: float = 0.0


__all__ = [
    "EpisodeRecord",
    "PolicyGroup",
]
