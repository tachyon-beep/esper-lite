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
    >>> group_a = PolicyGroup(
    ...     group_id="A",
    ...     device=torch.device("cuda:0"),
    ...     reward_mode="SHAPED",
    ...     agent=PPOAgent(...),
    ...     reward_config=ContributionRewardConfig(...),
    ... )
"""

from dataclasses import dataclass, field

import torch

from esper.simic.agent import PPOAgent
from esper.simic.rewards import ContributionRewardConfig
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

    Attributes:
        group_id: Identifier for this group (e.g., "A", "B")
        device: PyTorch device for this group's policy and environments
        reward_mode: Human-readable reward mode name (e.g., "SHAPED", "SIMPLIFIED")
        agent: PPO agent for this group (independent policy)
        envs: List of parallel environment states for this group
        reward_config: Reward computation configuration for this group
        episode_history: Per-episode metrics for this group (list of dicts)
        total_episodes: Total number of episodes completed by this group
        total_steps: Total environment steps taken by this group
        best_accuracy: Best final accuracy achieved by this group

    Usage:
        Groups are created at the start of dual-policy training and stepped
        in parallel. After training, their episode_history lists are compared
        to determine which reward mode performed better.
    """

    group_id: str
    device: torch.device
    reward_mode: str
    agent: PPOAgent
    envs: list[ParallelEnvState] = field(default_factory=list)
    reward_config: ContributionRewardConfig = field(default_factory=ContributionRewardConfig)
    episode_history: list[dict] = field(default_factory=list)

    # Per-group metrics
    total_episodes: int = 0
    total_steps: int = 0
    best_accuracy: float = 0.0
