"""Training loop orchestration for Tamiyo seed lifecycle.

This subpackage contains the high-level training coordination:
- vectorized.py: train_ppo_vectorized - main parallel training loop
- helpers.py: train_heuristic, train_one_epoch - single-env training
- parallel_env_state.py: ParallelEnvState - env context management
- config.py: TrainingConfig - hyperparameter containers
- policy_group.py: PolicyGroup - dual-policy A/B testing abstraction
"""

from .parallel_env_state import ParallelEnvState

from .config import TrainingConfig

from .policy_group import PolicyGroup

from .helpers import (
    train_heuristic,
    run_heuristic_episode,
)

from .vectorized import train_ppo_vectorized

from .dual_ab import train_dual_policy_ab

__all__ = [
    # Environment state
    "ParallelEnvState",
    # Config
    "TrainingConfig",
    # A/B testing
    "PolicyGroup",
    # Training functions
    "train_heuristic",
    "run_heuristic_episode",
    "train_ppo_vectorized",
    "train_dual_policy_ab",
]
