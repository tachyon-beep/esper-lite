"""Training loop orchestration for Tamiyo seed lifecycle.

This subpackage contains the high-level training coordination:
- vectorized.py: train_ppo_vectorized - main parallel training loop
- helpers.py: train_heuristic, train_one_epoch - single-env training
- parallel_env_state.py: ParallelEnvState - env context management
- config.py: TrainingConfig - hyperparameter containers
- policy_group.py: PolicyGroup - dual-policy A/B testing abstraction

IMPORTANT: Heavy training loops (vectorized, dual_ab) are NOT imported automatically
to avoid loading torch and CUDA machinery at import time. Import explicitly:
    from esper.simic.training.vectorized import train_ppo_vectorized
    from esper.simic.training.dual_ab import train_dual_policy_ab
For PolicyGroup, import directly to avoid PPO agent import cycles:
    from esper.simic.training.policy_group import PolicyGroup
Lightweight helpers should also be imported directly to avoid circular imports:
    from esper.simic.training.helpers import train_heuristic, run_heuristic_episode
"""

from .parallel_env_state import ParallelEnvState

from .config import TrainingConfig

from esper.simic.vectorized_types import EpisodeRecord

# NOTE: vectorized and dual_ab are NOT imported here to avoid import-time side effects
# Import them explicitly when needed

__all__ = [
    # Environment state
    "ParallelEnvState",
    # Config
    "TrainingConfig",
    # A/B testing
    "EpisodeRecord",
    # Heavy training functions (import explicitly)
    # "train_ppo_vectorized",  # from .vectorized
    # "train_dual_policy_ab",  # from .dual_ab
]
