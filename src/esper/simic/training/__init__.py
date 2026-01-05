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
"""

from .parallel_env_state import ParallelEnvState

from .config import TrainingConfig

from .policy_group import EpisodeRecord, PolicyGroup

from .helpers import (
    train_heuristic,
    run_heuristic_episode,
)

# NOTE: vectorized and dual_ab are NOT imported here to avoid import-time side effects
# Import them explicitly when needed

__all__ = [
    # Environment state
    "ParallelEnvState",
    # Config
    "TrainingConfig",
    # A/B testing
    "EpisodeRecord",
    "PolicyGroup",
    # Training functions (light)
    "train_heuristic",
    "run_heuristic_episode",
    # Heavy training functions (import explicitly)
    # "train_ppo_vectorized",  # from .vectorized
    # "train_dual_policy_ab",  # from .dual_ab
]
