"""PPO algorithm core for Tamiyo seed lifecycle control.

This subpackage contains the learnable agent components:
- ppo.py: PPOAgent class and signals_to_features helper
- network.py: FactoredRecurrentActorCritic neural architecture
- rollout_buffer.py: TamiyoRolloutBuffer for trajectory storage
- advantages.py: GAE advantage computation
"""

from .advantages import compute_per_head_advantages

from .rollout_buffer import (
    TamiyoRolloutStep,
    TamiyoRolloutBuffer,
)

from .network import FactoredRecurrentActorCritic

from .ppo import (
    CHECKPOINT_VERSION,
    signals_to_features,
    PPOAgent,
)

from .types import (
    GradientStats,
    PPOUpdateMetrics,
    HeadLogProbs,
    HeadEntropies,
    ActionDict,
)

__all__ = [
    # Advantages
    "compute_per_head_advantages",
    # Buffer
    "TamiyoRolloutStep",
    "TamiyoRolloutBuffer",
    # Network
    "FactoredRecurrentActorCritic",
    # PPO Agent
    "CHECKPOINT_VERSION",
    "signals_to_features",
    "PPOAgent",
    # Types
    "GradientStats",
    "PPOUpdateMetrics",
    "HeadLogProbs",
    "HeadEntropies",
    "ActionDict",
]
