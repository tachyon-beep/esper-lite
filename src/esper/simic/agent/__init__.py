"""PPO algorithm core for Tamiyo seed lifecycle control.

This subpackage contains the learnable agent components:
- ppo_agent.py: PPOAgent class
- ppo_update.py: PPO update math (losses/ratios/KL)
- ppo_metrics.py: PPO metrics aggregation
- rollout_buffer.py: TamiyoRolloutBuffer for trajectory storage
- advantages.py: GAE advantage computation
"""

from .advantages import compute_per_head_advantages

from .rollout_buffer import (
    TamiyoRolloutStep,
    TamiyoRolloutBuffer,
)

from .ppo_agent import (
    CHECKPOINT_VERSION,
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
    # PPO Agent
    "CHECKPOINT_VERSION",
    "PPOAgent",
    # Types
    "GradientStats",
    "PPOUpdateMetrics",
    "HeadLogProbs",
    "HeadEntropies",
    "ActionDict",
]
