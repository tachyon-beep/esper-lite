"""Buffer data structures for RL training (PPO)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch


# =============================================================================
# PPO Buffers
# =============================================================================

class RolloutStep(NamedTuple):
    """Single step in a PPO rollout."""
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    """Buffer for storing PPO rollout data.

    Stores trajectory steps and computes returns/advantages using GAE.
    """
    steps: list[RolloutStep] = field(default_factory=list)

    def add(self, state: torch.Tensor, action: int, log_prob: float,
            value: float, reward: float, done: bool) -> None:
        """Add a step to the buffer."""
        self.steps.append(RolloutStep(state, action, log_prob, value, reward, done))

    def clear(self) -> None:
        """Clear all steps from the buffer."""
        self.steps = []

    def __len__(self) -> int:
        return len(self.steps)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for state after last step (0 if done)
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff

        Returns:
            Tuple of (returns, advantages)
        """
        n_steps = len(self.steps)
        returns = torch.zeros(n_steps)
        advantages = torch.zeros(n_steps)

        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n_steps)):
            step = self.steps[t]

            if step.done:
                next_value = 0.0
                last_gae = 0.0

            delta = step.reward + gamma * next_value - step.value
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
            returns[t] = advantages[t] + step.value
            next_value = step.value

        return returns, advantages

    def get_batches(self, batch_size: int, device: str) -> list[tuple[dict, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Returns list of (batch_dict, batch_indices) tuples.
        """
        n_steps = len(self.steps)
        indices = torch.randperm(n_steps)

        batches = []
        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            batch_idx = indices[start:end]

            batch = {
                'states': torch.stack([self.steps[i].state for i in batch_idx]).to(device),
                'actions': torch.tensor([self.steps[i].action for i in batch_idx],
                                        dtype=torch.long, device=device),
                'old_log_probs': torch.tensor([self.steps[i].log_prob for i in batch_idx],
                                               dtype=torch.float32, device=device),
            }
            batches.append((batch, batch_idx))

        return batches


__all__ = [
    "RolloutStep",
    "RolloutBuffer",
]
