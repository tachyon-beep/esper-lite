"""Buffer data structures for RL training.

This module contains trajectory and replay buffers used by PPO and IQL:
- RolloutBuffer: On-policy trajectory storage for PPO with GAE computation
- ReplayBuffer: Off-policy experience storage for IQL
"""

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


# =============================================================================
# IQL Buffers
# =============================================================================

@dataclass
class Transition:
    """A single (s, a, r, s', done) transition for offline RL."""
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Replay buffer for offline RL (IQL/CQL).

    Pre-converts all data to tensors for efficient sampling.
    """

    def __init__(self, transitions: list[Transition], device: str = "cuda:0"):
        self.device = device
        self.size = len(transitions)

        self.states = torch.tensor(
            [t.state for t in transitions], dtype=torch.float32, device=device
        )
        self.actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=device
        )
        self.rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=device
        )
        self.next_states = torch.tensor(
            [t.next_state for t in transitions], dtype=torch.float32, device=device
        )
        self.dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=device
        )

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    @property
    def state_dim(self) -> int:
        """Dimension of state vectors."""
        return self.states.shape[1]


__all__ = [
    "RolloutStep",
    "RolloutBuffer",
    "Transition",
    "ReplayBuffer",
]
