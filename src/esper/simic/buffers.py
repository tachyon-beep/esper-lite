"""Buffer data structures for RL training (PPO)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple

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
    action_mask: torch.Tensor
    truncated: bool = False  # True if episode ended due to time limit (should bootstrap)
    bootstrap_value: float = 0.0  # Value to bootstrap from if truncated


@dataclass
class RolloutBuffer:
    """Buffer for storing PPO rollout data.

    Stores trajectory steps and computes returns/advantages using GAE.
    Action masks are stored alongside observations to ensure consistent
    masking during PPO updates (same mask that was active when action was taken).
    """
    steps: list[RolloutStep] = field(default_factory=list)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Add a step to the buffer.

        Args:
            state: Observation tensor
            action: Action taken
            log_prob: Log probability of action under policy
            value: Value estimate at this state
            reward: Reward received
            done: Whether episode ended (naturally or truncated)
            action_mask: Binary mask of valid actions at this state
            truncated: Whether episode ended due to time limit (not natural termination)
            bootstrap_value: Value to bootstrap from if truncated
        """
        self.steps.append(RolloutStep(state, action, log_prob, value, reward, done, action_mask, truncated, bootstrap_value))

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
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for state after last step (0 if done)
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff
            device: Device to create tensors on (avoids CPU->GPU transfer)

        Returns:
            Tuple of (returns, advantages) on the specified device
        """
        n_steps = len(self.steps)
        returns = torch.zeros(n_steps, device=device)
        advantages = torch.zeros(n_steps, device=device)

        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n_steps)):
            step = self.steps[t]

            if step.done:
                # For truncated episodes, bootstrap from the estimated value
                # For naturally terminated episodes, use 0.0
                if step.truncated:
                    next_value = step.bootstrap_value
                else:
                    next_value = 0.0
                # Reset GAE at ALL episode boundaries
                # bootstrap_value handles value continuation; GAE chain should not span episodes
                last_gae = 0.0

            delta = step.reward + gamma * next_value - step.value
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
            returns[t] = advantages[t] + step.value
            next_value = step.value

        return returns, advantages

    def get_batches(self, batch_size: int, device: str) -> list[tuple[dict, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Returns list of (batch_dict, batch_indices) tuples.
        Includes action_masks for correct masked policy evaluation.

        Performance: Pre-stacks all data once, then slices for batches to avoid
        repeated tensor creation and memory fragmentation.
        """
        n_steps = len(self.steps)
        indices = torch.randperm(n_steps)

        # Pre-stack all data once to avoid repeated tensor creation per batch
        all_states = torch.stack([s.state for s in self.steps]).to(device)
        all_actions = torch.tensor([s.action for s in self.steps], dtype=torch.long, device=device)
        all_log_probs = torch.tensor([s.log_prob for s in self.steps], dtype=torch.float32, device=device)
        all_values = torch.tensor([s.value for s in self.steps], dtype=torch.float32, device=device)
        all_masks = torch.stack([s.action_mask for s in self.steps]).to(device)

        batches = []
        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            batch_idx = indices[start:end]

            # Slice pre-stacked tensors (views, not copies)
            batch = {
                'states': all_states[batch_idx],
                'actions': all_actions[batch_idx],
                'old_log_probs': all_log_probs[batch_idx],
                'values': all_values[batch_idx],
                'action_masks': all_masks[batch_idx],
            }
            batches.append((batch, batch_idx))

        return batches


__all__ = ["RolloutStep", "RolloutBuffer"]
