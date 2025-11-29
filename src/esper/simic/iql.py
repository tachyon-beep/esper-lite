"""Simic IQL Module - Implicit Q-Learning for Offline RL

This module contains the IQL class for offline reinforcement learning.
For training functions, see simic.training.
For comparison modes, see simic.comparison.

References:
    - "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2021)
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from esper.simic.buffers import ReplayBuffer
from esper.simic.networks import QNetwork, VNetwork


# =============================================================================
# IQL Loss Functions
# =============================================================================

def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Asymmetric loss that penalizes overestimation.

    When tau > 0.5, positive errors (overestimation) are weighted more heavily.
    This makes the V-network conservative/pessimistic.

    Args:
        diff: Q - V differences
        tau: Expectile parameter (0.5 = symmetric, >0.5 = penalize overestimation)

    Returns:
        Weighted squared loss
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * (diff ** 2)


# =============================================================================
# IQL Agent
# =============================================================================

class IQL:
    """Implicit Q-Learning for offline RL.

    With optional CQL regularization to penalize OOD actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.7,
        beta: float = 3.0,
        lr: float = 3e-4,
        cql_alpha: float = 0.0,
        device: str = "cuda:0",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.action_dim = action_dim
        self.cql_alpha = cql_alpha

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.v_network = VNetwork(state_dim, hidden_dim).to(device)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=lr)

        self.target_update_rate = 0.005

    def update_v(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update V-network using expectile regression."""
        with torch.no_grad():
            q_values = self.q_target(states)
            q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        v_pred = self.v_network(states).squeeze(1)
        diff = q_a - v_pred
        v_loss = expectile_loss(diff, self.tau).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss.item()

    def update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[float, float]:
        """Update Q-network using V as target."""
        with torch.no_grad():
            v_next = self.v_network(next_states).squeeze(1)
            td_target = rewards + self.gamma * v_next * (1 - dones)

        q_values = self.q_network(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(q_a, td_target)

        cql_loss = 0.0
        if self.cql_alpha > 0:
            logsumexp_q = torch.logsumexp(q_values, dim=1)
            cql_loss = (logsumexp_q - q_a).mean()

        q_loss = td_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return td_loss.item(), cql_loss if isinstance(cql_loss, float) else cql_loss.item()

    def update_target(self) -> None:
        """Soft update of target Q-network."""
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.target_update_rate * param.data
                + (1 - self.target_update_rate) * target_param.data
            )

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """Single training step."""
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        v_loss = self.update_v(states, actions)
        td_loss, cql_loss = self.update_q(states, actions, rewards, next_states, dones)
        self.update_target()

        return {"v_loss": v_loss, "q_loss": td_loss, "cql_loss": cql_loss}

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """Get action from learned Q-values."""
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
            if deterministic:
                action = q_values.argmax(dim=1)
            else:
                probs = F.softmax(q_values * self.beta, dim=1)
                action = torch.multinomial(probs, 1).squeeze(1)
        self.q_network.train()
        return action.item() if state.shape[0] == 1 else action

    def get_policy_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from Q-values."""
        with torch.no_grad():
            q_values = self.q_network(states)
            probs = F.softmax(q_values * self.beta, dim=1)
        return probs


__all__ = [
    "IQL",
    "expectile_loss",
]
