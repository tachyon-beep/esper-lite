"""Simic IQL - Phase 3: Implicit Q-Learning for Offline RL

This module implements IQL (Implicit Q-Learning) for training Tamiyo
to optimize outcomes rather than imitate Kasmina.

Key insight: Standard Q-learning overestimates Q-values for rare actions
(like GERMINATE). IQL uses expectile regression to stay conservative.

Architecture:
- Q-network: Q(s, a) -> value of taking action a in state s
- V-network: V(s) -> conservative value estimate (via expectile loss)
- Policy: π(s) = argmax_a Q(s, a)

IQL avoids the max operator in targets:
- Standard: target = r + γ * max_a Q(s', a)  [overestimates OOD actions]
- IQL:      target = r + γ * V(s')           [V learned conservatively]

Usage:
    PYTHONPATH=src .venv/bin/python -m esper.simic_iql \
        --pack data/packs/simic_v2_research_2025-11-26.json \
        --epochs 100

References:
    - "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2021)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.simic import SimicAction
from esper.simic_train import obs_to_base_features, telemetry_to_features, safe


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Transition:
    """A single (s, a, r, s', done) transition."""
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Simple replay buffer for offline RL."""

    def __init__(self, transitions: list[Transition], device: str = "cuda:0"):
        self.device = device
        self.size = len(transitions)

        # Pre-convert to tensors for efficiency
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
        """Sample a batch of transitions."""
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
        return self.states.shape[1]


# =============================================================================
# Networks
# =============================================================================

class QNetwork(nn.Module):
    """Q-network: Q(s, a) for all actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: shape (batch, action_dim)."""
        return self.net(state)


class VNetwork(nn.Module):
    """V-network: V(s) state value."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns state value: shape (batch, 1)."""
        return self.net(state)


# =============================================================================
# IQL Algorithm
# =============================================================================

def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Asymmetric loss that penalizes overestimation.

    When tau > 0.5, positive errors (overestimation) are weighted more heavily.
    This makes the V-network conservative/pessimistic.

    Args:
        diff: Q - V differences
        tau: Expectile parameter (0.5 = symmetric, >0.5 = penalize overestimation)

    Returns:
        Weighted squared loss.
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * (diff ** 2)


class IQL:
    """Implicit Q-Learning for offline RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.7,  # Expectile parameter
        beta: float = 3.0,  # Temperature for policy extraction
        lr: float = 3e-4,
        device: str = "cuda:0",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.action_dim = action_dim

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.v_network = VNetwork(state_dim, hidden_dim).to(device)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=lr)

        # For soft target updates
        self.target_update_rate = 0.005

    def update_v(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update V-network using expectile regression.

        V should learn to underestimate Q (be conservative).
        """
        with torch.no_grad():
            # Get Q-values for the actions taken
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
    ) -> float:
        """Update Q-network using V as target (not max Q).

        This is the key IQL insight: using V(s') instead of max_a Q(s', a)
        avoids overestimation of OOD actions.
        """
        with torch.no_grad():
            # Use V-network for next state value (conservative!)
            v_next = self.v_network(next_states).squeeze(1)
            td_target = rewards + self.gamma * v_next * (1 - dones)

        # Get Q-values for the actions taken
        q_values = self.q_network(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        q_loss = F.mse_loss(q_a, td_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q_loss.item()

    def update_target(self):
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
        q_loss = self.update_q(states, actions, rewards, next_states, dones)
        self.update_target()

        return {"v_loss": v_loss, "q_loss": q_loss}

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """Get action from learned Q-values."""
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
            if deterministic:
                action = q_values.argmax(dim=1)
            else:
                # Softmax with temperature
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


# =============================================================================
# Data Loading
# =============================================================================

def extract_transitions(
    pack: dict,
    use_telemetry: bool = True,
    reward_scale: float = 0.1,  # Scale down rewards for stability
) -> list[Transition]:
    """Extract (s, a, r, s', done) transitions from episodes.

    Args:
        pack: Data pack dictionary.
        use_telemetry: Whether to include V2 telemetry features.
        reward_scale: Scale factor for rewards.

    Returns:
        List of Transition objects.
    """
    transitions = []

    for ep in pack['episodes']:
        decisions = ep['decisions']
        telem_hist = ep.get('telemetry_history', [])

        for i, decision in enumerate(decisions):
            # Current state
            state = obs_to_base_features(decision['observation'])
            if use_telemetry and telem_hist:
                epoch = decision['observation']['epoch']
                if epoch <= len(telem_hist):
                    state.extend(telemetry_to_features(telem_hist[epoch - 1]))
                else:
                    state.extend([0.0] * 27)
            elif use_telemetry:
                state.extend([0.0] * 27)

            # Action
            action = SimicAction[decision['action']['action']].value

            # Reward (scaled)
            reward = decision['outcome'].get('reward', 0) * reward_scale

            # Next state (or terminal)
            is_last = (i == len(decisions) - 1)
            if is_last:
                next_state = state  # Doesn't matter, will be masked by done=True
                done = True
            else:
                next_decision = decisions[i + 1]
                next_state = obs_to_base_features(next_decision['observation'])
                if use_telemetry and telem_hist:
                    next_epoch = next_decision['observation']['epoch']
                    if next_epoch <= len(telem_hist):
                        next_state.extend(telemetry_to_features(telem_hist[next_epoch - 1]))
                    else:
                        next_state.extend([0.0] * 27)
                elif use_telemetry:
                    next_state.extend([0.0] * 27)
                done = False

            transitions.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ))

    return transitions


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy(
    iql: IQL,
    buffer: ReplayBuffer,
    n_samples: int = 1000,
) -> dict:
    """Evaluate IQL policy against behavior policy (Kasmina).

    Returns metrics comparing IQL's choices to Kasmina's choices.
    """
    idx = torch.randint(0, buffer.size, (n_samples,), device=buffer.device)
    states = buffer.states[idx]
    behavior_actions = buffer.actions[idx]
    rewards = buffer.rewards[idx]

    # Get IQL's preferred actions
    with torch.no_grad():
        q_values = iql.q_network(states)
        iql_actions = q_values.argmax(dim=1)

    # Agreement rate
    agreement = (iql_actions == behavior_actions).float().mean().item()

    # Action distribution comparison
    behavior_dist = torch.bincount(behavior_actions, minlength=4).float() / n_samples
    iql_dist = torch.bincount(iql_actions, minlength=4).float() / n_samples

    # Q-values for behavior vs IQL actions
    q_behavior = q_values.gather(1, behavior_actions.unsqueeze(1)).squeeze(1).mean().item()
    q_iql = q_values.gather(1, iql_actions.unsqueeze(1)).squeeze(1).mean().item()

    return {
        "agreement": agreement,
        "q_behavior": q_behavior,
        "q_iql": q_iql,
        "q_improvement": q_iql - q_behavior,
        "behavior_dist": behavior_dist.tolist(),
        "iql_dist": iql_dist.tolist(),
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train_iql(
    pack_path: str,
    epochs: int = 100,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.7,
    beta: float = 3.0,
    lr: float = 3e-4,
    use_telemetry: bool = True,
    device: str = "cuda:0",
    save_path: Optional[str] = None,
) -> IQL:
    """Train IQL on offline data.

    Args:
        pack_path: Path to data pack JSON.
        epochs: Number of training epochs.
        steps_per_epoch: Training steps per epoch.
        batch_size: Batch size for training.
        gamma: Discount factor.
        tau: Expectile parameter for V-network.
        beta: Temperature for policy extraction.
        lr: Learning rate.
        use_telemetry: Whether to use V2 telemetry features.
        device: Device to train on.
        save_path: Path to save trained model.

    Returns:
        Trained IQL agent.
    """
    print("=" * 60)
    print("Tamiyo Phase 3: Implicit Q-Learning")
    print("=" * 60)

    # Load data
    print(f"Loading {pack_path}...")
    with open(pack_path) as f:
        pack = json.load(f)
    print(f"Episodes: {pack['metadata']['num_episodes']}")

    # Extract transitions
    print("Extracting transitions...")
    transitions = extract_transitions(pack, use_telemetry=use_telemetry)
    print(f"Transitions: {len(transitions)}")

    # Create buffer
    buffer = ReplayBuffer(transitions, device=device)
    print(f"State dim: {buffer.state_dim}")

    # Reward statistics
    rewards = buffer.rewards
    print(f"Reward range: [{rewards.min():.2f}, {rewards.max():.2f}], mean: {rewards.mean():.2f}")
    print()

    # Create IQL agent
    iql = IQL(
        state_dim=buffer.state_dim,
        action_dim=len(SimicAction),
        gamma=gamma,
        tau=tau,
        beta=beta,
        lr=lr,
        device=device,
    )

    print(f"Training for {epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"γ={gamma}, τ={tau}, β={beta}, lr={lr}")
    print()

    # Training loop
    best_q_improvement = -float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        epoch_v_loss = 0.0
        epoch_q_loss = 0.0

        for _ in range(steps_per_epoch):
            losses = iql.train_step(buffer, batch_size)
            epoch_v_loss += losses["v_loss"]
            epoch_q_loss += losses["q_loss"]

        epoch_v_loss /= steps_per_epoch
        epoch_q_loss /= steps_per_epoch

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics = evaluate_policy(iql, buffer)
            print(f"Epoch {epoch:3d}: v_loss={epoch_v_loss:.4f}, q_loss={epoch_q_loss:.4f} | "
                  f"agreement={metrics['agreement']*100:.1f}%, "
                  f"Q_improve={metrics['q_improvement']:.3f}")

            if metrics['q_improvement'] > best_q_improvement:
                best_q_improvement = metrics['q_improvement']
                best_state_dict = {
                    'q_network': copy.deepcopy(iql.q_network.state_dict()),
                    'v_network': copy.deepcopy(iql.v_network.state_dict()),
                }

    # Load best model
    if best_state_dict:
        iql.q_network.load_state_dict(best_state_dict['q_network'])
        iql.v_network.load_state_dict(best_state_dict['v_network'])

    # Final evaluation
    print()
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    metrics = evaluate_policy(iql, buffer, n_samples=min(5000, buffer.size))

    print(f"Agreement with Kasmina: {metrics['agreement']*100:.1f}%")
    print(f"Q-value (Kasmina's actions): {metrics['q_behavior']:.3f}")
    print(f"Q-value (IQL's actions):     {metrics['q_iql']:.3f}")
    print(f"Q-improvement:               {metrics['q_improvement']:.3f}")
    print()
    print("Action distribution comparison:")
    print(f"  {'Action':<12} {'Kasmina':>10} {'IQL':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for i, action in enumerate(SimicAction):
        print(f"  {action.name:<12} {metrics['behavior_dist'][i]*100:>9.1f}% {metrics['iql_dist'][i]*100:>9.1f}%")

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
            'state_dim': buffer.state_dim,
            'action_dim': len(SimicAction),
            'gamma': gamma,
            'tau': tau,
            'beta': beta,
            'metrics': metrics,
        }, save_path)
        print(f"\nModel saved to {save_path}")

    return iql


def main():
    parser = argparse.ArgumentParser(description="Train Tamiyo Phase 3 (IQL)")
    parser.add_argument("--pack", required=True, help="Path to data pack JSON")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.7, help="Expectile parameter")
    parser.add_argument("--beta", type=float, default=3.0, help="Policy temperature")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-telemetry", action="store_true", help="Use only base features")
    parser.add_argument("--save", help="Path to save trained model")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    train_iql(
        pack_path=args.pack,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        lr=args.lr,
        use_telemetry=not args.no_telemetry,
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
