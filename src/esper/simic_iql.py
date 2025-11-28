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

from esper.simic import SimicAction, TrainingSnapshot
from esper.simic_train import obs_to_base_features, telemetry_to_features, safe
from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig, SignalTracker


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
    """Implicit Q-Learning for offline RL.

    With optional CQL regularization to penalize OOD actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.7,  # Expectile parameter
        beta: float = 3.0,  # Temperature for policy extraction
        lr: float = 3e-4,
        cql_alpha: float = 0.0,  # CQL regularization (0 = pure IQL)
        device: str = "cuda:0",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.action_dim = action_dim
        self.cql_alpha = cql_alpha

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
    ) -> tuple[float, float]:
        """Update Q-network using V as target (not max Q).

        This is the key IQL insight: using V(s') instead of max_a Q(s', a)
        avoids overestimation of OOD actions.

        With CQL: adds penalty for high Q-values on non-dataset actions.
        CQL loss = E[logsumexp(Q(s,·))] - E[Q(s,a)]
        This pushes down Q-values for actions not in the dataset.
        """
        with torch.no_grad():
            # Use V-network for next state value (conservative!)
            v_next = self.v_network(next_states).squeeze(1)
            td_target = rewards + self.gamma * v_next * (1 - dones)

        # Get Q-values for the actions taken
        q_values = self.q_network(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        td_loss = F.mse_loss(q_a, td_target)

        # CQL regularization: penalize high Q-values for OOD actions
        cql_loss = 0.0
        if self.cql_alpha > 0:
            # logsumexp over all actions - Q-value of taken action
            # This penalizes Q-values that are high for actions not in data
            logsumexp_q = torch.logsumexp(q_values, dim=1)
            cql_loss = (logsumexp_q - q_a).mean()

        q_loss = td_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return td_loss.item(), cql_loss if isinstance(cql_loss, float) else cql_loss.item()

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
# Reward Shaping
# =============================================================================

def compute_seed_potential(obs: dict) -> float:
    """Compute potential value Φ(s) based on seed state.

    The potential captures the expected future value of having an active seed
    in various stages. This helps bridge the temporal gap where GERMINATE
    has negative immediate reward but high future value.

    Potential-based reward shaping: r' = r + γ*Φ(s') - Φ(s)
    This preserves optimal policy (PBRS guarantee) while improving learning.

    Seed stages and their value:
    - No seed (stage 0): 0 (no potential)
    - GERMINATED (stage 1): Low potential (seed just started)
    - TRAINING (stage 2): Higher potential (actively learning)
    - BLENDING (stage 3): Highest potential (about to integrate)
    - FOSSILIZED (stage 4): Medium potential (integrated, benefits realized)
    """
    has_active = obs.get('has_active_seed', 0)
    seed_stage = obs.get('seed_stage', 0)
    epochs_in_stage = obs.get('seed_epochs_in_stage', 0)

    if not has_active or seed_stage == 0:
        return 0.0

    # Stage-based potential values (tuned based on reward analysis)
    # GERMINATE has -7.43 immediate but +21.41 future -> ~28 total value
    # We want potential to give "credit" for this
    stage_potentials = {
        1: 5.0,   # GERMINATED - just started, some potential
        2: 15.0,  # TRAINING - actively learning, high potential
        3: 25.0,  # BLENDING - about to integrate, highest potential
        4: 10.0,  # FOSSILIZED - integrated, value mostly realized
    }

    base_potential = stage_potentials.get(seed_stage, 0.0)

    # Small bonus for progress within a stage (capped)
    progress_bonus = min(epochs_in_stage * 0.5, 3.0)

    return base_potential + progress_bonus


# =============================================================================
# Data Loading
# =============================================================================

def extract_transitions(
    pack: dict,
    use_telemetry: bool = True,
    reward_scale: float = 0.1,  # Scale down rewards for stability
    use_reward_shaping: bool = False,  # Enable potential-based shaping
    gamma: float = 0.99,  # Discount for reward shaping
) -> list[Transition]:
    """Extract (s, a, r, s', done) transitions from episodes.

    Args:
        pack: Data pack dictionary.
        use_telemetry: Whether to include V2 telemetry features.
        reward_scale: Scale factor for rewards.
        use_reward_shaping: Whether to add potential-based reward shaping.
        gamma: Discount factor for reward shaping (r' = r + γ*Φ(s') - Φ(s)).

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

            # Base reward (scaled)
            raw_reward = decision['outcome'].get('reward', 0) * reward_scale

            # Next state (or terminal)
            is_last = (i == len(decisions) - 1)
            if is_last:
                next_state = state  # Doesn't matter, will be masked by done=True
                done = True
                next_obs = None
            else:
                next_decision = decisions[i + 1]
                next_obs = next_decision['observation']
                next_state = obs_to_base_features(next_obs)
                if use_telemetry and telem_hist:
                    next_epoch = next_obs['epoch']
                    if next_epoch <= len(telem_hist):
                        next_state.extend(telemetry_to_features(telem_hist[next_epoch - 1]))
                    else:
                        next_state.extend([0.0] * 27)
                elif use_telemetry:
                    next_state.extend([0.0] * 27)
                done = False

            # Apply potential-based reward shaping: r' = r + γ*Φ(s') - Φ(s)
            if use_reward_shaping:
                current_obs = decision['observation']
                phi_s = compute_seed_potential(current_obs)
                if done:
                    phi_s_prime = 0.0  # Terminal state has zero potential
                else:
                    phi_s_prime = compute_seed_potential(next_obs)
                shaping_bonus = gamma * phi_s_prime - phi_s
                reward = raw_reward + shaping_bonus * reward_scale
            else:
                reward = raw_reward

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
    cql_alpha: float = 0.0,
    use_telemetry: bool = True,
    use_reward_shaping: bool = False,
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
        use_reward_shaping: Whether to use potential-based reward shaping.
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
    transitions = extract_transitions(
        pack,
        use_telemetry=use_telemetry,
        use_reward_shaping=use_reward_shaping,
        gamma=gamma,
    )
    print(f"Transitions: {len(transitions)}")
    if use_reward_shaping:
        print("Using potential-based reward shaping (PBRS)")

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
        cql_alpha=cql_alpha,
        device=device,
    )

    algo_name = "IQL+CQL" if cql_alpha > 0 else "IQL"
    print(f"Training for {epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"Algorithm: {algo_name}")
    print(f"γ={gamma}, τ={tau}, β={beta}, lr={lr}", end="")
    if cql_alpha > 0:
        print(f", cql_α={cql_alpha}")
    else:
        print()
    print()

    # Training loop
    best_q_improvement = -float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        epoch_v_loss = 0.0
        epoch_q_loss = 0.0
        epoch_cql_loss = 0.0

        for _ in range(steps_per_epoch):
            losses = iql.train_step(buffer, batch_size)
            epoch_v_loss += losses["v_loss"]
            epoch_q_loss += losses["q_loss"]
            epoch_cql_loss += losses["cql_loss"]

        epoch_v_loss /= steps_per_epoch
        epoch_q_loss /= steps_per_epoch
        epoch_cql_loss /= steps_per_epoch

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics = evaluate_policy(iql, buffer)
            loss_str = f"v_loss={epoch_v_loss:.4f}, q_loss={epoch_q_loss:.4f}"
            if cql_alpha > 0:
                loss_str += f", cql_loss={epoch_cql_loss:.4f}"
            print(f"Epoch {epoch:3d}: {loss_str} | "
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


# =============================================================================
# Live Comparison
# =============================================================================

def load_iql_model(model_path: str, device: str = "cpu") -> IQL:
    """Load a trained IQL model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    iql = IQL(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        gamma=checkpoint.get('gamma', 0.99),
        tau=checkpoint.get('tau', 0.7),
        beta=checkpoint.get('beta', 3.0),
        device=device,
    )
    iql.q_network.load_state_dict(checkpoint['q_network'])
    iql.v_network.load_state_dict(checkpoint['v_network'])

    return iql


def snapshot_to_features(snapshot: TrainingSnapshot, use_telemetry: bool = False) -> list[float]:
    """Convert TrainingSnapshot to feature vector for IQL."""
    # Convert snapshot to observation dict format expected by obs_to_base_features
    obs = {
        'epoch': snapshot.epoch,
        'global_step': snapshot.global_step,
        'train_loss': snapshot.train_loss,
        'val_loss': snapshot.val_loss,
        'loss_delta': snapshot.loss_delta,
        'train_accuracy': snapshot.train_accuracy,
        'val_accuracy': snapshot.val_accuracy,
        'accuracy_delta': snapshot.accuracy_delta,
        'plateau_epochs': snapshot.plateau_epochs,
        'best_val_accuracy': snapshot.best_val_accuracy,
        'best_val_loss': snapshot.best_val_loss,
        'loss_history_5': list(snapshot.loss_history_5),
        'accuracy_history_5': list(snapshot.accuracy_history_5),
        'has_active_seed': snapshot.has_active_seed,
        'seed_stage': snapshot.seed_stage,
        'seed_epochs_in_stage': snapshot.seed_epochs_in_stage,
        'seed_alpha': snapshot.seed_alpha,
        'seed_improvement': snapshot.seed_improvement,
        'available_slots': snapshot.available_slots,
    }

    features = obs_to_base_features(obs)

    if use_telemetry:
        # Pad with zeros for telemetry features (not available in live comparison)
        features.extend([0.0] * 27)

    return features


def live_comparison(
    model_path: str,
    n_episodes: int = 5,
    max_epochs: int = 25,
    device: str = "cpu",
    use_telemetry: bool = True,
) -> dict:
    """Run live comparison between heuristic Tamiyo and IQL policy.

    Both policies make decisions, but only heuristic actually controls training.
    We track what IQL *would* have done and compare accuracy outcomes.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("Live Comparison: Heuristic Tamiyo vs IQL Policy")
    print("=" * 60)

    # Load IQL model
    print(f"Loading IQL model from {model_path}...")
    iql = load_iql_model(model_path, device=device)
    print(f"  State dim: {iql.q_network.net[0].in_features}")

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {
        'heuristic_accuracies': [],
        'iql_would_have': [],  # What IQL would have chosen
        'agreements': [],
        'action_counts': {'heuristic': {a.name: 0 for a in SimicAction},
                         'iql': {a.name: 0 for a in SimicAction}},
    }

    for ep_idx in range(n_episodes):
        print(f"\n--- Episode {ep_idx + 1}/{n_episodes} ---")
        torch.manual_seed(42 + ep_idx)

        # Setup
        from esper.simic_overnight import create_model
        model = create_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.get_host_parameters(), lr=0.01, momentum=0.9)

        tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
        tracker = SignalTracker()

        ep_agreements = 0
        ep_total = 0

        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, pred = outputs.max(1)
                train_total += labels.size(0)
                train_correct += pred.eq(labels).sum().item()
            train_loss /= len(trainloader)
            train_acc = 100.0 * train_correct / train_total

            # Validate
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()
            val_loss /= len(testloader)
            val_acc = 100.0 * val_correct / val_total

            # Update tracker
            active_seeds = []
            if model.has_active_seed:
                active_seeds = [model.seed_state]
            available_slots = 0 if model.has_active_seed else 1
            signals = tracker.update(
                epoch=epoch,
                global_step=epoch * len(trainloader),
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                active_seeds=active_seeds,
                available_slots=available_slots,
            )

            # Get heuristic decision
            h_action = tamiyo.decide(signals)

            # Get IQL decision
            # Pad history to 5 elements
            loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
            while len(loss_hist) < 5:
                loss_hist.insert(0, 0.0)
            acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
            while len(acc_hist) < 5:
                acc_hist.insert(0, 0.0)

            snapshot = TrainingSnapshot(
                epoch=signals.epoch,
                global_step=signals.global_step,
                train_loss=signals.train_loss,
                val_loss=signals.val_loss,
                loss_delta=signals.loss_delta,
                train_accuracy=signals.train_accuracy,
                val_accuracy=signals.val_accuracy,
                accuracy_delta=signals.accuracy_delta,
                plateau_epochs=signals.plateau_epochs,
                best_val_accuracy=signals.best_val_accuracy,
                best_val_loss=min(signals.loss_history) if signals.loss_history else float('inf'),
                loss_history_5=tuple(loss_hist),
                accuracy_history_5=tuple(acc_hist),
                has_active_seed=model.has_active_seed,
                available_slots=available_slots,
            )

            features = snapshot_to_features(snapshot, use_telemetry=use_telemetry)
            state_tensor = torch.tensor([features], dtype=torch.float32, device=device)
            iql_action_idx = iql.get_action(state_tensor, deterministic=True)
            iql_action = SimicAction(iql_action_idx).name

            # Track - convert TamiyoAction to SimicAction name
            h_action_name = h_action.action.name  # TamiyoAction enum
            results['action_counts']['heuristic'][h_action_name] += 1
            results['action_counts']['iql'][iql_action] += 1

            if h_action_name == iql_action:
                ep_agreements += 1
            ep_total += 1

            # Note: We don't execute actions in this comparison - both policies
            # just observe the same training trajectory to compare decisions

        final_acc = val_acc
        agreement_rate = ep_agreements / ep_total if ep_total > 0 else 0

        results['heuristic_accuracies'].append(final_acc)
        results['agreements'].append(agreement_rate)

        print(f"  Final accuracy: {final_acc:.2f}%")
        print(f"  Agreement rate: {agreement_rate*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    avg_acc = sum(results['heuristic_accuracies']) / len(results['heuristic_accuracies'])
    avg_agreement = sum(results['agreements']) / len(results['agreements'])

    print(f"Average accuracy (heuristic): {avg_acc:.2f}%")
    print(f"Average agreement rate: {avg_agreement*100:.1f}%")
    print()
    print("Action distribution:")
    print(f"  {'Action':<12} {'Heuristic':>10} {'IQL':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    h_total = sum(results['action_counts']['heuristic'].values())
    i_total = sum(results['action_counts']['iql'].values())

    for action in SimicAction:
        h_pct = results['action_counts']['heuristic'][action.name] / h_total * 100 if h_total > 0 else 0
        i_pct = results['action_counts']['iql'][action.name] / i_total * 100 if i_total > 0 else 0
        print(f"  {action.name:<12} {h_pct:>9.1f}% {i_pct:>9.1f}%")

    return results


def head_to_head_comparison(
    model_path: str,
    n_episodes: int = 5,
    max_epochs: int = 25,
    device: str = "cpu",
    use_telemetry: bool = True,
) -> dict:
    """Run head-to-head comparison where each policy ACTUALLY controls training.

    Unlike live_comparison (where both observe the same trajectory), this function
    runs TWO SEPARATE training runs per episode:
    1. Heuristic Tamiyo controls one run
    2. IQL policy controls another run

    We compare final accuracies to determine which policy is better.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim

    from esper.kasmina import SeedStage

    print("=" * 60)
    print("Head-to-Head: Heuristic Tamiyo vs IQL Policy")
    print("=" * 60)
    print("Each policy EXECUTES its decisions in separate training runs")
    print()

    # Load IQL model
    print(f"Loading IQL model from {model_path}...")
    iql = load_iql_model(model_path, device=device)
    state_dim = iql.q_network.net[0].in_features
    print(f"  State dim: {state_dim}")

    # Load CIFAR-10 once
    print("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {
        'heuristic_accuracies': [],
        'iql_accuracies': [],
        'heuristic_wins': 0,
        'iql_wins': 0,
        'ties': 0,
        'action_counts': {
            'heuristic': {a.name: 0 for a in SimicAction},
            'iql': {a.name: 0 for a in SimicAction},
        },
    }

    def run_training_episode(
        policy_name: str,
        action_fn,  # function(signals, model, tracker) -> SimicAction
        seed: int,
    ) -> tuple[float, dict]:
        """Run a single training episode controlled by the given policy."""
        torch.manual_seed(seed)

        from esper.simic_overnight import create_model
        model = create_model(device)
        criterion = nn.CrossEntropyLoss()
        host_optimizer = optim.SGD(model.get_host_parameters(), lr=0.01, momentum=0.9)
        seed_optimizer = None

        tracker = SignalTracker()
        action_counts = {a.name: 0 for a in SimicAction}
        seeds_created = 0

        for epoch in range(1, max_epochs + 1):
            seed_state = model.seed_state

            # Training phase - mode depends on seed state
            if seed_state is None:
                # No seed - normal training
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    host_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    host_optimizer.step()
            elif seed_state.stage == SeedStage.GERMINATED:
                # Auto-advance to TRAINING
                seed_state.transition(SeedStage.TRAINING)
                seed_optimizer = optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    seed_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    seed_optimizer.step()
            elif seed_state.stage == SeedStage.TRAINING:
                # Seed isolated training
                if seed_optimizer is None:
                    seed_optimizer = optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    seed_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    seed_optimizer.step()
            elif seed_state.stage in (SeedStage.BLENDING, SeedStage.FOSSILIZED):
                # Blending or fossilized - joint training
                if seed_state.stage == SeedStage.BLENDING:
                    step = seed_state.metrics.epochs_in_current_stage
                    model.seed_slot.update_alpha_for_step(step)
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    host_optimizer.zero_grad()
                    if seed_optimizer:
                        seed_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    host_optimizer.step()
                    if seed_optimizer:
                        seed_optimizer.step()
            else:
                # Fallback - normal training
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    host_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    host_optimizer.step()

            # Validation phase
            model.eval()
            train_loss, train_correct, train_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()
                    _, pred = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += pred.eq(labels).sum().item()
            train_loss /= len(trainloader)
            train_acc = 100.0 * train_correct / train_total

            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()
            val_loss /= len(testloader)
            val_acc = 100.0 * val_correct / val_total

            # Update seed metrics if active
            if model.has_active_seed:
                model.seed_state.metrics.record_accuracy(val_acc)

            # Update signal tracker
            active_seeds = [model.seed_state] if model.has_active_seed else []
            available_slots = 0 if model.has_active_seed else 1
            signals = tracker.update(
                epoch=epoch,
                global_step=epoch * len(trainloader),
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                active_seeds=active_seeds,
                available_slots=available_slots,
            )

            # Get action from policy
            action = action_fn(signals, model, tracker, use_telemetry)
            action_counts[action.name] += 1

            # Execute action
            if action == SimicAction.GERMINATE:
                if not model.has_active_seed:
                    seed_id = f"seed_{seeds_created}"
                    model.germinate_seed("conv_enhance", seed_id)
                    seeds_created += 1
                    seed_optimizer = None

            elif action == SimicAction.ADVANCE:
                if model.has_active_seed:
                    if model.seed_state.stage == SeedStage.TRAINING:
                        model.seed_state.transition(SeedStage.BLENDING)
                        model.seed_state.metrics.reset_stage_baseline()
                        model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                    elif model.seed_state.stage == SeedStage.BLENDING:
                        model.seed_state.transition(SeedStage.FOSSILIZED)
                        model.seed_state.metrics.reset_stage_baseline()
                        model.seed_slot.set_alpha(1.0)

            elif action == SimicAction.CULL:
                if model.has_active_seed:
                    model.cull_seed()
                    seed_optimizer = None

        return val_acc, action_counts

    # Define policy action functions
    def heuristic_action_fn(signals, model, tracker, use_telemetry):
        """Get action from heuristic Tamiyo."""
        tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
        decision = tamiyo.decide(signals)
        # Map TamiyoAction to SimicAction
        action_name = decision.action.name
        if action_name == "WAIT":
            return SimicAction.WAIT
        elif action_name == "GERMINATE":
            return SimicAction.GERMINATE
        elif action_name in ("ADVANCE_TRAINING", "ADVANCE_BLENDING", "ADVANCE_FOSSILIZE"):
            return SimicAction.ADVANCE
        elif action_name in ("CULL", "CHANGE_BLUEPRINT"):
            return SimicAction.CULL
        return SimicAction.WAIT

    def iql_action_fn(signals, model, tracker, use_telemetry):
        """Get action from IQL policy."""
        # Build features
        loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
        while len(loss_hist) < 5:
            loss_hist.insert(0, 0.0)
        acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
        while len(acc_hist) < 5:
            acc_hist.insert(0, 0.0)

        available_slots = 0 if model.has_active_seed else 1
        snapshot = TrainingSnapshot(
            epoch=signals.epoch,
            global_step=signals.global_step,
            train_loss=signals.train_loss,
            val_loss=signals.val_loss,
            loss_delta=signals.loss_delta,
            train_accuracy=signals.train_accuracy,
            val_accuracy=signals.val_accuracy,
            accuracy_delta=signals.accuracy_delta,
            plateau_epochs=signals.plateau_epochs,
            best_val_accuracy=signals.best_val_accuracy,
            best_val_loss=min(signals.loss_history) if signals.loss_history else float('inf'),
            loss_history_5=tuple(loss_hist),
            accuracy_history_5=tuple(acc_hist),
            has_active_seed=model.has_active_seed,
            available_slots=available_slots,
        )

        features = snapshot_to_features(snapshot, use_telemetry=use_telemetry)
        state_tensor = torch.tensor([features], dtype=torch.float32, device=device)
        action_idx = iql.get_action(state_tensor, deterministic=True)
        return SimicAction(action_idx)

    # Run episodes
    for ep_idx in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{n_episodes}")
        print(f"{'='*60}")

        base_seed = 42 + ep_idx * 1000

        # Run heuristic policy
        print(f"\nRunning HEURISTIC policy (seed={base_seed})...")
        h_acc, h_actions = run_training_episode("heuristic", heuristic_action_fn, base_seed)
        print(f"  Final accuracy: {h_acc:.2f}%")
        for action, count in h_actions.items():
            results['action_counts']['heuristic'][action] += count

        # Run IQL policy (SAME seed for fair comparison)
        print(f"\nRunning IQL policy (seed={base_seed})...")
        iql_acc, iql_actions = run_training_episode("iql", iql_action_fn, base_seed)
        print(f"  Final accuracy: {iql_acc:.2f}%")
        for action, count in iql_actions.items():
            results['action_counts']['iql'][action] += count

        # Record results
        results['heuristic_accuracies'].append(h_acc)
        results['iql_accuracies'].append(iql_acc)

        # Determine winner
        if iql_acc > h_acc + 0.5:  # 0.5% threshold to avoid noise
            results['iql_wins'] += 1
            winner = "IQL"
        elif h_acc > iql_acc + 0.5:
            results['heuristic_wins'] += 1
            winner = "Heuristic"
        else:
            results['ties'] += 1
            winner = "TIE"

        print(f"\n  WINNER: {winner} (H={h_acc:.2f}% vs IQL={iql_acc:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    avg_h = sum(results['heuristic_accuracies']) / len(results['heuristic_accuracies'])
    avg_iql = sum(results['iql_accuracies']) / len(results['iql_accuracies'])

    print("\nAverage Accuracy:")
    print(f"  Heuristic: {avg_h:.2f}%")
    print(f"  IQL:       {avg_iql:.2f}%")
    print(f"  Δ:         {avg_iql - avg_h:+.2f}%")

    print("\nWin/Loss Record:")
    print(f"  IQL wins:       {results['iql_wins']}")
    print(f"  Heuristic wins: {results['heuristic_wins']}")
    print(f"  Ties:           {results['ties']}")

    print("\nAction Distributions:")
    print(f"  {'Action':<12} {'Heuristic':>10} {'IQL':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    h_total = sum(results['action_counts']['heuristic'].values())
    i_total = sum(results['action_counts']['iql'].values())

    for action in SimicAction:
        h_pct = results['action_counts']['heuristic'][action.name] / h_total * 100 if h_total > 0 else 0
        i_pct = results['action_counts']['iql'][action.name] / i_total * 100 if i_total > 0 else 0
        print(f"  {action.name:<12} {h_pct:>9.1f}% {i_pct:>9.1f}%")

    # Final verdict
    print(f"\n{'='*60}")
    if results['iql_wins'] > results['heuristic_wins']:
        print("VERDICT: IQL is BETTER than heuristic Tamiyo! 🎉")
    elif results['heuristic_wins'] > results['iql_wins']:
        print("VERDICT: Heuristic Tamiyo is better than IQL.")
    else:
        print("VERDICT: It's a TIE - no clear winner.")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Tamiyo Phase 3 (IQL/CQL)")
    parser.add_argument("--pack", help="Path to data pack JSON (required for training)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.7, help="Expectile parameter")
    parser.add_argument("--beta", type=float, default=3.0, help="Policy temperature")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cql-alpha", type=float, default=0.0,
                        help="CQL regularization strength (0=pure IQL, >0 adds CQL penalty)")
    parser.add_argument("--reward-shaping", action="store_true",
                        help="Use potential-based reward shaping (PBRS) to bridge temporal gap")
    parser.add_argument("--no-telemetry", action="store_true", help="Use only base features")
    parser.add_argument("--save", help="Path to save trained model")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--compare", action="store_true",
                        help="Run live comparison (observation mode) instead of training")
    parser.add_argument("--head-to-head", action="store_true",
                        help="Run head-to-head comparison (each policy executes)")
    parser.add_argument("--model", help="Path to trained model (for --compare/--head-to-head)")
    parser.add_argument("--compare-episodes", type=int, default=5,
                        help="Number of episodes for comparison")
    parser.add_argument("--max-epochs", type=int, default=25,
                        help="Max epochs per episode in head-to-head")
    args = parser.parse_args()

    if args.head_to_head:
        if not args.model:
            parser.error("--head-to-head requires --model")
        head_to_head_comparison(
            model_path=args.model,
            n_episodes=args.compare_episodes,
            max_epochs=args.max_epochs,
            device=args.device,
            use_telemetry=not args.no_telemetry,
        )
    elif args.compare:
        if not args.model:
            parser.error("--compare requires --model")
        live_comparison(
            model_path=args.model,
            n_episodes=args.compare_episodes,
            device=args.device,
            use_telemetry=not args.no_telemetry,
        )
    else:
        if not args.pack:
            parser.error("--pack is required for training")
        train_iql(
            pack_path=args.pack,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            beta=args.beta,
            lr=args.lr,
            cql_alpha=args.cql_alpha,
            use_telemetry=not args.no_telemetry,
            use_reward_shaping=args.reward_shaping,
            device=args.device,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
