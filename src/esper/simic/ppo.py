"""Simic PPO Module - Online Policy Gradient Training

This module implements online PPO for the Tamiyo seed lifecycle controller.
Unlike offline RL (IQL/CQL), this trains by actually running episodes and
learning from the resulting rewards.

Key Features:
- Actor-Critic architecture with shared features
- Clipped surrogate objective (PPO-Clip)
- Generalized Advantage Estimation (GAE)
- Entropy bonus for exploration
- Online rollout collection

Usage:
    # Train PPO agent
    PYTHONPATH=src .venv/bin/python -m esper.simic_ppo \
        --episodes 100 --device cuda:0

    # Train and save
    PYTHONPATH=src .venv/bin/python -m esper.simic_ppo \
        --episodes 100 --save models/ppo_tamiyo.pt

    # Compare trained PPO vs heuristic
    PYTHONPATH=src .venv/bin/python -m esper.simic_ppo \
        --compare --model models/ppo_tamiyo.pt
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from esper.leyline import SimicAction
from esper.simic.rewards import compute_shaped_reward, SeedInfo
from esper.tamiyo import SignalTracker


# =============================================================================
# Observation Normalization (Running Mean/Std)
# =============================================================================

class RunningMeanStd:
    """Running mean and std for observation normalization.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon  # Prevent div by zero
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Update running stats with new batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor,
                             batch_count: int) -> None:
        """Update using batch moments (Welford's algorithm)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize observation using running stats."""
        return torch.clamp((x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + self.epsilon),
                          -clip, clip)

    def to(self, device: str) -> "RunningMeanStd":
        """Move stats to device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self


# =============================================================================
# Rollout Storage
# =============================================================================

class RolloutStep(NamedTuple):
    """Single step in a rollout."""
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    steps: list[RolloutStep] = field(default_factory=list)

    def add(self, state: torch.Tensor, action: int, log_prob: float,
            value: float, reward: float, done: bool):
        self.steps.append(RolloutStep(state, action, log_prob, value, reward, done))

    def clear(self):
        self.steps = []

    def __len__(self):
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

        # Work backwards from last step
        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n_steps)):
            step = self.steps[t]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            if step.done:
                next_value = 0.0
                last_gae = 0.0

            delta = step.reward + gamma * next_value - step.value

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae

            # Returns for value function update
            returns[t] = advantages[t] + step.value

            next_value = step.value

        return returns, advantages

    def get_batches(self, batch_size: int, device: str) -> list[dict]:
        """Get shuffled minibatches for PPO update.

        Returns list of dicts with keys: states, actions, old_log_probs, returns, advantages
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
# Actor-Critic Network
# =============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Uses shared feature extraction with separate actor and critic heads.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for actor output (more uniform initial policy)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

        # Smaller init for critic output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Forward pass returning action distribution and value.

        Args:
            state: State tensor of shape (batch, state_dim)

        Returns:
            Tuple of (action_distribution, value)
        """
        features = self.shared(state)

        logits = self.actor(features)
        dist = Categorical(logits=logits)

        value = self.critic(features).squeeze(-1)

        return dist, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> tuple[int, float, float]:
        """Sample action from policy.

        Args:
            state: State tensor of shape (1, state_dim)
            deterministic: If True, return argmax action

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            dist, value = self.forward(state)

            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def get_action_batch(self, states: torch.Tensor, deterministic: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for a batch of states (keeps tensors on device).

        Args:
            states: State tensor of shape (n_envs, state_dim)
            deterministic: If True, return argmax actions

        Returns:
            Tuple of (actions, log_probs, values) - all tensors on device
        """
        with torch.no_grad():
            dist, values = self.forward(states)

            if deterministic:
                actions = dist.probs.argmax(dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)

            return actions, log_probs, values

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            states: State tensor of shape (batch, state_dim)
            actions: Action tensor of shape (batch,)

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        dist, values = self.forward(states)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent:
    """PPO agent for training Tamiyo."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,  # WAIT, GERMINATE_* (4 variants), ADVANCE, CULL
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda:0",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        # Network
        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.train_steps = 0

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> tuple[int, float, float]:
        """Get action for current state."""
        return self.network.get_action(state, deterministic)

    def store_transition(self, state: torch.Tensor, action: int, log_prob: float,
                         value: float, reward: float, done: bool):
        """Store transition in buffer."""
        self.buffer.add(state, action, log_prob, value, reward, done)

    def update(self, last_value: float = 0.0) -> dict:
        """Perform PPO update.

        Args:
            last_value: Value estimate for state after rollout (0 if done)

        Returns:
            Dict with training metrics
        """
        if len(self.buffer) == 0:
            return {}

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages (critical for stable training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Move to device
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # PPO epochs
        metrics = defaultdict(list)

        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size, self.device)

            for batch, batch_idx in batches:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                batch_returns = returns[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)

                # Get current policy outputs
                log_probs, values, entropy = self.network.evaluate_actions(states, actions)

                # Policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['approx_kl'].append(((ratio - 1) - (ratio.log())).mean().item())
                metrics['clip_fraction'].append(
                    ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                )

        self.train_steps += 1
        self.buffer.clear()

        # Average metrics
        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def save(self, path: str | Path, metadata: dict = None):
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'config': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
            }
        }
        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda:0") -> "PPOAgent":
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Get state dim from weights
        state_dim = checkpoint['network_state_dict']['shared.0.weight'].shape[1]
        action_dim = checkpoint['network_state_dict']['actor.2.weight'].shape[0]

        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **checkpoint.get('config', {})
        )

        agent.network.load_state_dict(checkpoint['network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train_steps = checkpoint.get('train_steps', 0)

        return agent


# =============================================================================
# Feature Extraction (reused from simic_iql.py)
# =============================================================================

def safe(v, default=0.0, max_val=100.0):
    """Safely convert value to float."""
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return max(-max_val, min(float(v), max_val))


def signals_to_features(signals, model, tracker=None, use_telemetry: bool = True) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo module
        model: MorphogeneticModel
        tracker: Optional DiagnosticTracker for V2 features
        use_telemetry: Whether to include telemetry features

    Returns:
        Feature vector (27 dims base, +27 if telemetry)
    """
    # Loss history (5 values)
    loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
    while len(loss_hist) < 5:
        loss_hist.insert(0, 0.0)

    # Accuracy history (5 values)
    acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
    while len(acc_hist) < 5:
        acc_hist.insert(0, 0.0)

    # Seed state from signals (already tracked)
    has_active_seed = 1.0 if signals.active_seeds else 0.0
    if signals.active_seeds:
        seed_state = signals.active_seeds[0]
        seed_stage = seed_state.stage.value if seed_state else 0
        seed_epochs = seed_state.epochs_in_stage if seed_state else 0
        seed_alpha = model.seed_slot.alpha if model.seed_slot else 0.0
        # Compute improvement from metrics
        if seed_state and seed_state.metrics:
            seed_improvement = seed_state.metrics.current_val_accuracy - seed_state.metrics.accuracy_at_stage_start
        else:
            seed_improvement = 0.0
    else:
        seed_stage = 0
        seed_epochs = 0
        seed_alpha = 0.0
        seed_improvement = 0.0

    available_slots = signals.available_slots

    # Base features (27 dims)
    features = [
        float(signals.epoch),
        float(signals.global_step),
        safe(signals.train_loss, 10.0),
        safe(signals.val_loss, 10.0),
        safe(signals.loss_delta, 0.0),
        signals.train_accuracy,
        signals.val_accuracy,
        safe(signals.accuracy_delta, 0.0),
        float(signals.plateau_epochs),
        signals.best_val_accuracy,
        min(signals.loss_history) if signals.loss_history else 10.0,
        *[safe(v, 10.0) for v in loss_hist],
        *acc_hist,
        has_active_seed,
        float(seed_stage),
        float(seed_epochs),
        seed_alpha,
        seed_improvement,
        float(available_slots),
    ]

    # Telemetry features (27 dims)
    if use_telemetry and tracker:
        telem = tracker.compute_telemetry()
        features.extend(telemetry_to_features(telem))
    elif use_telemetry:
        features.extend([0.0] * 27)

    return features


def telemetry_to_features(telem: dict) -> list[float]:
    """Extract V2 telemetry features (27 dims)."""
    features = []

    # Gradient health (5 features)
    gh = telem.get('gradient_health', {})
    features.extend([
        safe(gh.get('overall_norm', 0), 0, 10),
        safe(gh.get('norm_variance', 0), 0, 10),
        float(gh.get('vanishing_layers', 0)),
        float(gh.get('exploding_layers', 0)),
        safe(gh.get('health_score', 1), 1, 1),
    ])

    # Per-class accuracy (10 features)
    pca = telem.get('per_class_accuracy', {})
    for i in range(10):
        features.append(safe(pca.get(str(i), 50), 50, 100))

    # Class variance (1 feature)
    features.append(safe(telem.get('class_variance', 0), 0, 1000))

    # Sharpness (1 feature)
    features.append(safe(telem.get('sharpness', 0), 0, 100))

    # Gradient stats per layer (7 features)
    gs = telem.get('gradient_stats', [])
    layer_norms = [safe(g.get('norm', 0), 0, 10) for g in gs[:7]]
    while len(layer_norms) < 7:
        layer_norms.append(0.0)
    features.extend(layer_norms)

    # Red flags (3 features)
    rf = telem.get('red_flags', [])
    features.append(1.0 if 'severe_class_imbalance' in rf else 0.0)
    features.append(1.0 if 'sharp_minimum' in rf else 0.0)
    features.append(1.0 if 'gradient_issues' in rf else 0.0)

    return features


# Reward shaping: see esper.rewards module for compute_shaped_reward()


# =============================================================================
# Training Episode
# =============================================================================

def run_episode(
    agent: PPOAgent,
    trainloader,
    testloader,
    max_epochs: int = 25,
    base_seed: int = 42,
    device: str = "cuda:0",
    use_telemetry: bool = True,
    collect_rollout: bool = True,
    deterministic: bool = False,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with the PPO agent.

    Args:
        agent: PPO agent
        trainloader: Training data loader
        testloader: Test data loader
        max_epochs: Maximum training epochs
        base_seed: Random seed
        device: Device for training
        use_telemetry: Whether to use telemetry features
        collect_rollout: Whether to store transitions (False for eval)
        deterministic: Whether to use deterministic actions (for eval)

    Returns:
        Tuple of (final_accuracy, action_counts, episode_rewards)
    """
    # Lazy imports to avoid circular deps
    from esper.leyline import SeedStage
    from esper.simic_overnight import create_model

    # Set seeds
    torch.manual_seed(base_seed)
    random.seed(base_seed)

    # Setup
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer for host model
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None

    signal_tracker = SignalTracker()

    seeds_created = 0
    action_counts = {a.name: 0 for a in SimicAction}
    episode_rewards = []

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_state

        # Training phase - mode depends on seed state
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if seed_state is None:
            # No seed - normal training
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.GERMINATED:
            # Auto-advance to TRAINING
            seed_state.transition(SeedStage.TRAINING)
            seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.TRAINING:
            # Seed isolated training
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.BLENDING:
            # Train both, blending
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                if seed_optimizer:
                    seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                if seed_optimizer:
                    seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.FOSSILIZED:
            # Train host only
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(testloader)
        val_acc = 100.0 * correct / total

        # Update signal tracker
        active_seeds = [seed_state] if seed_state else []
        available_slots = 0 if model.has_active_seed else 1
        signals = signal_tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=available_slots,
        )

        acc_delta = signals.accuracy_delta
        prev_val_acc = signals.val_accuracy - acc_delta if acc_delta else 0

        # Get features and action from agent (no telemetry for online PPO)
        features = signals_to_features(signals, model, tracker=None, use_telemetry=False)
        state = torch.tensor([features], dtype=torch.float32, device=device)

        action_idx, log_prob, value = agent.get_action(state, deterministic=deterministic)
        action = SimicAction(action_idx)
        action_counts[action.name] += 1

        # Compute shaped reward for seed lifecycle learning
        reward = compute_shaped_reward(
            action=action.value,
            acc_delta=acc_delta,
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state),
            epoch=epoch,
            max_epochs=max_epochs,
        )

        # Execute action
        if SimicAction.is_germinate(action):
            if not model.has_active_seed:
                blueprint_id = SimicAction.get_blueprint_id(action)
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id)
                seeds_created += 1
                seed_optimizer = None  # Will be created next epoch

        elif action == SimicAction.ADVANCE:
            if model.has_active_seed:
                if model.seed_state.stage == SeedStage.TRAINING:
                    model.seed_state.transition(SeedStage.BLENDING)
                    model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                elif model.seed_state.stage == SeedStage.BLENDING:
                    model.seed_state.transition(SeedStage.FOSSILIZED)
                    model.seed_slot.set_alpha(1.0)

        elif action == SimicAction.CULL:
            if model.has_active_seed:
                model.cull_seed()
                seed_optimizer = None

        # Store transition
        done = (epoch == max_epochs)

        if collect_rollout:
            # Store state (not on device to save memory)
            agent.store_transition(
                state.squeeze(0).cpu(),
                action_idx,
                log_prob,
                value,
                reward,  # Shaped rewards already scaled appropriately
                done
            )

        episode_rewards.append(reward)

    return val_acc, action_counts, episode_rewards


# =============================================================================
# Training Loop
# =============================================================================

def train_ppo(
    n_episodes: int = 100,
    max_epochs: int = 25,
    update_every: int = 5,
    device: str = "cuda:0",
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    save_path: str = None,
) -> tuple[PPOAgent, list[dict]]:
    """Train PPO agent.

    Args:
        n_episodes: Number of training episodes
        max_epochs: Max epochs per episode
        update_every: Update after this many episodes
        device: Device for training
        use_telemetry: Whether to use telemetry features
        lr: Learning rate
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        gamma: Discount factor
        save_path: Optional path to save model

    Returns:
        Tuple of (trained_agent, training_history)
    """
    from esper.simic_overnight import load_cifar10

    print("=" * 60)
    print("PPO Training for Tamiyo")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Max epochs per episode: {max_epochs}")
    print(f"Update every: {update_every} episodes")
    print(f"Device: {device}")
    print(f"Telemetry: {use_telemetry}")
    print()

    # Load CIFAR-10 once
    print("Loading CIFAR-10...")
    trainloader, testloader = load_cifar10(batch_size=128)

    # Determine state dimension
    # For now, we disable telemetry in online PPO (base features only = 27 dims)
    state_dim = 27

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=len(SimicAction),
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        gamma=gamma,
        device=device,
    )

    history = []
    best_avg_acc = 0.0
    best_state = None

    # Rolling stats
    recent_accuracies = []
    recent_rewards = []

    for ep in range(1, n_episodes + 1):
        base_seed = 42 + ep * 1000

        # Run episode
        final_acc, action_counts, rewards = run_episode(
            agent=agent,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=max_epochs,
            base_seed=base_seed,
            device=device,
            use_telemetry=use_telemetry,
            collect_rollout=True,
            deterministic=False,
        )

        total_reward = sum(rewards)
        recent_accuracies.append(final_acc)
        recent_rewards.append(total_reward)

        # Keep rolling window
        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        # Update every N episodes
        if ep % update_every == 0 or ep == n_episodes:
            # Get last value estimate (0 since episodes end)
            metrics = agent.update(last_value=0.0)

            avg_acc = sum(recent_accuracies) / len(recent_accuracies)
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            print(f"Episode {ep:3d}/{n_episodes}: "
                  f"acc={final_acc:.1f}% (avg={avg_acc:.1f}%), "
                  f"reward={total_reward:.1f} (avg={avg_reward:.1f})")
            print(f"  Actions: {dict(action_counts)}")
            if metrics:
                print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
                      f"Value loss: {metrics['value_loss']:.4f}, "
                      f"Entropy: {metrics['entropy']:.4f}")

            history.append({
                'episode': ep,
                'accuracy': final_acc,
                'avg_accuracy': avg_acc,
                'total_reward': total_reward,
                'action_counts': dict(action_counts),
                **metrics,
            })

            # Save best
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_state = {k: v.clone() for k, v in agent.network.state_dict().items()}
        else:
            # Just log basic info
            if ep % 10 == 0:
                print(f"Episode {ep:3d}/{n_episodes}: acc={final_acc:.1f}%, reward={total_reward:.1f}")

    # Load best weights
    if best_state:
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")

    # Save if requested
    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': n_episodes,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
            'use_telemetry': use_telemetry,
        })
        print(f"Model saved to {save_path}")

    return agent, history


# =============================================================================
# Vectorized Environment Training
# =============================================================================

@dataclass
class ParallelEnvState:
    """State for a single parallel environment with CUDA stream for async execution.

    Each environment has its own DataLoader instances to avoid GIL contention
    when multiple environments iterate in parallel. This eliminates the bottleneck
    of a shared iterator being accessed from multiple CUDA streams.
    """
    model: nn.Module
    host_optimizer: torch.optim.Optimizer
    seed_optimizer: torch.optim.Optimizer | None
    signal_tracker: "SignalTracker"
    env_device: str = "cuda:0"  # Device this env runs on
    stream: torch.cuda.Stream | None = None  # CUDA stream for async execution
    # Independent DataLoaders per env to avoid GIL contention
    trainloader: any = None  # Environment's own trainloader
    testloader: any = None   # Environment's own testloader
    train_iter: any = None  # Persistent dataloader iterator
    test_iter: any = None   # Persistent testloader iterator
    seeds_created: int = 0
    episode_rewards: list = field(default_factory=list)
    action_counts: dict = field(default_factory=lambda: {a.name: 0 for a in SimicAction})
    # Metrics for current batch step
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0


def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = 4,
    max_epochs: int = 25,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    use_telemetry: bool = False,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.1,
    gamma: float = 0.99,
    save_path: str = None,
) -> tuple[PPOAgent, list[dict]]:
    """Train PPO with vectorized environments using INVERTED CONTROL FLOW.

    Key architecture: Instead of iterating environments then dataloaders,
    we iterate dataloader batches FIRST, then run all environments in parallel
    using CUDA streams. This ensures both GPUs are working simultaneously.

    Args:
        n_episodes: Total episodes to train
        n_envs: Number of parallel environments
        max_epochs: Max epochs per episode (RL timesteps per episode)
        device: Device for policy network
        devices: List of devices for environments (e.g., ["cuda:0", "cuda:1"])
        use_telemetry: Whether to use telemetry features
        lr: Learning rate
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        gamma: Discount factor
        save_path: Optional path to save model

    Returns:
        Tuple of (trained_agent, training_history)
    """
    from esper.simic_overnight import load_cifar10, create_model
    from esper.tamiyo import SignalTracker
    from esper.leyline import SeedStage

    if devices is None:
        devices = [device]

    print("=" * 60)
    print("PPO Vectorized Training (INVERTED CONTROL FLOW + CUDA STREAMS)")
    print("=" * 60)
    print(f"Episodes: {n_episodes} (across {n_envs} parallel envs)")
    print(f"Max epochs per episode: {max_epochs}")
    print(f"Policy device: {device}")
    print(f"Env devices: {devices} ({n_envs // len(devices)} envs per device)")
    print(f"Entropy coef: {entropy_coef}")
    print(f"Learning rate: {lr}")
    print()

    # Create independent DataLoaders per environment to avoid GIL contention.
    # When multiple CUDA streams try to iterate a shared DataLoader, the GIL
    # serializes access. Independent DataLoaders with unique random seeds
    # allow true parallel data loading.
    print(f"Loading CIFAR-10 ({n_envs} independent DataLoaders)...")

    def create_env_dataloaders(env_idx: int, base_seed: int):
        """Create DataLoaders with unique random seed for this environment."""
        # Use unique seed per environment for data shuffling
        gen = torch.Generator()
        gen.manual_seed(base_seed + env_idx * 7919)  # Prime number for seed spacing
        return load_cifar10(batch_size=128, generator=gen)

    # Get batch counts from first loader (all have same size)
    sample_train, sample_test = load_cifar10(batch_size=128)
    num_train_batches = len(sample_train)
    num_test_batches = len(sample_test)
    del sample_train, sample_test  # Free memory

    # State dimension and observation normalizer
    state_dim = 27
    obs_normalizer = RunningMeanStd((state_dim,))

    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=len(SimicAction),
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        gamma=gamma,
        device=device,
    )

    # Map environments to devices in round-robin
    env_device_map = [devices[i % len(devices)] for i in range(n_envs)]

    def create_env_state(env_idx: int, base_seed: int) -> ParallelEnvState:
        """Create environment state with CUDA stream and independent DataLoaders."""
        env_device = env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model = create_model(env_device)
        host_optimizer = torch.optim.SGD(
            model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        # Create CUDA stream for this environment
        stream = torch.cuda.Stream(device=env_device) if 'cuda' in env_device else None

        # Create independent DataLoaders for this environment
        env_trainloader, env_testloader = create_env_dataloaders(env_idx, base_seed)

        return ParallelEnvState(
            model=model,
            host_optimizer=host_optimizer,
            seed_optimizer=None,
            signal_tracker=SignalTracker(),
            env_device=env_device,
            stream=stream,
            trainloader=env_trainloader,
            testloader=env_testloader,
            train_iter=None,  # Will be set per epoch
            test_iter=None,
            seeds_created=0,
            episode_rewards=[],
            action_counts={a.name: 0 for a in SimicAction},
        )

    def process_train_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                            targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process a single training batch for one environment (runs in CUDA stream).

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.
        """
        model = env_state.model
        seed_state = model.seed_state
        env_dev = env_state.env_device

        # Use CUDA stream for async execution
        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            # Move data asynchronously
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            model.train()

            # Determine which optimizer to use based on seed state
            if seed_state is None or seed_state.stage == SeedStage.FOSSILIZED:
                optimizer = env_state.host_optimizer
            elif seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING):
                if seed_state.stage == SeedStage.GERMINATED:
                    seed_state.transition(SeedStage.TRAINING)
                    env_state.seed_optimizer = torch.optim.SGD(
                        model.get_seed_parameters(), lr=0.01, momentum=0.9
                    )
                if env_state.seed_optimizer is None:
                    env_state.seed_optimizer = torch.optim.SGD(
                        model.get_seed_parameters(), lr=0.01, momentum=0.9
                    )
                optimizer = env_state.seed_optimizer
            else:  # BLENDING
                optimizer = env_state.host_optimizer

            optimizer.zero_grad()
            if seed_state and seed_state.stage == SeedStage.BLENDING and env_state.seed_optimizer:
                env_state.seed_optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            if seed_state and seed_state.stage == SeedStage.BLENDING and env_state.seed_optimizer:
                env_state.seed_optimizer.step()

            _, predicted = outputs.max(1)
            correct_tensor = predicted.eq(targets).sum()  # Keep as tensor, no .item()

            # Return tensors - .item() called after stream sync
            return loss.detach(), correct_tensor, targets.size(0)

    def process_val_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                          targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process a validation batch for one environment.

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.
        """
        model = env_state.model
        env_dev = env_state.env_device

        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct_tensor = predicted.eq(targets).sum()  # Keep as tensor, no .item()

            # Return tensors - .item() called after stream sync
            return loss, correct_tensor, targets.size(0)

    history = []
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []

    episodes_per_env = (n_episodes + n_envs - 1) // n_envs

    for batch_idx in range(episodes_per_env):
        # Create fresh environments for this batch
        base_seed = 42 + batch_idx * 10000
        env_states = [create_env_state(i, base_seed) for i in range(n_envs)]
        criterion = nn.CrossEntropyLoss()

        # Per-env accumulators
        env_final_accs = [0.0] * n_envs
        env_total_rewards = [0.0] * n_envs

        # Run epochs with INVERTED CONTROL FLOW
        for epoch in range(1, max_epochs + 1):
            # Reset per-epoch metrics
            train_losses = [0.0] * n_envs
            train_corrects = [0] * n_envs
            train_totals = [0] * n_envs

            # ===== TRAINING: Iterate batches first, launch all envs via CUDA streams =====
            # Use first env's DataLoader as shared batch source (all envs see same data)
            train_iter = iter(env_states[0].trainloader)
            for batch_step in range(num_train_batches):
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    break

                # Launch all environments in their respective CUDA streams (async)
                env_results = []
                for i, env_state in enumerate(env_states):
                    result = process_train_batch(env_state, inputs.clone(), targets.clone(), criterion)
                    env_results.append((i, result))

                # Sync all streams after launching all work (THE BARRIER)
                for env_state in env_states:
                    if env_state.stream:
                        env_state.stream.synchronize()

                # NOW safe to call .item() - GPUs are done
                for env_idx, (loss_tensor, correct_tensor, total) in env_results:
                    train_losses[env_idx] += loss_tensor.item()
                    train_corrects[env_idx] += correct_tensor.item()
                    train_totals[env_idx] += total

            # ===== VALIDATION: Same inverted pattern with CUDA streams =====
            val_losses = [0.0] * n_envs
            val_corrects = [0] * n_envs
            val_totals = [0] * n_envs

            # Use first env's testloader as shared batch source
            test_iter = iter(env_states[0].testloader)
            for batch_step in range(num_test_batches):
                try:
                    inputs, targets = next(test_iter)
                except StopIteration:
                    break

                # Launch all environments in their respective CUDA streams (async)
                env_results = []
                for i, env_state in enumerate(env_states):
                    result = process_val_batch(env_state, inputs.clone(), targets.clone(), criterion)
                    env_results.append((i, result))

                # Sync all streams (THE BARRIER)
                for env_state in env_states:
                    if env_state.stream:
                        env_state.stream.synchronize()

                # NOW safe to call .item() - GPUs are done
                for env_idx, (loss_tensor, correct_tensor, total) in env_results:
                    val_losses[env_idx] += loss_tensor.item()
                    val_corrects[env_idx] += correct_tensor.item()
                    val_totals[env_idx] += total

            # ===== Compute epoch metrics and get BATCHED actions =====
            # Collect features from all environments
            all_features = []
            all_signals = []

            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                seed_state = model.seed_state

                train_loss = train_losses[env_idx] / num_train_batches
                train_acc = 100.0 * train_corrects[env_idx] / max(train_totals[env_idx], 1)
                val_loss = val_losses[env_idx] / num_test_batches
                val_acc = 100.0 * val_corrects[env_idx] / max(val_totals[env_idx], 1)

                # Store metrics for later
                env_state.train_loss = train_loss
                env_state.train_acc = train_acc
                env_state.val_loss = val_loss
                env_state.val_acc = val_acc

                # Update signal tracker
                active_seeds = [seed_state] if seed_state else []
                available_slots = 0 if model.has_active_seed else 1
                signals = env_state.signal_tracker.update(
                    epoch=epoch,
                    global_step=epoch * num_train_batches,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    active_seeds=active_seeds,
                    available_slots=available_slots,
                )
                all_signals.append(signals)

                features = signals_to_features(signals, model, tracker=None, use_telemetry=False)
                all_features.append(features)

            # Batch all states into a single tensor
            states_batch = torch.tensor(all_features, dtype=torch.float32, device=device)

            # Update observation normalizer and normalize
            obs_normalizer.update(states_batch.cpu())
            states_batch = obs_normalizer.normalize(states_batch)

            # Get BATCHED actions from policy network (single forward pass!)
            actions, log_probs, values = agent.network.get_action_batch(states_batch, deterministic=False)

            # Execute actions and store transitions for each environment
            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                seed_state = model.seed_state
                signals = all_signals[env_idx]

                action_idx = actions[env_idx].item()
                log_prob = log_probs[env_idx].item()
                value = values[env_idx].item()

                action = SimicAction(action_idx)
                env_state.action_counts[action.name] += 1

                # Compute shaped reward
                reward = compute_shaped_reward(
                    action=action.value,
                    acc_delta=signals.accuracy_delta,
                    val_acc=env_state.val_acc,
                    seed_info=SeedInfo.from_seed_state(seed_state),
                    epoch=epoch,
                    max_epochs=max_epochs,
                )

                # Execute action
                if SimicAction.is_germinate(action):
                    if not model.has_active_seed:
                        blueprint_id = SimicAction.get_blueprint_id(action)
                        seed_id = f"env{env_idx}_seed_{env_state.seeds_created}"
                        model.germinate_seed(blueprint_id, seed_id)
                        env_state.seeds_created += 1
                        env_state.seed_optimizer = None

                elif action == SimicAction.ADVANCE:
                    if model.has_active_seed:
                        if model.seed_state.stage == SeedStage.TRAINING:
                            model.seed_state.transition(SeedStage.BLENDING)
                            model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                        elif model.seed_state.stage == SeedStage.BLENDING:
                            model.seed_state.transition(SeedStage.FOSSILIZED)
                            model.seed_slot.set_alpha(1.0)

                elif action == SimicAction.CULL:
                    if model.has_active_seed:
                        model.cull_seed()
                        env_state.seed_optimizer = None

                # Store transition
                done = (epoch == max_epochs)
                agent.store_transition(
                    states_batch[env_idx].cpu(),
                    action_idx,
                    log_prob,
                    value,
                    reward,
                    done
                )

                env_state.episode_rewards.append(reward)

                if epoch == max_epochs:
                    env_final_accs[env_idx] = env_state.val_acc
                    env_total_rewards[env_idx] = sum(env_state.episode_rewards)

        # PPO Update after all episodes in batch complete
        metrics = agent.update(last_value=0.0)

        # Track results
        avg_acc = sum(env_final_accs) / len(env_final_accs)
        avg_reward = sum(env_total_rewards) / len(env_total_rewards)

        recent_accuracies.append(avg_acc)
        recent_rewards.append(avg_reward)
        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)

        ep_num = (batch_idx + 1) * n_envs
        print(f"Batch {batch_idx + 1}: Episodes {ep_num}/{n_episodes}")
        print(f"  Env accuracies: {[f'{a:.1f}%' for a in env_final_accs]}")
        print(f"  Avg acc: {avg_acc:.1f}% (rolling: {rolling_avg_acc:.1f}%)")
        print(f"  Avg reward: {avg_reward:.1f}")

        total_actions = {a.name: 0 for a in SimicAction}
        for env_state in env_states:
            for a, c in env_state.action_counts.items():
                total_actions[a] += c
        print(f"  Actions: {total_actions}")

        if metrics:
            print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
                  f"Value loss: {metrics['value_loss']:.4f}, "
                  f"Entropy: {metrics['entropy']:.4f}")

        history.append({
            'batch': batch_idx + 1,
            'episodes': ep_num,
            'env_accuracies': list(env_final_accs),
            'avg_accuracy': avg_acc,
            'rolling_avg_accuracy': rolling_avg_acc,
            'avg_reward': avg_reward,
            'action_counts': total_actions,
            **metrics,
        })

        if rolling_avg_acc > best_avg_acc:
            best_avg_acc = rolling_avg_acc
            best_state = {k: v.clone() for k, v in agent.network.state_dict().items()}

    if best_state:
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")

    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': n_episodes,
            'n_envs': n_envs,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
            'use_telemetry': use_telemetry,
            'obs_normalizer_mean': obs_normalizer.mean.tolist(),
            'obs_normalizer_var': obs_normalizer.var.tolist(),
        })
        print(f"Model saved to {save_path}")

    return agent, history


# =============================================================================
# NOTE: Comparison mode removed - the compare_with_heuristic and run_heuristic_episode
# functions were broken (referenced non-existent esper.simic_cifar10 module).
# For PPO vs heuristic comparison, use simic_iql.py --head-to-head instead.


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train PPO for Tamiyo")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--max-epochs", type=int, default=25, help="Max epochs per episode")
    parser.add_argument("--update-every", type=int, default=5, help="Update after N episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry features")
    parser.add_argument("--save", help="Path to save model")
    parser.add_argument("--device", default="cuda:0", help="Device")

    # Vectorized training
    parser.add_argument("--vectorized", action="store_true", help="Use vectorized environments")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--devices", nargs="+", help="Devices for env distribution (e.g. cuda:0 cuda:1)")

    args = parser.parse_args()

    if args.vectorized:
        train_ppo_vectorized(
            n_episodes=args.episodes,
            n_envs=args.n_envs,
            max_epochs=args.max_epochs,
            device=args.device,
            devices=args.devices,
            use_telemetry=not args.no_telemetry,
            lr=args.lr,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            save_path=args.save,
        )
    else:
        train_ppo(
            n_episodes=args.episodes,
            max_epochs=args.max_epochs,
            update_every=args.update_every,
            device=args.device,
            use_telemetry=not args.no_telemetry,
            lr=args.lr,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
