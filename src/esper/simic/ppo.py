"""Simic PPO Module - PPO Agent for Seed Lifecycle Control

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.simic.buffers import RolloutBuffer
from esper.simic.networks import ActorCritic
from esper.simic.features import safe


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
# =============================================================================

def signals_to_features(signals, model, tracker=None, use_telemetry: bool = True) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo
        model: MorphogeneticModel
        tracker: Optional tracker (unused, kept for API compatibility)
        use_telemetry: Whether to include telemetry features

    Returns:
        Feature vector (27 dims base, +10 if telemetry = 37 dims total)

    Note:
        TrainingSignals.active_seeds contains seed IDs (strings), not SeedState
        objects, so seed-specific features and telemetry are always zero-padded.
    """
    from esper.simic.features import obs_to_base_features

    # Build observation dict
    loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
    while len(loss_hist) < 5:
        loss_hist.insert(0, 0.0)

    acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
    while len(acc_hist) < 5:
        acc_hist.insert(0, 0.0)

    obs = {
        'epoch': signals.metrics.epoch,
        'global_step': signals.metrics.global_step,
        'train_loss': signals.metrics.train_loss,
        'val_loss': signals.metrics.val_loss,
        'loss_delta': signals.metrics.loss_delta,
        'train_accuracy': signals.metrics.train_accuracy,
        'val_accuracy': signals.metrics.val_accuracy,
        'accuracy_delta': signals.metrics.accuracy_delta,
        'plateau_epochs': signals.metrics.plateau_epochs,
        'best_val_accuracy': signals.metrics.best_val_accuracy,
        'best_val_loss': signals.metrics.best_val_loss,
        'loss_history_5': loss_hist,
        'accuracy_history_5': acc_hist,
        'has_active_seed': 1.0 if signals.active_seeds else 0.0,
        'available_slots': signals.available_slots,
    }

    # Seed state features
    # NOTE: TrainingSignals.active_seeds is a list of seed IDs (strings),
    # not SeedState objects. Use model.seed_state to access full seed state.
    if model and model.has_active_seed:
        seed_state = model.seed_state
        obs['seed_stage'] = seed_state.stage.value
        obs['seed_epochs_in_stage'] = seed_state.metrics.epochs_in_current_stage
        obs['seed_alpha'] = seed_state.alpha
        obs['seed_improvement'] = seed_state.metrics.improvement_since_stage_start
    else:
        obs['seed_stage'] = 0
        obs['seed_epochs_in_stage'] = 0
        obs['seed_alpha'] = 0.0
        obs['seed_improvement'] = 0.0

    features = obs_to_base_features(obs)

    if use_telemetry:
        # Use real telemetry from model.seed_state when available
        from esper.leyline import SeedTelemetry
        if model and model.has_active_seed:
            seed_state = model.seed_state
            # SeedState always has telemetry field (initialized in __post_init__)
            features.extend(seed_state.telemetry.to_features())
        else:
            features.extend([0.0] * SeedTelemetry.feature_dim())

    return features


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent:
    """PPO agent for training Tamiyo seed lifecycle controller."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
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

        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.train_steps = 0

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> tuple[int, float, float]:
        """Get action for current state."""
        return self.network.get_action(state, deterministic)

    def store_transition(self, state: torch.Tensor, action: int, log_prob: float,
                         value: float, reward: float, done: bool) -> None:
        """Store transition in buffer."""
        self.buffer.add(state, action, log_prob, value, reward, done)

    def update(self, last_value: float = 0.0) -> dict:
        """Perform PPO update."""
        if len(self.buffer) == 0:
            return {}

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        metrics = defaultdict(list)

        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size, self.device)

            for batch, batch_idx in batches:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                batch_returns = returns[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)

                log_probs, values, entropy = self.network.evaluate_actions(states, actions)

                # PPO-Clip loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['approx_kl'].append(((ratio - 1) - (ratio.log())).mean().item())
                metrics['clip_fraction'].append(
                    ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                )

        self.train_steps += 1
        self.buffer.clear()

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def save(self, path: str | Path, metadata: dict = None) -> None:
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


__all__ = [
    "PPOAgent",
    "signals_to_features",
]
