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
        # Use TrainingSignals context when no live model is available (offline/replay)
        obs['seed_stage'] = signals.seed_stage
        obs['seed_epochs_in_stage'] = signals.seed_epochs_in_stage
        obs['seed_alpha'] = signals.seed_alpha
        obs['seed_improvement'] = signals.seed_improvement

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
        entropy_coef: float = 0.015,  # Scaled for normalized entropy (was 0.01 for raw)
        entropy_coef_start: float | None = None,
        entropy_coef_end: float | None = None,
        entropy_coef_min: float = 0.015,  # Scaled for normalized entropy
        entropy_anneal_steps: int = 0,
        value_coef: float = 0.5,
        clip_value: bool = True,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        target_kl: float | None = 0.015,
        device: str = "cuda:0",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.entropy_coef_start = entropy_coef_start if entropy_coef_start is not None else entropy_coef
        self.entropy_coef_end = entropy_coef_end if entropy_coef_end is not None else entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_anneal_steps = entropy_anneal_steps
        self.value_coef = value_coef
        self.clip_value = clip_value
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.device = device

        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.train_steps = 0

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient (annealed if configured).

        Returns fixed entropy_coef when entropy_anneal_steps=0 (legacy behavior).
        Otherwise linearly interpolates from entropy_coef_start to entropy_coef_end
        over entropy_anneal_steps training updates.

        The returned value is always >= entropy_coef_min to prevent exploration
        collapse. Default floor is 0.01 (standard PPO entropy coefficient).
        """
        if self.entropy_anneal_steps == 0:
            return max(self.entropy_coef, self.entropy_coef_min)

        progress = min(1.0, self.train_steps / self.entropy_anneal_steps)
        annealed = self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)
        return max(annealed, self.entropy_coef_min)

    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Get action for current state with masking.

        Args:
            state: Observation tensor
            action_mask: Binary mask of valid actions (1=valid, 0=invalid)
            deterministic: If True, return argmax instead of sampling

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
        """
        return self.network.get_action(state, action_mask, deterministic)

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
    ) -> None:
        """Store transition in buffer.

        Args:
            state: Observation tensor
            action: Action taken
            log_prob: Log probability of action
            value: Value estimate
            reward: Reward received
            done: Whether episode ended
            action_mask: Binary mask of valid actions (stored for PPO update)
        """
        self.buffer.add(state, action, log_prob, value, reward, done, action_mask)

    def update(self, last_value: float = 0.0, clear_buffer: bool = True) -> dict:
        """Perform PPO update.

        Args:
            last_value: Value estimate for bootstrapping (0.0 for terminal states)
            clear_buffer: Whether to clear the rollout buffer after update.
                Set to False if calling multiple times on the same data
                (e.g., for higher sample efficiency via ppo_updates_per_batch).

        Returns:
            Dictionary of training metrics.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute returns and advantages directly on device (avoids CPU->GPU transfer)
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda, device=self.device
        )

        metrics = defaultdict(list)
        early_stopped = False

        for epoch_i in range(self.n_epochs):
            if early_stopped:
                break

            batches = self.buffer.get_batches(self.batch_size, self.device)
            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for batch, batch_idx in batches:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                old_values = batch['values']
                action_masks = batch['action_masks']
                batch_returns = returns[batch_idx]  # Already on device
                batch_advantages = advantages[batch_idx]  # Already on device

                # Per-minibatch advantage normalization for better gradient stability
                # (especially important with small batch sizes)
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                log_probs, values, entropy = self.network.evaluate_actions(states, actions, action_masks)

                # PPO-Clip loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (with optional clipping)
                if self.clip_value:
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -self.clip_ratio, self.clip_ratio
                    )
                    value_loss_unclipped = (values - batch_returns) ** 2
                    value_loss_clipped = (values_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.get_entropy_coef() * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # KL(old||new) first-order approximation: E[log(p_old) - log(p_new)]
                # Note: This is KL(old||new), not KL(new||old). The sign convention
                # matches stable-baselines3 and other PPO implementations for early
                # stopping diagnostics. Positive values indicate the new policy assigns
                # lower probability than old to sampled actions.
                batch_kl = (old_log_probs - log_probs).mean().item()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['approx_kl'].append(batch_kl)
                metrics['clip_fraction'].append(
                    ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                )

                epoch_kl_sum += batch_kl
                epoch_kl_count += 1

            # KL-based early stopping: stop if average KL exceeds 1.5 * target_kl
            # This prevents the policy from diverging too far from the data collection policy,
            # which can destabilize training. (Schulman et al., 2017)
            if self.target_kl is not None and epoch_kl_count > 0:
                epoch_kl_avg = epoch_kl_sum / epoch_kl_count
                if epoch_kl_avg > 1.5 * self.target_kl:
                    early_stopped = True
                    metrics['early_stop_epoch'] = [epoch_i + 1]

        self.train_steps += 1
        if clear_buffer:
            self.buffer.clear()

        result = {k: sum(v) / len(v) for k, v in metrics.items()}
        if early_stopped:
            result['early_stopped'] = 1.0
        return result

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
                'entropy_coef_start': self.entropy_coef_start,
                'entropy_coef_end': self.entropy_coef_end,
                'entropy_coef_min': self.entropy_coef_min,
                'entropy_anneal_steps': self.entropy_anneal_steps,
                'value_coef': self.value_coef,
                'clip_value': self.clip_value,
                'target_kl': self.target_kl,
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
