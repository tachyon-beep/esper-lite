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

from esper.simic.buffers import RolloutBuffer, RecurrentRolloutBuffer
from esper.simic.networks import ActorCritic, RecurrentActorCritic
from esper.simic.features import safe
import logging

logger = logging.getLogger(__name__)


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
        # Recurrence params
        recurrent: bool = False,
        lstm_hidden_dim: int = 128,
        chunk_length: int = 25,  # Must match max_epochs to avoid hidden state issues
    ):
        self.recurrent = recurrent
        self.chunk_length = chunk_length
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
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.device = device

        if recurrent:
            self.network = RecurrentActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_dim=lstm_hidden_dim,
            ).to(device)
            self.recurrent_buffer = RecurrentRolloutBuffer(
                chunk_length=chunk_length,
                lstm_hidden_dim=lstm_hidden_dim,
            )
        else:
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
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float, float, tuple | None]:
        """Get action, optionally with hidden state for recurrent policy.

        Args:
            state: Observation tensor
            action_mask: Binary mask of valid actions (1=valid, 0=invalid)
            hidden: LSTM hidden state (h, c) for recurrent policy, None for MLP
            deterministic: If True, return argmax instead of sampling

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
            hidden: Updated hidden state (recurrent) or None (non-recurrent)
        """
        if self.recurrent:
            return self.network.get_action(state, action_mask, hidden, deterministic)
        else:
            action, log_prob, value, _ = self.network.get_action(state, action_mask, deterministic)
            return action, log_prob, value, None

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

    def store_recurrent_transition(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
        env_id: int,
    ) -> None:
        """Store transition for recurrent policy (no hidden stored per-step)."""
        self.recurrent_buffer.add(
            state=state, action=action, log_prob=log_prob, value=value,
            reward=reward, done=done, action_mask=action_mask, env_id=env_id,
        )

    def update(
        self,
        last_value: float = 0.0,
        clear_buffer: bool = True,
        telemetry_config: "TelemetryConfig | None" = None,
    ) -> dict:
        """Perform PPO update.

        Args:
            last_value: Value estimate for bootstrapping (0.0 for terminal states)
            clear_buffer: Whether to clear the rollout buffer after update.
                Set to False if calling multiple times on the same data
                (e.g., for higher sample efficiency via ppo_updates_per_batch).
            telemetry_config: Optional telemetry configuration for diagnostic collection.
                If None, creates default TelemetryConfig(level=TelemetryLevel.NORMAL).

        Returns:
            Dictionary of training metrics.
        """
        from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel

        if telemetry_config is None:
            telemetry_config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        if len(self.buffer) == 0:
            return {}

        # Compute returns and advantages directly on device (avoids CPU->GPU transfer)
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda, device=self.device
        )

        # Compute explained variance for value function diagnostics
        # Uses values from buffer before any updates
        values_tensor = torch.tensor(
            [t.value for t in self.buffer.steps],
            device=self.device,
        )
        var_returns = returns.var()
        if var_returns > 1e-8:
            explained_variance = 1.0 - (returns - values_tensor).var() / var_returns
            explained_variance = explained_variance.item()
        else:
            explained_variance = 0.0

        metrics = defaultdict(list)
        metrics['explained_variance'] = [explained_variance]  # Single value, not per-batch
        early_stopped = False

        # === AUTO-ESCALATION: Check for anomalies and escalate if needed ===
        from esper.simic.anomaly_detector import AnomalyDetector

        anomaly_detector = AnomalyDetector()
        anomaly_detected = False

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

                # Track ratio statistics for telemetry
                metrics['ratio_mean'].append(ratio.mean().item())
                metrics['ratio_std'].append(ratio.std().item())
                metrics['ratio_max'].append(ratio.max().item())
                metrics['ratio_min'].append(ratio.min().item())

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

                # === DEBUG TELEMETRY: Collect expensive diagnostics if enabled ===
                if telemetry_config.should_collect("debug"):
                    from esper.simic.debug_telemetry import (
                        collect_per_layer_gradients,
                        check_numerical_stability,
                    )
                    # Collect once per update (first batch only) to limit overhead
                    if 'debug_gradient_stats' not in metrics:
                        layer_stats = collect_per_layer_gradients(self.network)
                        metrics['debug_gradient_stats'] = [
                            [s.to_dict() for s in layer_stats]
                        ]
                        stability = check_numerical_stability(self.network, loss)
                        metrics['debug_numerical_stability'] = [stability.to_dict()]

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

        # === AUTO-ESCALATION: Analyze collected metrics for anomalies ===
        if metrics['ratio_max']:  # Only if we have ratio data
            max_ratio = max(metrics['ratio_max'])
            min_ratio = min(metrics['ratio_min'])

            anomaly_report = anomaly_detector.check_all(
                ratio_max=max_ratio,
                ratio_min=min_ratio,
                explained_variance=explained_variance,
                has_nan=False,  # Would need to track during update
                has_inf=False,
            )

            if anomaly_report.has_anomaly:
                anomaly_detected = True
                if telemetry_config.auto_escalate_on_anomaly:
                    telemetry_config.escalate_temporarily()

        # Tick escalation countdown
        telemetry_config.tick_escalation()

        if clear_buffer:
            self.buffer.clear()

        # Build result dict, handling special debug telemetry
        result = {}
        for k, v in metrics.items():
            if k in ('debug_gradient_stats', 'debug_numerical_stability'):
                # Debug metrics are already in final form (list of dicts)
                result[k] = v[0] if v else None
            else:
                # Regular metrics are lists of scalars that need averaging
                result[k] = sum(v) / len(v)

        result['anomaly_detected'] = 1.0 if anomaly_detected else 0.0
        if early_stopped:
            result['early_stopped'] = 1.0
        return result

    def update_recurrent(self, n_epochs: int | None = None, chunk_batch_size: int = 8) -> dict:
        """PPO update for recurrent policy using batched sequences.

        Key design decisions (from reviewer feedback):
        1. GAE computed once, distributed to chunks
        2. Batched chunk processing for GPU efficiency
        3. SINGLE EPOCH (n_epochs=1) default to avoid hidden state drift between epochs
           - After gradient updates, policy changes, so recomputed log_probs differ
           - With multiple epochs, ratio = new_log_prob / old_log_prob becomes biased
           - Single epoch is standard practice for recurrent PPO (OpenAI Five, CleanRL)
        4. value_coef from agent, not hardcoded
        5. Value clipping for training stability (parity with feedforward)
        6. Gradient clipping at max_grad_norm
        7. train_steps incremented for entropy annealing

        Args:
            n_epochs: Number of PPO epochs. Default 1 for recurrent (safest).
                      Higher values (2-4) work due to PPO clipping but emit warning.
            chunk_batch_size: Number of chunks per batch for GPU efficiency.
        """
        # Default to 1 epoch for recurrent (safest)
        if n_epochs is None:
            n_epochs = 1

        # Log info if using multiple epochs (this is fine - CleanRL uses n_epochs=4)
        if n_epochs > 1:
            logger.info(
                f"Using n_epochs={n_epochs} with recurrent policy. This is supported "
                f"(CleanRL uses n_epochs=4). PPO clipping handles the stale log_prob bias."
            )

        # Compute GAE for all episodes
        self.recurrent_buffer.compute_gae(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0  # For diagnostics
        n_updates = 0

        for epoch in range(n_epochs):
            # Process chunks in batches
            for batch in self.recurrent_buffer.get_batched_chunks(
                device=self.device,
                batch_size=chunk_batch_size,
            ):
                batch_size = batch['states'].size(0)
                seq_len = batch['states'].size(1)

                # Get initial hidden [num_layers, batch, hidden]
                initial_hidden = (
                    batch['initial_hidden_h'],
                    batch['initial_hidden_c'],
                )

                # Forward through sequences
                log_probs, values, entropy, _ = self.network.evaluate_actions(
                    batch['states'],
                    batch['actions'],
                    batch['action_masks'],
                    hidden=initial_hidden,
                )

                # Get valid mask and flatten
                valid = batch['valid_mask']  # [batch, seq]

                # Extract valid timesteps
                log_probs_valid = log_probs[valid]
                values_valid = values[valid]
                entropy_valid = entropy[valid]
                old_log_probs_valid = batch['old_log_probs'][valid]
                old_values_valid = batch['old_values'][valid]
                returns_valid = batch['returns'][valid]
                advantages_valid = batch['advantages'][valid]

                # Normalize advantages
                if len(advantages_valid) > 1:
                    advantages_valid = (advantages_valid - advantages_valid.mean()) / (advantages_valid.std() + 1e-8)

                # PPO clipped objective
                ratio = torch.exp(log_probs_valid - old_log_probs_valid)
                surr1 = ratio * advantages_valid
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_valid
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (parity with feedforward PPO)
                values_clipped = old_values_valid + torch.clamp(
                    values_valid - old_values_valid,
                    -self.clip_ratio,
                    self.clip_ratio,
                )
                value_loss_unclipped = (values_valid - returns_valid) ** 2
                value_loss_clipped = (values_clipped - returns_valid) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy_valid.mean()

                # Total loss (using agent's value_coef, not hardcoded)
                loss = policy_loss + self.value_coef * value_loss + self.get_entropy_coef() * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Increment train_steps for entropy annealing (CRITICAL for get_entropy_coef)
                self.train_steps += 1

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_valid.mean().item()
                total_approx_kl += (old_log_probs_valid - log_probs_valid).mean().item()
                n_updates += 1

        self.recurrent_buffer.clear()

        # Compute final metrics
        avg_approx_kl = total_approx_kl / max(n_updates, 1)

        # Warn if KL divergence is high (indicates policy changing too fast)
        if avg_approx_kl > 0.03:
            logger.warning(
                f"High KL divergence ({avg_approx_kl:.4f}) during recurrent update. "
                f"Consider reducing lr or n_epochs to stabilize training."
            )

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'approx_kl': avg_approx_kl,
        }

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
