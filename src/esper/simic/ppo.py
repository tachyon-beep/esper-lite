"""Simic PPO Module - PPO Agent for Seed Lifecycle Control

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.simic.buffers import RolloutBuffer, RecurrentRolloutBuffer
from esper.simic.networks import ActorCritic, RecurrentActorCritic
from esper.simic.features import safe
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import get_hub
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
# =============================================================================

def signals_to_features(signals, model, use_telemetry: bool = True, max_epochs: int = 200) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo
        model: MorphogeneticModel
        use_telemetry: Whether to include telemetry features
        max_epochs: Maximum epochs for learning phase normalization

    Returns:
        Feature vector (30 dims base, +10 if telemetry = 40 dims total)

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
        obs['seed_counterfactual'] = seed_state.metrics.counterfactual_contribution or 0.0
        obs['host_grad_norm'] = signals.metrics.grad_norm_host
        obs['host_learning_phase'] = signals.metrics.epoch / max(1, max_epochs)
        # Blueprint one-hot encoding (DRL Expert recommendation for categorical data)
        # Convert blueprint_id string to integer index for one-hot encoding
        # Standard blueprint order: conv_light=1, conv_heavy=2, attention=3, depthwise=4, norm=5
        blueprint_map = {'conv_light': 1, 'conv_heavy': 2, 'attention': 3, 'depthwise': 4, 'norm': 5}
        obs['seed_blueprint_id'] = blueprint_map.get(seed_state.blueprint_id, 0)
    else:
        # Use TrainingSignals context when no live model is available (offline/replay)
        obs['seed_stage'] = signals.seed_stage
        obs['seed_epochs_in_stage'] = signals.seed_epochs_in_stage
        obs['seed_alpha'] = signals.seed_alpha
        obs['seed_improvement'] = signals.seed_improvement
        obs['seed_counterfactual'] = signals.seed_counterfactual
        obs['host_grad_norm'] = signals.metrics.grad_norm_host
        obs['host_learning_phase'] = signals.metrics.epoch / max(1, max_epochs)
        obs['seed_blueprint_id'] = 0  # No active seed

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
        # Entropy coef operates on NORMALIZED entropy [0, 1] from MaskedCategorical.
        # See MaskedCategorical.entropy() docstring for normalization details.
        # 0.05 normalized ≈ 0.098 raw nats with 7 actions (log(7) ≈ 1.95)
        entropy_coef: float = 0.05,
        entropy_coef_start: float | None = None,
        entropy_coef_end: float | None = None,
        entropy_coef_min: float = 0.01,  # Unified minimum for exploration floor
        adaptive_entropy_floor: bool = False,  # Scale floor with valid action count
        entropy_anneal_steps: int = 0,
        value_coef: float = 0.5,
        clip_value: bool = True,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        recurrent_n_epochs: int | None = None,  # Default 1 for recurrent (hidden state safety)
        batch_size: int = 64,
        target_kl: float | None = 0.015,
        weight_decay: float = 0.0,  # Applied to critic only (RL best practice)
        device: str = "cuda:0",
        # Recurrence params
        recurrent: bool = False,
        lstm_hidden_dim: int = 128,
        chunk_length: int = 25,  # Must match max_epochs to avoid hidden state issues
        # Compilation
        compile_network: bool = True,  # Use torch.compile() for 10-30% speedup
    ):
        self.recurrent = recurrent
        self.chunk_length = chunk_length
        self.gamma = gamma
        # Separate epoch defaults: feedforward uses n_epochs, recurrent defaults to 1
        # Recurrent PPO with multiple epochs causes hidden state staleness (policy drift)
        # See update_recurrent() docstring for detailed explanation
        self.recurrent_n_epochs = recurrent_n_epochs if recurrent_n_epochs is not None else 1
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.entropy_coef_start = entropy_coef_start if entropy_coef_start is not None else entropy_coef
        self.entropy_coef_end = entropy_coef_end if entropy_coef_end is not None else entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.adaptive_entropy_floor = adaptive_entropy_floor
        self.entropy_anneal_steps = entropy_anneal_steps
        self.value_coef = value_coef
        self.clip_value = clip_value
        self.max_grad_norm = max_grad_norm
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.weight_decay = weight_decay
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

        # [PyTorch 2.9] Compile network for 10-30% speedup on forward/backward
        # mode="default" is safest for networks with MaskedCategorical
        # MaskedCategorical._validate_action_mask has @torch.compiler.disable
        if compile_network:
            self.network = torch.compile(self.network, mode="default")

        # [PyTorch 2.9] Use fused=True for CUDA, foreach=True for CPU
        use_cuda = device.startswith("cuda")
        optimizer_kwargs = {'lr': lr, 'eps': 1e-5}
        if use_cuda:
            optimizer_kwargs['fused'] = True
        else:
            optimizer_kwargs['foreach'] = True  # Multi-tensor optimization for CPU

        if weight_decay > 0:
            # [DRL Best Practice] Apply weight decay ONLY to critic, not actor or shared
            # Weight decay on actor biases toward determinism (smaller weights =
            # smaller logits = sharper softmax), which kills exploration.
            # Shared layers feed into actor, so they must also have wd=0.
            # Reference: SAC, TD3 implementations apply WD only to critic.
            actor_params = list(self.network.actor.parameters())
            critic_params = list(self.network.critic.parameters())

            if self.recurrent:
                # RecurrentActorCritic: encoder + lstm feed into actor, must be wd=0
                shared_params = (list(self.network.encoder.parameters()) +
                                list(self.network.lstm.parameters()))
            else:
                # ActorCritic: shared feeds into actor
                shared_params = list(self.network.shared.parameters())

            self.optimizer = torch.optim.AdamW([
                {'params': actor_params, 'weight_decay': 0.0, 'name': 'actor'},
                {'params': shared_params, 'weight_decay': 0.0, 'name': 'shared'},  # Must be 0!
                {'params': critic_params, 'weight_decay': weight_decay, 'name': 'critic'},
            ], **optimizer_kwargs)
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), **optimizer_kwargs
            )
        self.buffer = RolloutBuffer()
        self.train_steps = 0

    @property
    def _base_network(self):
        """Get the original (uncompiled) network module.

        torch.compile() wraps the network in OptimizedModule. This property
        provides consistent access to the underlying network for save/load
        and architecture introspection.
        """
        # hasattr AUTHORIZED by John on 2025-12-10 21:30:00 UTC
        # Justification: torch.compile() wraps modules in OptimizedModule which has _orig_mod
        if hasattr(self.network, '_orig_mod'):
            return self.network._orig_mod
        return self.network

    def get_entropy_coef(self, action_mask: torch.Tensor | None = None) -> float:
        """Get current entropy coefficient with optional adaptive floor.

        Args:
            action_mask: Optional action mask for adaptive floor computation

        Returns:
            Entropy coefficient (decayed if enabled, floored if adaptive)
        """
        if self.entropy_anneal_steps == 0:
            # No annealing - use fixed coefficient with adaptive floor
            floor = self.get_entropy_floor(action_mask)
            return max(self.entropy_coef, floor)

        # With annealing - interpolate and apply adaptive floor
        progress = min(1.0, self.train_steps / self.entropy_anneal_steps)
        annealed = self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)

        # Get floor (adaptive if enabled, otherwise base floor)
        floor = self.get_entropy_floor(action_mask)

        return max(annealed, floor)

    def get_entropy_floor(self, action_mask: torch.Tensor | None = None) -> float:
        """Get entropy floor, optionally scaled by valid action count.

        When adaptive_entropy_floor=True, uses information-theoretic scaling:
        scale_factor = log(num_total) / log(num_valid)

        This maintains the same "relative exploration" level - if we want
        10% of max entropy with 7 actions, we want 10% of max entropy with
        2 actions, but max_entropy(2) = log(2) < max_entropy(7) = log(7).

        Args:
            action_mask: Binary mask of valid actions [action_dim] or None

        Returns:
            Entropy coefficient floor (minimum value)
        """
        if not self.adaptive_entropy_floor or action_mask is None:
            return self.entropy_coef_min

        # Count valid actions
        num_valid = int(action_mask.sum().item())
        num_total = action_mask.shape[-1]

        if num_valid >= num_total or num_valid <= 1:
            return self.entropy_coef_min

        # Information-theoretic scaling: ratio of maximum entropies
        # max_entropy_full = log(num_total), max_entropy_valid = log(num_valid)
        max_entropy_full = math.log(num_total)
        max_entropy_valid = math.log(num_valid)

        # Scale to maintain same fraction of max entropy
        scale_factor = max_entropy_full / max_entropy_valid

        # Cap at 3x to avoid extreme values
        scale_factor = min(scale_factor, 3.0)

        return self.entropy_coef_min * scale_factor

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
        truncated: bool = False,
        bootstrap_value: float = 0.0,
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
            truncated: Whether episode ended due to time limit
            bootstrap_value: Value to bootstrap from if truncated
        """
        self.buffer.add(state, action, log_prob, value, reward, done, action_mask, truncated, bootstrap_value)

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
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Store transition for recurrent policy (no hidden stored per-step)."""
        self.recurrent_buffer.add(
            state=state, action=action, log_prob=log_prob, value=value,
            reward=reward, done=done, action_mask=action_mask, env_id=env_id,
            truncated=truncated, bootstrap_value=bootstrap_value,
        )

    def update(
        self,
        last_value: float = 0.0,
        clear_buffer: bool = True,
        telemetry_config: "TelemetryConfig | None" = None,
        current_episode: int = 0,
        total_episodes: int = 0,
    ) -> dict:
        """Perform PPO update.

        Args:
            last_value: Value estimate for bootstrapping (0.0 for terminal states)
            clear_buffer: Whether to clear the rollout buffer after update.
                Set to False if calling multiple times on the same data
                (e.g., for higher sample efficiency via ppo_updates_per_batch).
            telemetry_config: Optional telemetry configuration for diagnostic collection.
                If None, creates default TelemetryConfig(level=TelemetryLevel.NORMAL).
            current_episode: Current episode number for phase-dependent anomaly thresholds.
            total_episodes: Total configured episodes for phase detection.

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
        flags = {}  # Non-list metrics (boolean flags)
        early_stopped = False

        # Normalize advantages once over full buffer (before batching)
        # This prevents per-batch normalization variance in gradient updates
        # Store pre-normalization stats for stability diagnostics (BEFORE normalization)
        advantages_prenorm_mean = advantages.mean()
        advantages_prenorm_std = advantages.std()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === Value Function Diagnostics (DRL Expert recommendations) ===
        # Compute all stats as a single tensor for batched extraction (fewer CPU syncs)
        # [PyTorch Expert] Cast to float32 for stable statistics under mixed precision
        with torch.no_grad():
            v = values_tensor.float() if values_tensor.dtype != torch.float32 else values_tensor
            r = returns.float() if returns.dtype != torch.float32 else returns

            value_stats = torch.stack([
                v.mean(),
                v.std(),
                r.mean(),
                r.std(),
                r.min(),
                r.max(),
                F.mse_loss(v, r),  # Critic error before update
                advantages_prenorm_mean,  # Pre-normalization (critical for stability debugging)
                advantages_prenorm_std,   # If very small, normalization amplifies noise
            ])

        # Single CPU sync to extract all values
        stats_list = value_stats.tolist()

        metrics['value_pred_mean'] = [stats_list[0]]
        metrics['value_pred_std'] = [stats_list[1]]
        metrics['return_mean'] = [stats_list[2]]
        metrics['return_std'] = [stats_list[3]]
        metrics['return_min'] = [stats_list[4]]
        metrics['return_max'] = [stats_list[5]]
        metrics['value_mse_before'] = [stats_list[6]]
        metrics['advantage_mean_prenorm'] = [stats_list[7]]
        metrics['advantage_std_prenorm'] = [stats_list[8]]

        # [DRL Expert] Warn when advantage std is very low - normalization amplifies noise
        if stats_list[8] < 0.1:
            logger.warning(
                f"Very low advantage std ({stats_list[8]:.4f}) before normalization. "
                f"Normalization may amplify noise. Consider reducing gamma or checking reward scale."
            )

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
                batch_advantages = advantages[batch_idx]  # Already on device (pre-normalized)

                log_probs, values, entropy = self.network.evaluate_actions(states, actions, action_masks)

                # PPO-Clip loss
                ratio = torch.exp(log_probs - old_log_probs)

                # Check ratio for numerical issues - single fused kernel
                # isfinite() combines isnan + isinf into one kernel, all() is single reduction
                if not torch.isfinite(ratio).all():
                    flags['ratio_has_numerical_issue'] = True
                    # Debug breakdown only when telemetry requests it (avoids extra syncs)
                    if telemetry_config.should_collect("debug"):
                        flags['ratio_has_nan'] = torch.isnan(ratio).any().item()
                        flags['ratio_has_inf'] = torch.isinf(ratio).any().item()

                # Track ratio statistics for telemetry (single sync for all 4 stats)
                ratio_stats = torch.stack([ratio.mean(), ratio.std(), ratio.max(), ratio.min()])
                r_mean, r_std, r_max, r_min = ratio_stats.tolist()
                metrics['ratio_mean'].append(r_mean)
                metrics['ratio_std'].append(r_std)
                metrics['ratio_max'].append(r_max)
                metrics['ratio_min'].append(r_min)

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

                # Get entropy coefficient with adaptive floor
                # Use batch's action mask for floor computation
                # For batched masks, use the most restrictive (min valid actions)
                representative_mask = action_masks[0] if action_masks is not None else None
                entropy_coef = self.get_entropy_coef(representative_mask)

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
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
                batch_kl_tensor = (old_log_probs - log_probs).mean()
                clip_frac_tensor = ((ratio - 1).abs() > self.clip_ratio).float().mean()

                # Single sync for all 5 metrics
                batch_metrics = torch.stack([
                    policy_loss, value_loss, -entropy_loss, batch_kl_tensor, clip_frac_tensor
                ])
                pl, vl, ent, batch_kl, cf = batch_metrics.tolist()
                metrics['policy_loss'].append(pl)
                metrics['value_loss'].append(vl)
                metrics['entropy'].append(ent)
                metrics['approx_kl'].append(batch_kl)
                metrics['clip_fraction'].append(cf)

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

            # Check for NaN/Inf in all loss values from the mini-batches
            all_losses = metrics['policy_loss'] + metrics['value_loss']
            loss_has_issue = any(math.isnan(v) or math.isinf(v) for v in all_losses)
            ratio_has_issue = flags.get('ratio_has_numerical_issue', False)
            has_numerical_issue = loss_has_issue or ratio_has_issue

            # Use debug breakdown values when available for accurate anomaly reporting
            anomaly_report = anomaly_detector.check_all(
                ratio_max=max_ratio,
                ratio_min=min_ratio,
                explained_variance=explained_variance,
                has_nan=flags.get('ratio_has_nan', has_numerical_issue),
                has_inf=flags.get('ratio_has_inf', has_numerical_issue),
                current_episode=current_episode,
                total_episodes=total_episodes,
            )

            if anomaly_report.has_anomaly:
                anomaly_detected = True
                if telemetry_config.auto_escalate_on_anomaly:
                    telemetry_config.escalate_temporarily()

                # Emit specific anomaly events
                hub = get_hub()
                for anomaly_type in anomaly_report.anomaly_types:
                    if anomaly_type == "ratio_explosion":
                        event_type = TelemetryEventType.RATIO_EXPLOSION_DETECTED
                    elif anomaly_type == "ratio_collapse":
                        event_type = TelemetryEventType.RATIO_COLLAPSE_DETECTED
                    elif anomaly_type == "value_collapse":
                        event_type = TelemetryEventType.VALUE_COLLAPSE_DETECTED
                    elif anomaly_type == "numerical_instability":
                        event_type = TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED
                    else:
                        event_type = TelemetryEventType.GRADIENT_ANOMALY

                    hub.emit(TelemetryEvent(
                        event_type=event_type,
                        data={
                            "anomaly_type": anomaly_type,
                            "detail": anomaly_report.details.get(anomaly_type, ""),
                            "ratio_max": max_ratio,
                            "ratio_min": min_ratio,
                            "explained_variance": explained_variance,
                            "train_steps": self.train_steps,
                            "approx_kl": sum(metrics['approx_kl']) / len(metrics['approx_kl']) if metrics['approx_kl'] else 0.0,
                            "clip_fraction": sum(metrics['clip_fraction']) / len(metrics['clip_fraction']) if metrics['clip_fraction'] else 0.0,
                            "entropy": sum(metrics['entropy']) / len(metrics['entropy']) if metrics['entropy'] else 0.0,
                        },
                        severity="warning",
                    ))

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

        # Merge in boolean flags from separate dict
        result.update(flags)

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
            n_epochs: Number of PPO epochs. Default uses self.recurrent_n_epochs (1).
                      Higher values (2-4) work due to PPO clipping but emit warning.
            chunk_batch_size: Number of chunks per batch for GPU efficiency.
        """
        # Use instance default (recurrent_n_epochs=1) when not specified
        if n_epochs is None:
            n_epochs = self.recurrent_n_epochs

        # [DRL Best Practice] Two-tier warnings for recurrent PPO n_epochs
        # After gradient updates, policy changes, so recomputed log_probs differ
        # from stored log_probs. With multiple epochs, this staleness compounds.

        WARN_THRESHOLD = 2   # Warn when > 2 (early warning)
        MAX_RECURRENT_EPOCHS = 4  # Hard cap

        if n_epochs > MAX_RECURRENT_EPOCHS:
            warnings.warn(
                f"n_epochs={n_epochs} is too high for recurrent PPO and has been capped "
                f"to {MAX_RECURRENT_EPOCHS}. Values > {MAX_RECURRENT_EPOCHS} cause severe "
                f"policy drift due to hidden state staleness.",
                RuntimeWarning,
            )
            n_epochs = MAX_RECURRENT_EPOCHS
        elif n_epochs > WARN_THRESHOLD:
            warnings.warn(
                f"n_epochs={n_epochs} is elevated for recurrent PPO. "
                f"Values > {WARN_THRESHOLD} risk policy drift due to hidden state staleness. "
                f"Consider n_epochs=1-2 for maximum stability.",
                RuntimeWarning,
            )

        # Compute GAE for all episodes
        self.recurrent_buffer.compute_gae(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Normalize advantages once over full buffer (before batching)
        # This prevents per-batch normalization variance in gradient updates
        self.recurrent_buffer.normalize_advantages()

        # Compute explained variance BEFORE any gradient updates
        # This measures how well the pre-update value function predicted returns
        all_returns = []
        all_old_values = []
        for chunk in self.recurrent_buffer.get_chunks(device=self.device):
            valid = chunk['valid_mask'].squeeze(0)  # [seq]
            all_returns.append(chunk['returns'].squeeze(0)[valid])
            all_old_values.append(chunk['old_values'].squeeze(0)[valid])

        if all_returns:
            returns_tensor = torch.cat(all_returns)
            values_tensor = torch.cat(all_old_values)
            var_returns = returns_tensor.var()
            if var_returns > 1e-8:
                explained_variance = 1.0 - (returns_tensor - values_tensor).var() / var_returns
                explained_variance = float(torch.clamp(explained_variance, -1.0, 1.0).item())
            else:
                explained_variance = 0.0
        else:
            explained_variance = 0.0

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0  # For diagnostics
        n_updates = 0
        all_ratios = []  # Track ratio statistics for anomaly detection

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
                advantages_valid = batch['advantages'][valid]  # Already normalized globally

                # PPO clipped objective
                ratio = torch.exp(log_probs_valid - old_log_probs_valid)

                # Track ratio for anomaly detection
                all_ratios.append(ratio.detach())

                surr1 = ratio * advantages_valid
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_valid
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (with optional clipping, matching feedforward PPO)
                if self.clip_value:
                    values_clipped = old_values_valid + torch.clamp(
                        values_valid - old_values_valid,
                        -self.clip_ratio,
                        self.clip_ratio,
                    )
                    value_loss_unclipped = (values_valid - returns_valid) ** 2
                    value_loss_clipped = (values_clipped - returns_valid) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values_valid, returns_valid)

                # Entropy bonus
                entropy_loss = -entropy_valid.mean()

                # Get entropy coefficient with adaptive floor
                # Use first action mask from batch as representative
                representative_mask = batch['action_masks'][0, 0] if batch['action_masks'] is not None else None
                entropy_coef = self.get_entropy_coef(representative_mask)

                # Total loss (using agent's value_coef, not hardcoded)
                loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics (single sync for all 4)
                approx_kl_tensor = (old_log_probs_valid - log_probs_valid).mean()
                batch_metrics = torch.stack([
                    policy_loss, value_loss, entropy_valid.mean(), approx_kl_tensor
                ])
                pl, vl, ent, kl = batch_metrics.tolist()
                total_policy_loss += pl
                total_value_loss += vl
                total_entropy += ent
                total_approx_kl += kl
                n_updates += 1

        # Increment train_steps for entropy annealing (CRITICAL for get_entropy_coef)
        # Must be AFTER epoch/batch loops to match update() behavior (was at line 565, causing ~8x too fast annealing)
        self.train_steps += 1

        self.recurrent_buffer.clear()

        # Compute final metrics
        avg_approx_kl = total_approx_kl / max(n_updates, 1)

        # Warn if KL divergence is high (indicates policy changing too fast)
        if avg_approx_kl > 0.03:
            logger.warning(
                f"High KL divergence ({avg_approx_kl:.4f}) during recurrent update. "
                f"Consider reducing lr or n_epochs to stabilize training."
            )

        # Compute ratio statistics for anomaly detection
        metrics = {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'approx_kl': avg_approx_kl,
            'explained_variance': explained_variance,
        }

        if all_ratios:
            stacked = torch.cat(all_ratios)
            # Single sync for all 3 ratio stats
            ratio_stats = torch.stack([stacked.max(), stacked.min(), stacked.std()])
            r_max, r_min, r_std = ratio_stats.tolist()
            metrics['ratio_max'] = [r_max]
            metrics['ratio_min'] = [r_min]
            metrics['ratio_std'] = [r_std]

            # Check for NaN/Inf in ratios - single fused check
            # Store in separate dict to avoid type conflict with list-based metrics
            if not torch.isfinite(stacked).all():
                metrics['ratio_has_numerical_issue'] = True

        return metrics

    def save(self, path: str | Path, metadata: dict = None) -> None:
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use _base_network to get uncompiled module for consistent state dict keys
        base_net = self._base_network
        save_dict = {
            'network_state_dict': base_net.state_dict(),
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
                'adaptive_entropy_floor': self.adaptive_entropy_floor,
                'entropy_anneal_steps': self.entropy_anneal_steps,
                'value_coef': self.value_coef,
                'clip_value': self.clip_value,
                'target_kl': self.target_kl,
                'recurrent': self.recurrent,
                'recurrent_n_epochs': self.recurrent_n_epochs,  # Epoch default for recurrent PPO
                'lstm_hidden_dim': self.lstm_hidden_dim,
                'chunk_length': self.chunk_length,
            },
            # Architecture info for load-time reconstruction
            'architecture': {
                'recurrent': self.recurrent,
                'state_dim': base_net.state_dim if self.recurrent else None,
                'action_dim': base_net.action_dim,
            }
        }
        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda:0") -> "PPOAgent":
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        state_dict = checkpoint['network_state_dict']
        arch = checkpoint.get('architecture', {})
        is_recurrent = arch.get('recurrent', False)

        # Infer dimensions from state_dict keys
        if is_recurrent:
            # RecurrentActorCritic: encoder.0.weight has shape [hidden_dim, state_dim]
            state_dim = state_dict['encoder.0.weight'].shape[1]
            # actor.2.weight has shape [action_dim, lstm_hidden_dim // 2]
            action_dim = state_dict['actor.2.weight'].shape[0]
        else:
            # ActorCritic: shared.0.weight has shape [hidden_dim, state_dim]
            state_dim = state_dict['shared.0.weight'].shape[1]
            action_dim = state_dict['actor.2.weight'].shape[0]

        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **checkpoint.get('config', {})
        )

        # Load into base network (handles compiled modules)
        agent._base_network.load_state_dict(state_dict)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train_steps = checkpoint.get('train_steps', 0)

        return agent


__all__ = [
    "PPOAgent",
    "signals_to_features",
]
