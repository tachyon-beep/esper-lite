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

from esper.simic.networks import ActorCritic
from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer
from esper.simic.tamiyo_network import FactoredRecurrentActorCritic
from esper.simic.advantages import compute_per_head_advantages
from esper.simic.features import safe
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import get_hub
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
# =============================================================================

def signals_to_features(
    signals,
    model,
    use_telemetry: bool = True,
    max_epochs: int = 200,
    slots: list[str] | None = None,
    total_seeds: int = 0,
    max_seeds: int = 0,
) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo
        model: MorphogeneticModel
        use_telemetry: Whether to include telemetry features
        max_epochs: Maximum epochs for learning phase normalization
        slots: List of slot names to extract features from (uses first slot)
        total_seeds: Current total seeds across all slots (for utilization calc)
        max_seeds: Maximum allowed seeds (for utilization calc)

    Returns:
        Feature vector (50 dims base, +10 if telemetry = 60 dims total)

    Note:
        TrainingSignals.active_seeds contains seed IDs (strings), not SeedState
        objects, so seed-specific features and telemetry are always zero-padded.
    """
    if not slots:
        raise ValueError("signals_to_features: slots parameter is required and cannot be empty")

    target_slot = slots[0]
    from esper.simic.features import obs_to_multislot_features

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
        'total_params': model.total_params if model else 0,
    }

    # Build per-slot state dict
    slot_states = {}
    for slot_id in ['early', 'mid', 'late']:
        if model and slot_id in model.seed_slots:
            slot = model.seed_slots[slot_id]
            if slot.is_active and slot.state:
                slot_states[slot_id] = {
                    'is_active': 1.0,
                    'stage': slot.state.stage.value,
                    'alpha': slot.state.alpha,
                    'improvement': slot.state.metrics.improvement_since_stage_start,
                    'blueprint_id': slot.state.blueprint_id,
                }
            else:
                slot_states[slot_id] = {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None}
        else:
            slot_states[slot_id] = {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None}

    obs['slots'] = slot_states

    features = obs_to_multislot_features(obs, total_seeds=total_seeds, max_seeds=max_seeds)

    if use_telemetry:
        # Use real telemetry from model.seed_slots[target_slot].state when available
        from esper.leyline import SeedTelemetry
        if model and model.has_active_seed:
            seed_state = model.seed_slots[target_slot].state
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
        # Factored action space
        factored: bool = False,  # Use FactoredActorCritic with multi-head actions
        num_envs: int = 4,  # For TamiyoRolloutBuffer
        max_steps_per_env: int = 25,  # For TamiyoRolloutBuffer (matches max_epochs)
        # Compilation
        compile_network: bool = True,  # Use torch.compile() for 10-30% speedup
    ):
        self.recurrent = recurrent
        self.factored = factored
        self.num_envs = num_envs
        self.max_steps_per_env = max_steps_per_env
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

        # Unified factored + recurrent mode
        self.network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        ).to(device)
        self.buffer = TamiyoRolloutBuffer(
            num_envs=num_envs,
            max_steps_per_env=max_steps_per_env,
            state_dim=state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=torch.device(device),
        )

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
            # FactoredRecurrentActorCritic: slot_head, blueprint_head, blend_head, op_head are actors
            actor_params = (
                list(self._base_network.slot_head.parameters()) +
                list(self._base_network.blueprint_head.parameters()) +
                list(self._base_network.blend_head.parameters()) +
                list(self._base_network.op_head.parameters())
            )
            critic_params = list(self._base_network.value_head.parameters())
            shared_params = (
                list(self._base_network.feature_net.parameters()) +
                list(self._base_network.lstm.parameters()) +
                list(self._base_network.lstm_ln.parameters())
            )

            self.optimizer = torch.optim.AdamW([
                {'params': actor_params, 'weight_decay': 0.0, 'name': 'actor'},
                {'params': shared_params, 'weight_decay': 0.0, 'name': 'shared'},  # Must be 0!
                {'params': critic_params, 'weight_decay': weight_decay, 'name': 'critic'},
            ], **optimizer_kwargs)
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), **optimizer_kwargs
            )
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
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Get action from policy.

        Args:
            state: Observation tensor
            action_mask: Binary mask of valid actions (1=valid, 0=invalid)
            deterministic: If True, return argmax instead of sampling

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
        """
        action, log_prob, value, _ = self.network.get_action(state, action_mask, deterministic)
        return action, log_prob, value

    def update_tamiyo(
        self,
        clear_buffer: bool = True,
    ) -> dict:
        """PPO update for Tamiyo (factored + recurrent).

        Uses per-head advantages with causal masking and LSTM hidden states.

        Args:
            clear_buffer: Whether to clear buffer after update

        Returns:
            Dict of training metrics
        """
        if len(self.tamiyo_buffer) == 0:
            return {}

        # Compute GAE per-environment (fixes P0 bug)
        self.tamiyo_buffer.compute_advantages_and_returns(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        self.tamiyo_buffer.normalize_advantages()

        # Get batched data
        data = self.tamiyo_buffer.get_batched_sequences(device=self.device)
        valid_mask = data["valid_mask"]

        # Compute explained variance before updates
        valid_values = data["values"][valid_mask]
        valid_returns = data["returns"][valid_mask]
        var_returns = valid_returns.var()
        if var_returns > 1e-8:
            explained_variance = 1.0 - (valid_returns - valid_values).var() / var_returns
            explained_variance = explained_variance.item()
        else:
            explained_variance = 0.0

        metrics = defaultdict(list)
        metrics["explained_variance"] = [explained_variance]
        early_stopped = False

        for epoch_i in range(self.recurrent_n_epochs):
            if early_stopped:
                break

            # Forward pass through network
            actions = {
                "slot": data["slot_actions"],
                "blueprint": data["blueprint_actions"],
                "blend": data["blend_actions"],
                "op": data["op_actions"],
            }

            log_probs, values, entropy, _ = self.network.evaluate_actions(
                data["states"],
                actions,
                slot_mask=data["slot_masks"],
                blueprint_mask=data["blueprint_masks"],
                blend_mask=data["blend_masks"],
                op_mask=data["op_masks"],
                hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
            )

            # Extract valid timesteps
            for key in log_probs:
                log_probs[key] = log_probs[key][valid_mask]
            values = values[valid_mask]
            for key in entropy:
                entropy[key] = entropy[key][valid_mask]

            valid_advantages = data["advantages"][valid_mask]
            valid_returns = data["returns"][valid_mask]

            # Compute per-head advantages with causal masking
            valid_op_actions = data["op_actions"][valid_mask]
            per_head_advantages = compute_per_head_advantages(
                valid_advantages, valid_op_actions
            )

            # Compute per-head ratios
            old_log_probs = {
                "slot": data["slot_log_probs"][valid_mask],
                "blueprint": data["blueprint_log_probs"][valid_mask],
                "blend": data["blend_log_probs"][valid_mask],
                "op": data["op_log_probs"][valid_mask],
            }

            per_head_ratios = {}
            for key in ["slot", "blueprint", "blend", "op"]:
                per_head_ratios[key] = torch.exp(log_probs[key] - old_log_probs[key])

            # Compute policy loss per head and sum
            policy_loss = 0.0
            for key in ["slot", "blueprint", "blend", "op"]:
                ratio = per_head_ratios[key]
                adv = per_head_advantages[key]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                head_loss = -torch.min(surr1, surr2).mean()
                policy_loss = policy_loss + head_loss

            # Value loss
            valid_old_values = data["values"][valid_mask]
            if self.clip_value:
                values_clipped = valid_old_values + torch.clamp(
                    values - valid_old_values, -self.clip_ratio, self.clip_ratio
                )
                value_loss_unclipped = (values - valid_returns) ** 2
                value_loss_clipped = (values_clipped - valid_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = F.mse_loss(values, valid_returns)

            # Entropy loss (sum across heads, each normalized)
            entropy_loss = 0.0
            for key, ent in entropy.items():
                entropy_loss = entropy_loss - ent.mean()

            entropy_coef = self.get_entropy_coef()

            loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            joint_ratio = per_head_ratios["op"]  # Use op ratio as representative
            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["ratio_mean"].append(joint_ratio.mean().item())
            metrics["ratio_max"].append(joint_ratio.max().item())

        self.train_steps += 1

        if clear_buffer:
            self.tamiyo_buffer.reset()

        # Aggregate
        result = {}
        for k, v in metrics.items():
            result[k] = sum(v) / len(v) if v else 0.0

        return result

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
        from esper.simic.debug_telemetry import RatioExplosionDiagnostic

        anomaly_detector = AnomalyDetector()
        anomaly_detected = False
        ratio_diagnostic: RatioExplosionDiagnostic | None = None

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

                # Collect ratio explosion diagnostic when thresholds exceeded
                # Only collect first occurrence to capture the triggering batch
                if ratio_diagnostic is None:
                    if r_max > anomaly_detector.max_ratio_threshold or r_min < anomaly_detector.min_ratio_threshold:
                        ratio_diagnostic = RatioExplosionDiagnostic.from_batch(
                            ratio=ratio,
                            old_log_probs=old_log_probs,
                            new_log_probs=log_probs,
                            actions=actions,
                            max_threshold=anomaly_detector.max_ratio_threshold,
                            min_threshold=anomaly_detector.min_ratio_threshold,
                        )

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

                    event_data = {
                        "anomaly_type": anomaly_type,
                        "detail": anomaly_report.details.get(anomaly_type, ""),
                        "ratio_max": max_ratio,
                        "ratio_min": min_ratio,
                        "explained_variance": explained_variance,
                        "train_steps": self.train_steps,
                        "approx_kl": sum(metrics['approx_kl']) / len(metrics['approx_kl']) if metrics['approx_kl'] else 0.0,
                        "clip_fraction": sum(metrics['clip_fraction']) / len(metrics['clip_fraction']) if metrics['clip_fraction'] else 0.0,
                        "entropy": sum(metrics['entropy']) / len(metrics['entropy']) if metrics['entropy'] else 0.0,
                    }

                    # Include detailed diagnostic for ratio anomalies
                    if anomaly_type in ("ratio_explosion", "ratio_collapse") and ratio_diagnostic is not None:
                        event_data["diagnostic"] = ratio_diagnostic.to_dict()

                    hub.emit(TelemetryEvent(
                        event_type=event_type,
                        data=event_data,
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
                'tamiyo': self.tamiyo,
                'recurrent_n_epochs': self.recurrent_n_epochs,
                'lstm_hidden_dim': self.lstm_hidden_dim,
                'chunk_length': self.chunk_length,
            },
            # Architecture info for load-time reconstruction
            'architecture': {
                'tamiyo': self.tamiyo,
                'state_dim': base_net.state_dim if self.tamiyo else None,
                'action_dim': base_net.action_dim if not self.tamiyo else None,
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
        is_tamiyo = arch.get('tamiyo', False)

        # Infer dimensions from state_dict keys
        if is_tamiyo:
            # FactoredRecurrentActorCritic: feature_net.0.weight has shape [hidden_dim, state_dim]
            state_dim = state_dict['feature_net.0.weight'].shape[1]
            action_dim = 7  # Not used for tamiyo
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
