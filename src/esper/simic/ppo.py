"""Simic PPO Module - PPO Agent for Seed Lifecycle Control

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer
from esper.simic.tamiyo_network import FactoredRecurrentActorCritic
from esper.simic.advantages import compute_per_head_advantages
from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_N_ENVS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_CLIP_RATIO,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_VALUE_COEF,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_N_PPO_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
    DEFAULT_VALUE_CLIP,
)
from esper.nissa import get_hub
import logging

if TYPE_CHECKING:
    from esper.leyline import SeedStateReport

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
# =============================================================================

def signals_to_features(
    signals,
    *,
    slot_reports: dict[str, "SeedStateReport"],
    use_telemetry: bool = True,
    max_epochs: int = 200,
    slots: list[str] | None = None,
    total_params: int = 0,
    total_seeds: int = 0,
    max_seeds: int = 0,
) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo
        slot_reports: Slot -> SeedStateReport snapshot for this timestep
        use_telemetry: Whether to include telemetry features
        max_epochs: Maximum epochs for learning phase normalization
        slots: Enabled slot IDs (used to pick telemetry seed deterministically)
        total_params: Total model params (host + active seeds)
        total_seeds: Current total seeds across all slots (for utilization calc)
        max_seeds: Maximum allowed seeds (for utilization calc)

    Returns:
        Feature vector: base (50) + telemetry per slot (3 * 10) = 80 when telemetry enabled.

    Note:
        TrainingSignals.active_seeds contains seed IDs (strings), not SeedState
        objects, so seed-specific features are zero-padded when slot reports
        are missing.
    """
    if not slots:
        raise ValueError("signals_to_features: slots parameter is required and cannot be empty")

    from esper.simic.features import obs_to_multislot_features
    from esper.simic.slots import CANONICAL_SLOTS, ordered_slots

    enabled_slots = ordered_slots(slots)
    enabled_set = set(enabled_slots)

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
        'total_params': total_params,
    }

    # Build per-slot state dict from reports
    slot_states = {}
    for slot_id in ['early', 'mid', 'late']:
        report = slot_reports.get(slot_id)
        if report:
            contribution = report.metrics.counterfactual_contribution
            if contribution is None:
                contribution = report.metrics.improvement_since_stage_start
            slot_states[slot_id] = {
                'is_active': 1.0,
                'stage': report.stage.value,
                'alpha': report.metrics.current_alpha,
                'improvement': contribution,
                'blueprint_id': report.blueprint_id,
            }
        else:
            slot_states[slot_id] = {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None}

    obs['slots'] = slot_states

    features = obs_to_multislot_features(obs, total_seeds=total_seeds, max_seeds=max_seeds)

    if use_telemetry:
        # Deterministic `[early][mid][late]` ordering; zero-pad empty/disabled slots.
        from esper.leyline import SeedTelemetry

        telemetry_features: list[float] = []
        for slot_id in CANONICAL_SLOTS:
            if slot_id not in enabled_set:
                telemetry_features.extend([0.0] * SeedTelemetry.feature_dim())
                continue

            report = slot_reports.get(slot_id)
            if report is None or report.telemetry is None:
                telemetry_features.extend([0.0] * SeedTelemetry.feature_dim())
                continue

            telemetry_features.extend(report.telemetry.to_features())

        features.extend(telemetry_features)

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
        lr: float = DEFAULT_LEARNING_RATE,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        clip_ratio: float = DEFAULT_CLIP_RATIO,
        # Entropy coef operates on NORMALIZED entropy [0, 1] from MaskedCategorical.
        # See MaskedCategorical.entropy() docstring for normalization details.
        # 0.05 normalized ≈ 0.098 raw nats with 7 actions (log(7) ≈ 1.95)
        entropy_coef: float = DEFAULT_ENTROPY_COEF,
        entropy_coef_start: float | None = None,
        entropy_coef_end: float | None = None,
        entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,  # Exploration floor (from leyline)
        adaptive_entropy_floor: bool = False,  # Scale floor with valid action count
        entropy_anneal_steps: int = 0,
        value_coef: float = DEFAULT_VALUE_COEF,
        clip_value: bool = True,
        # Separate clip range for value function (larger than policy clip_ratio)
        # Note: Some research (Engstrom et al., 2020) suggests value clipping often
        # hurts performance. Consider clip_value=False if value learning is slow.
        value_clip: float = DEFAULT_VALUE_CLIP,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        n_epochs: int = DEFAULT_N_PPO_EPOCHS,
        recurrent_n_epochs: int | None = None,  # Default 1 for recurrent (hidden state safety)
        batch_size: int = DEFAULT_BATCH_SIZE,
        target_kl: float | None = 0.015,
        weight_decay: float = 0.0,  # Applied to critic only (RL best practice)
        device: str = "cuda:0",
        # LSTM configuration (unified architecture always uses LSTM)
        lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
        chunk_length: int = DEFAULT_EPISODE_LENGTH,  # Must match max_epochs (from leyline)
        num_envs: int = DEFAULT_N_ENVS,  # For TamiyoRolloutBuffer
        max_steps_per_env: int = DEFAULT_EPISODE_LENGTH,  # For TamiyoRolloutBuffer (from leyline)
        # Compilation
        compile_network: bool = True,  # Use torch.compile() for 10-30% speedup
    ):
        self.num_envs = num_envs
        self.max_steps_per_env = max_steps_per_env
        self.chunk_length = chunk_length
        self.gamma = gamma
        # Recurrent PPO with multiple epochs can cause hidden state staleness (policy drift)
        # Default to 1 epoch for LSTM safety; increase with caution
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
        self.value_clip = value_clip
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

    def update(
        self,
        clear_buffer: bool = True,
    ) -> dict:
        """PPO update (factored + recurrent).

        Uses per-head advantages with causal masking and LSTM hidden states.

        Args:
            clear_buffer: Whether to clear buffer after update

        Returns:
            Dict of training metrics
        """
        if len(self.buffer) == 0:
            return {}

        # Compute GAE per-environment (fixes P0 bug)
        self.buffer.compute_advantages_and_returns(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        self.buffer.normalize_advantages()

        # Get batched data
        data = self.buffer.get_batched_sequences(device=self.device)
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
                # Use separate value_clip (not policy clip_ratio) since value scale differs
                # Value predictions can range from -10 to +50, so clip_ratio=0.2 is too tight
                values_clipped = valid_old_values + torch.clamp(
                    values - valid_old_values, -self.value_clip, self.value_clip
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
            metrics["ratio_min"].append(joint_ratio.min().item())

            # Compute approximate KL divergence using the log-ratio trick:
            # KL(old||new) ≈ E[(ratio - 1) - log(ratio)]
            # This is the "KL3" estimator from Schulman's TRPO/PPO papers
            # Average across all heads for factored policy (PyTorch expert review)
            with torch.no_grad():
                head_kls = []
                for key in ["slot", "blueprint", "blend", "op"]:
                    log_ratio = log_probs[key] - old_log_probs[key]
                    head_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    head_kls.append(head_kl)
                approx_kl = torch.stack(head_kls).mean().item()
                metrics["approx_kl"].append(approx_kl)

                # Clip fraction: how often clipping was active
                clip_fraction = ((joint_ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                metrics["clip_fraction"].append(clip_fraction)

                # Early stopping if KL divergence exceeds threshold
                # 1.5x multiplier is standard (OpenAI baselines, Stable-Baselines3)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    metrics["early_stop_epoch"] = [epoch_i]

        self.train_steps += 1

        if clear_buffer:
            self.buffer.reset()

        # Aggregate
        result = {}
        for k, v in metrics.items():
            result[k] = sum(v) / len(v) if v else 0.0

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
                'value_clip': self.value_clip,
                'target_kl': self.target_kl,
                'recurrent_n_epochs': self.recurrent_n_epochs,
                'lstm_hidden_dim': self.lstm_hidden_dim,
                'chunk_length': self.chunk_length,
                # Buffer dimensions (must match training loop)
                'num_envs': self.num_envs,
                'max_steps_per_env': self.max_steps_per_env,
            },
            # Architecture info for load-time reconstruction
            'architecture': {
                'state_dim': base_net.state_dim,
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

        # Infer state_dim from FactoredRecurrentActorCritic state dict
        # feature_net.0.weight has shape [hidden_dim, state_dim]
        state_dim = state_dict['feature_net.0.weight'].shape[1]

        agent = cls(
            state_dim=state_dim,
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
