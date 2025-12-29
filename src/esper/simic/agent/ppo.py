"""Simic PPO Module - PPO Agent for Seed Lifecycle Control

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rollout_buffer import TamiyoRolloutBuffer
from .advantages import compute_per_head_advantages
from .types import PPOUpdateMetrics
from esper.simic.telemetry import RatioExplosionDiagnostic
from esper.tamiyo.policy.protocol import PolicyBundle
from esper.leyline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_RATIO,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_N_ENVS,
    DEFAULT_N_PPO_EPOCHS,
    DEFAULT_RATIO_COLLAPSE_THRESHOLD,
    DEFAULT_RATIO_EXPLOSION_THRESHOLD,
    DEFAULT_VALUE_CLIP,
    DEFAULT_VALUE_COEF,
    HEAD_NAMES,
    compute_causal_masks,
)
from esper.leyline.slot_config import SlotConfig
import logging

if TYPE_CHECKING:
    from esper.leyline import SeedStateReport

logger = logging.getLogger(__name__)

# Checkpoint format version for forward compatibility
# Increment when checkpoint structure changes in backwards-incompatible ways
CHECKPOINT_VERSION = 1


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
# =============================================================================

def signals_to_features(
    signals: Any,
    *,
    slot_reports: dict[str, "SeedStateReport"],
    use_telemetry: bool = True,
    max_epochs: int = DEFAULT_EPISODE_LENGTH,
    slots: list[str] | None = None,
    total_params: int = 0,
    total_seeds: int = 0,
    max_seeds: int = 0,
    slot_config: "SlotConfig | None" = None,
) -> list[float]:
    """Convert training signals to a flat feature vector for PPO.

    This is a thin wrapper around `tamiyo.policy.features.obs_to_multislot_features`
    plus optional per-slot `SeedTelemetry` features appended in deterministic
    `slot_config` order.
    """
    from esper.leyline.slot_id import validate_slot_ids
    from esper.tamiyo.policy.features import obs_to_multislot_features

    if slot_config is None:
        slot_config = SlotConfig.default()

    if not slots:
        raise ValueError("signals_to_features: slots parameter is required and cannot be empty")

    enabled_slots = validate_slot_ids(list(slots))
    enabled_set = set(enabled_slots)

    # loss_history and accuracy_history are required fields on TrainingSignals
    # (leyline/signals.py lines 79-80) with default empty lists
    loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
    while len(loss_hist) < 5:
        loss_hist.insert(0, 0.0)

    acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
    while len(acc_hist) < 5:
        acc_hist.insert(0, 0.0)

    obs: dict[str, Any] = {
        "epoch": signals.metrics.epoch,
        "global_step": signals.metrics.global_step,
        "train_loss": signals.metrics.train_loss,
        "val_loss": signals.metrics.val_loss,
        "loss_delta": signals.metrics.loss_delta,
        "train_accuracy": signals.metrics.train_accuracy,
        "val_accuracy": signals.metrics.val_accuracy,
        "accuracy_delta": signals.metrics.accuracy_delta,
        "plateau_epochs": signals.metrics.plateau_epochs,
        "best_val_accuracy": signals.metrics.best_val_accuracy,
        "best_val_loss": signals.metrics.best_val_loss,
        "loss_history_5": loss_hist,
        "accuracy_history_5": acc_hist,
        "total_params": total_params,
        "max_epochs": max_epochs,
        "slots": {},
    }

    # Build per-slot state dict from reports.
    # ALL slots in slot_config must be present (Task 4: fail loudly on missing required fields).
    slot_states: dict[str, dict[str, Any]] = {}
    for slot_id in slot_config.slot_ids:
        report = slot_reports.get(slot_id)
        if report is None:
            # Missing report -> inactive slot with default values
            slot_states[slot_id] = {
                "is_active": 0.0,
                "stage": 0,
                "alpha": 0.0,
                "improvement": 0.0,
            }
            continue

        contribution = report.metrics.counterfactual_contribution
        if contribution is None:
            contribution = report.metrics.improvement_since_stage_start

        slot_states[slot_id] = {
            "is_active": 1.0,
            "stage": report.stage.value,
            "alpha": report.metrics.current_alpha,
            "improvement": contribution,
            "blend_tempo_epochs": report.blend_tempo_epochs,
            "alpha_target": report.alpha_target,
            "alpha_mode": report.alpha_mode,
            "alpha_steps_total": report.alpha_steps_total,
            "alpha_steps_done": report.alpha_steps_done,
            "time_to_target": report.time_to_target,
            "alpha_velocity": report.alpha_velocity,
            "alpha_algorithm": report.alpha_algorithm,
            "blueprint_id": report.blueprint_id,
        }

    obs["slots"] = slot_states

    features = obs_to_multislot_features(
        obs,
        total_seeds=total_seeds,
        max_seeds=max_seeds,
        slot_config=slot_config,
    )

    if use_telemetry:
        from esper.leyline import SeedTelemetry

        telemetry_features: list[float] = []
        dim = SeedTelemetry.feature_dim()
        for slot_id in slot_config.slot_ids:
            if slot_id not in enabled_set:
                telemetry_features.extend([0.0] * dim)
                continue

            report = slot_reports.get(slot_id)
            if report is None or report.telemetry is None:
                telemetry_features.extend([0.0] * dim)
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
        policy: PolicyBundle,
        slot_config: "SlotConfig | None" = None,
        lr: float = DEFAULT_LEARNING_RATE,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        clip_ratio: float = DEFAULT_CLIP_RATIO,
        # Entropy coef operates on NORMALIZED entropy [0, 1] from MaskedCategorical.
        # See MaskedCategorical.entropy() docstring for normalization details.
        # 0.05 normalized ≈ 0.08 raw nats with 5 actions (log(5) ≈ 1.61)
        entropy_coef: float = DEFAULT_ENTROPY_COEF,
        entropy_coef_start: float | None = None,
        entropy_coef_end: float | None = None,
        entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,  # Exploration floor (from leyline)
        adaptive_entropy_floor: bool = False,  # Scale floor with valid action count
        entropy_anneal_steps: int = 0,
        # Per-head entropy coefficients for relative weighting.
        # Blueprint/style heads may warrant different entropy weighting since they
        # are only active during GERMINATE actions (less frequent than slot/op).
        entropy_coef_per_head: dict[str, float] | None = None,
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
        chunk_length: int = DEFAULT_EPISODE_LENGTH,  # Must match max_epochs (from leyline)
        num_envs: int = DEFAULT_N_ENVS,  # For TamiyoRolloutBuffer
        max_steps_per_env: int = DEFAULT_EPISODE_LENGTH,  # For TamiyoRolloutBuffer (from leyline)
        compile_mode: str = "off",  # For checkpoint persistence (policy may already be compiled)
    ):
        # Store policy and extract slot_config
        self.policy = policy
        self.compile_mode = compile_mode  # Persisted in checkpoint for resume
        if slot_config is None:
            # Get slot_config from the PolicyBundle, not the inner network
            slot_config = policy.slot_config
        self.slot_config = slot_config

        # Extract state_dim from policy network
        # Handle torch.compile wrapper (_orig_mod)
        # getattr AUTHORIZED by Code Review 2025-12-17
        # Justification: torch.compile wraps modules - must unwrap to access state_dim
        base_net_untyped = getattr(policy.network, '_orig_mod', policy.network)
        # Type assertion: we know this is the actual network module
        from esper.tamiyo.networks import FactoredRecurrentActorCritic
        assert isinstance(base_net_untyped, FactoredRecurrentActorCritic)
        state_dim: int = base_net_untyped.state_dim

        self.num_envs = num_envs
        self.max_steps_per_env = max_steps_per_env
        self.chunk_length = chunk_length
        self.gamma = gamma
        # C4: RECURRENT POLICY VALUE STALENESS WARNING
        # Recurrent PPO with multiple epochs can cause hidden state staleness (policy drift).
        # Default to 1 epoch for LSTM safety; increase with caution.
        #
        # With recurrent_n_epochs > 1 AND clip_value=True:
        # - Rollout values are computed with specific LSTM hidden state trajectories
        # - After epoch 0 update, network weights change
        # - Epoch 1+ forward passes produce different hidden state trajectories
        # - Value clipping compares new values against stale rollout values
        # - This creates incorrect gradient signals, slowing value learning
        #
        # Recommended: Keep recurrent_n_epochs=1 for recurrent policies.
        self.recurrent_n_epochs = recurrent_n_epochs if recurrent_n_epochs is not None else 1
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.entropy_coef_start = entropy_coef_start if entropy_coef_start is not None else entropy_coef
        self.entropy_coef_end = entropy_coef_end if entropy_coef_end is not None else entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.adaptive_entropy_floor = adaptive_entropy_floor
        self.entropy_anneal_steps = entropy_anneal_steps
        # Per-head entropy multipliers (default to uniform)
        self.entropy_coef_per_head = entropy_coef_per_head or {
            "slot": 1.0,
            "blueprint": 1.0,
            "style": 1.0,
            "tempo": 1.0,
            "alpha_target": 1.0,
            "alpha_speed": 1.0,
            "alpha_curve": 1.0,
            "op": 1.0,
        }
        self.value_coef = value_coef
        self.clip_value = clip_value
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.lstm_hidden_dim = policy.hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.weight_decay = weight_decay
        self.device = device

        # C4/H5: Runtime warning for risky recurrent configuration
        if self.recurrent_n_epochs > 1:
            import warnings
            if self.clip_value:
                warnings.warn(
                    f"recurrent_n_epochs={self.recurrent_n_epochs} with clip_value=True: "
                    "Value clipping uses rollout values which have different LSTM hidden state "
                    "trajectories than training forward passes after epoch 0. This causes stale "
                    "value comparisons that may slow learning. Consider recurrent_n_epochs=1 or "
                    "clip_value=False.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"recurrent_n_epochs={self.recurrent_n_epochs} > 1: Hidden states stored from "
                    "rollout collection may diverge from current policy's hidden state evolution "
                    "after epoch 0 weight updates. This can cause gradient estimation errors. "
                    "Consider recurrent_n_epochs=1 for recurrent policies.",
                    UserWarning,
                    stacklevel=2,
                )
        self.buffer = TamiyoRolloutBuffer(
            num_envs=num_envs,
            max_steps_per_env=max_steps_per_env,
            state_dim=state_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            slot_config=self.slot_config,
            device=torch.device(device),
        )
        # Validate buffer and policy use same hidden dim and slot config
        assert self.buffer.lstm_hidden_dim == self.policy.hidden_dim, (
            f"Buffer lstm_hidden_dim ({self.buffer.lstm_hidden_dim}) != "
            f"policy hidden_dim ({self.policy.hidden_dim})"
        )
        assert self.buffer.num_slots == self.policy.slot_config.num_slots, (
            f"Buffer num_slots ({self.buffer.num_slots}) != "
            f"policy num_slots ({self.policy.slot_config.num_slots})"
        )
        # CRITICAL: Validate slot_ids ORDERING, not just count
        # This catches observation-action misalignment bugs where configs have
        # same num_slots but different slot orderings (e.g., from SlotConfig.from_specs
        # vs SlotConfig.for_grid). Such misalignment causes silent training corruption
        # where the policy learns phantom correlations between wrong slots.
        assert self.buffer.slot_config.slot_ids == self.policy.slot_config.slot_ids, (
            f"Slot config ordering mismatch! Buffer slot_ids={self.buffer.slot_config.slot_ids} "
            f"!= policy slot_ids={self.policy.slot_config.slot_ids}. "
            f"This WILL cause silent training corruption where actions target wrong slots."
        )
        # M21: Ratio anomaly thresholds from leyline (single source of truth)
        self.ratio_explosion_threshold = DEFAULT_RATIO_EXPLOSION_THRESHOLD
        self.ratio_collapse_threshold = DEFAULT_RATIO_COLLAPSE_THRESHOLD

        # [PyTorch 2.9] Use fused=True for CUDA, foreach=True for CPU
        use_cuda = device.startswith("cuda")
        optimizer_kwargs: dict[str, float | bool] = {'lr': lr, 'eps': 1e-5}
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
            # FactoredRecurrentActorCritic: slot/blueprint/style/tempo/alpha_* /op heads are actors
            # Handle torch.compile wrapper (_orig_mod)
            # getattr AUTHORIZED by Code Review 2025-12-17
            # Justification: torch.compile wraps modules - must unwrap to access head modules
            base_net_untyped = getattr(self.policy.network, '_orig_mod', self.policy.network)
            # Type assertion: we know this is the actual network module
            from esper.tamiyo.networks import FactoredRecurrentActorCritic
            assert isinstance(base_net_untyped, FactoredRecurrentActorCritic)
            base_net = base_net_untyped
            actor_params = (
                list(base_net.slot_head.parameters()) +
                list(base_net.blueprint_head.parameters()) +
                list(base_net.style_head.parameters()) +
                list(base_net.tempo_head.parameters()) +
                list(base_net.alpha_target_head.parameters()) +
                list(base_net.alpha_speed_head.parameters()) +
                list(base_net.alpha_curve_head.parameters()) +
                list(base_net.op_head.parameters())
            )
            critic_params = list(base_net.value_head.parameters())
            shared_params = (
                list(base_net.feature_net.parameters()) +
                list(base_net.lstm.parameters()) +
                list(base_net.lstm_ln.parameters())
            )

            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW([
                {'params': actor_params, 'weight_decay': 0.0, 'name': 'actor'},
                {'params': shared_params, 'weight_decay': 0.0, 'name': 'shared'},  # Must be 0!
                {'params': critic_params, 'weight_decay': weight_decay, 'name': 'critic'},
            ], **optimizer_kwargs)  # type: ignore[arg-type]
        else:
            self.optimizer = torch.optim.Adam(
                self.policy.network.parameters(), **optimizer_kwargs  # type: ignore[arg-type]
            )
        self.train_steps = 0

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

    # TODO: potential dead code - action_mask is not currently threaded through callers.
    def get_entropy_floor(self, action_mask: torch.Tensor | None = None) -> float:
        """Get entropy floor, optionally scaled by valid action count.

        When adaptive_entropy_floor=True, uses information-theoretic scaling:
        scale_factor = log(num_total) / log(num_valid)

        This maintains the same "relative exploration" level - if we want
        10% of max entropy with 5 actions, we want 10% of max entropy with
        2 actions, but max_entropy(2) = log(2) < max_entropy(5) = log(5).

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
    ) -> PPOUpdateMetrics:
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

        # C3: INTENTIONAL SINGLE-BATCH PROCESSING FOR RECURRENT POLICIES
        # We process the entire rollout as a single batch WITHOUT minibatch shuffling.
        # This is intentional for LSTM state coherence:
        #
        # 1. Temporal Dependency: LSTM hidden states encode sequential context.
        #    Shuffling would break temporal dependencies that the LSTM learned during rollout.
        #
        # 2. Hidden State Alignment: We store initial hidden states per sequence and
        #    reconstruct LSTM state during training. Shuffling would misalign hidden
        #    states with their corresponding observations.
        #
        # 3. BPTT Coherence: Backpropagation through time requires contiguous sequences.
        #    Random minibatches would create discontinuous gradients.
        #
        # Trade-off: Higher gradient variance vs correct recurrent credit assignment.
        # For non-recurrent policies, minibatch shuffling would be preferred.
        data = self.buffer.get_batched_sequences(device=self.device)
        valid_mask = data["valid_mask"]

        # Compute explained variance BEFORE updates (intentional - standard practice)
        # This measures how well the value function from rollout collection predicts returns.
        # Post-update EV would measure the updated value function against stale returns,
        # which is less meaningful. High pre-update EV (>0.8) indicates good value estimates.
        valid_values = data["values"][valid_mask]
        valid_returns = data["returns"][valid_mask]
        var_returns = valid_returns.var()
        explained_variance: float
        if var_returns > 1e-8:
            ev_tensor = 1.0 - (valid_returns - valid_values).var() / var_returns
            explained_variance = ev_tensor.item()
        else:
            explained_variance = 0.0

        metrics: dict[str, Any] = defaultdict(list)
        metrics["explained_variance"] = [explained_variance]
        early_stopped = False

        # Compute advantage stats for status banner diagnostics
        # These indicate if advantage normalization is working correctly
        # PERF: Batch mean/std into single GPU→CPU transfer
        valid_advantages_for_stats = data["advantages"][valid_mask]
        if valid_advantages_for_stats.numel() > 0:
            adv_stats = torch.stack([
                valid_advantages_for_stats.mean(),
                valid_advantages_for_stats.std(),
            ]).cpu().tolist()
            metrics["advantage_mean"] = [adv_stats[0]]
            metrics["advantage_std"] = [adv_stats[1]]
        else:
            metrics["advantage_mean"] = [0.0]
            metrics["advantage_std"] = [0.0]

        # Initialize per-head entropy tracking (P3-1)
        head_entropy_history: dict[str, list[float]] = {head: [] for head in HEAD_NAMES}
        # Initialize per-head gradient norm tracking (P4-6)
        head_grad_norm_history: dict[str, list[float]] = {head: [] for head in HEAD_NAMES + ("value",)}

        for epoch_i in range(self.recurrent_n_epochs):
            if early_stopped:
                break

            # Forward pass through network
            actions = {
                "slot": data["slot_actions"],
                "blueprint": data["blueprint_actions"],
                "style": data["style_actions"],
                "tempo": data["tempo_actions"],
                "alpha_target": data["alpha_target_actions"],
                "alpha_speed": data["alpha_speed_actions"],
                "alpha_curve": data["alpha_curve_actions"],
                "op": data["op_actions"],
            }

            masks = {
                "slot": data["slot_masks"],
                "blueprint": data["blueprint_masks"],
                "style": data["style_masks"],
                "tempo": data["tempo_masks"],
                "alpha_target": data["alpha_target_masks"],
                "alpha_speed": data["alpha_speed_masks"],
                "alpha_curve": data["alpha_curve_masks"],
                "op": data["op_masks"],
            }
            result = self.policy.evaluate_actions(
                data["states"],
                actions,
                masks,
                hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
            )
            log_probs = result.log_prob
            values = result.value
            entropy = result.entropy

            # Extract valid timesteps
            for key in log_probs:
                log_probs[key] = log_probs[key][valid_mask]
            values = values[valid_mask]
            for key in entropy:
                entropy[key] = entropy[key][valid_mask]

            # Track per-head entropy (P3-1)
            # PERF: Batch all 8 head entropies into single GPU→CPU transfer
            head_entropy_tensors = [entropy[key].mean() for key in HEAD_NAMES]
            head_entropy_values = torch.stack(head_entropy_tensors).cpu().tolist()
            for key, val in zip(HEAD_NAMES, head_entropy_values):
                head_entropy_history[key].append(val)

            valid_advantages = data["advantages"][valid_mask]
            valid_returns = data["returns"][valid_mask]

            # Compute per-head advantages with causal masking
            valid_op_actions = data["op_actions"][valid_mask]
            per_head_advantages = compute_per_head_advantages(
                valid_advantages, valid_op_actions
            )

            # B4-DRL-01: Use single source of truth for causal masks (from leyline)
            # Causal structure documented in esper/leyline/causal_masks.py
            head_masks = compute_causal_masks(valid_op_actions)

            # Compute per-head ratios
            old_log_probs = {
                "slot": data["slot_log_probs"][valid_mask],
                "blueprint": data["blueprint_log_probs"][valid_mask],
                "style": data["style_log_probs"][valid_mask],
                "tempo": data["tempo_log_probs"][valid_mask],
                "alpha_target": data["alpha_target_log_probs"][valid_mask],
                "alpha_speed": data["alpha_speed_log_probs"][valid_mask],
                "alpha_curve": data["alpha_curve_log_probs"][valid_mask],
                "op": data["op_log_probs"][valid_mask],
            }

            per_head_ratios = {}
            for key in HEAD_NAMES:
                # Clamp log-ratio to prevent inf/NaN from exp() when probabilities diverge
                # significantly. log(exp(20)) ≈ 4.85e8 is already extreme; log(exp(88)) overflows.
                # This provides early protection before ratio explosion detection (lines 809-821).
                log_ratio = log_probs[key] - old_log_probs[key]
                log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
                per_head_ratios[key] = torch.exp(log_ratio_clamped)

            # Compute KL divergence EARLY (before optimizer step) for effective early stopping
            # BUG-003 FIX: With recurrent_n_epochs=1, the old check at loop end was useless
            # because there's no "next epoch" to skip. By checking here, we can skip the
            # optimizer.step() entirely if KL is already too high.
            #
            # KL(old||new) ≈ E[(ratio - 1) - log(ratio)] (KL3 estimator from Schulman)
            #
            # H6 FIX: Weight per-head KL by causal relevance.
            # Sparse heads (blueprint, style) are only active during GERMINATE (~5-15%).
            # Without weighting, they contribute full KL to the sum despite being rarely
            # causally relevant, inflating joint KL and triggering premature early stopping.
            #
            # PyTorch Expert Review 2025-12-26: Normalize weights to sum to 1.0 for proper
            # weighted average. Without normalization, joint KL was biased toward high-activity
            # heads since overlapping masks (slot ~90%, blueprint ~10%) don't sum to 1.0.
            with torch.inference_mode():
                total_timesteps = valid_mask.sum().float().clamp(min=1)
                weighted_kl_sum = torch.tensor(0.0, device=self.device)
                total_weight = torch.tensor(0.0, device=self.device)
                for key in HEAD_NAMES:
                    mask = head_masks[key]
                    log_ratio = log_probs[key] - old_log_probs[key]
                    kl_per_step = (torch.exp(log_ratio) - 1) - log_ratio
                    n_valid = mask.sum().float().clamp(min=1)
                    # Masked mean KL for this head
                    head_kl = (kl_per_step * mask.float()).sum() / n_valid
                    # Weight by fraction of timesteps where head is causally relevant
                    causal_weight = n_valid / total_timesteps
                    weighted_kl_sum = weighted_kl_sum + causal_weight * head_kl
                    total_weight = total_weight + causal_weight
                # Normalize to proper weighted average (weights sum to 1)
                approx_kl = (weighted_kl_sum / total_weight.clamp(min=1e-8)).item()
                metrics["approx_kl"].append(approx_kl)

                # Clip fraction: how often clipping was active
                joint_ratio = per_head_ratios["op"]
                clip_fraction_t = ((joint_ratio - 1.0).abs() > self.clip_ratio).float().mean()
                # Defer .item() - will be synced with approx_kl check anyway
                metrics["clip_fraction"].append(clip_fraction_t.cpu().item())

                # Early stopping: if KL exceeds threshold, skip this update entirely
                # 1.5x multiplier is standard (OpenAI baselines, Stable-Baselines3)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    metrics["early_stop_epoch"] = [epoch_i]
                    # Record ratio metrics even when early stopping
                    metrics["ratio_mean"].append(joint_ratio.mean().item())
                    metrics["ratio_max"].append(joint_ratio.max().item())
                    metrics["ratio_min"].append(joint_ratio.min().item())
                    break  # Skip loss computation, backward, and optimizer step

            # Compute policy loss per head and sum
            # Use masked mean to avoid bias from averaging zeros with real values
            policy_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
            for key in HEAD_NAMES:
                ratio = per_head_ratios[key]
                adv = per_head_advantages[key]
                mask = head_masks[key]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                clipped_surr = torch.min(surr1, surr2)

                # Masked mean: only average over causally-relevant positions
                # This prevents zeros from masked positions from biasing the loss
                n_valid = mask.sum().clamp(min=1)  # Avoid div-by-zero
                head_loss = -(clipped_surr * mask.float()).sum() / n_valid
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

            # Entropy loss with per-head weighting and causal masking.
            # H3 FIX: Use masked mean for sparse heads (blueprint, style).
            # These heads are only active during GERMINATE (~5-15% of timesteps).
            # Without masking, entropy gradient is diluted by averaging over zeros,
            # starving exploration signal for rare-but-important action heads.
            #
            # NOTE: Entropy floors were removed because torch.clamp to a constant
            # provides zero gradient (d(constant)/d(params) = 0). When a head has
            # only one valid action, entropy is correctly 0 with no gradient signal.
            entropy_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
            for key, ent in entropy.items():
                head_coef = self.entropy_coef_per_head.get(key, 1.0)
                mask = head_masks[key]
                n_valid = mask.sum().clamp(min=1)
                masked_ent = (ent * mask.float()).sum() / n_valid
                entropy_loss = entropy_loss - head_coef * masked_ent

            entropy_coef = self.get_entropy_coef()

            loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]

            # Collect per-head gradient norms BEFORE clipping (P4-6)
            # Measures raw gradients to diagnose head dominance
            # PERF: Batch all norm computations, then single .tolist() at end
            with torch.inference_mode():
                # Handle torch.compile wrapper (_orig_mod)
                # getattr AUTHORIZED by Code Review 2025-12-17
                # Justification: torch.compile wraps modules - must unwrap to access head modules
                base_net_untyped = getattr(self.policy.network, '_orig_mod', self.policy.network)
                # Type assertion: we know this is the actual network module
                from esper.tamiyo.networks import FactoredRecurrentActorCritic
                assert isinstance(base_net_untyped, FactoredRecurrentActorCritic)
                base_net = base_net_untyped

                head_names = ["slot", "blueprint", "style", "tempo", "alpha_target",
                              "alpha_speed", "alpha_curve", "op", "value"]
                head_modules = [
                    base_net.slot_head, base_net.blueprint_head, base_net.style_head,
                    base_net.tempo_head, base_net.alpha_target_head, base_net.alpha_speed_head,
                    base_net.alpha_curve_head, base_net.op_head, base_net.value_head,
                ]

                # Collect all head norms as tensors (no .item() yet)
                head_norm_tensors: list[torch.Tensor] = []
                for head_module in head_modules:
                    params_with_grad = [p for p in head_module.parameters() if p.grad is not None]
                    if params_with_grad:
                        norm_t = torch.linalg.vector_norm(
                            torch.stack([torch.linalg.vector_norm(p.grad) for p in params_with_grad])
                        )
                    else:
                        norm_t = torch.tensor(0.0, device=self.device)
                    head_norm_tensors.append(norm_t)

                # Single GPU→CPU sync: stack all norms, then .tolist()
                all_norms = torch.stack(head_norm_tensors).cpu().tolist()
                for head_name, grad_norm in zip(head_names, all_norms):
                    head_grad_norm_history[head_name].append(grad_norm)

            nn.utils.clip_grad_norm_(self.policy.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            # PERF: Batch all 10 logging metrics into single GPU→CPU transfer
            joint_ratio = per_head_ratios["op"]  # Use op ratio as representative
            logging_tensors = torch.stack([
                policy_loss,
                value_loss,
                -entropy_loss,  # negate here to match original semantics
                joint_ratio.mean(),
                joint_ratio.max(),
                joint_ratio.min(),
                # Value function stats (single GPU sync with rest)
                values.mean(),
                values.std(),
                values.min(),
                values.max(),
            ]).cpu().tolist()
            metrics["policy_loss"].append(logging_tensors[0])
            metrics["value_loss"].append(logging_tensors[1])
            metrics["entropy"].append(logging_tensors[2])
            metrics["ratio_mean"].append(logging_tensors[3])
            metrics["ratio_max"].append(logging_tensors[4])
            metrics["ratio_min"].append(logging_tensors[5])
            metrics["value_mean"].append(logging_tensors[6])
            metrics["value_std"].append(logging_tensors[7])
            metrics["value_min"].append(logging_tensors[8])
            metrics["value_max"].append(logging_tensors[9])
            if (
                joint_ratio.max() > self.ratio_explosion_threshold
                or joint_ratio.min() < self.ratio_collapse_threshold
            ):
                diag = RatioExplosionDiagnostic.from_batch(
                    ratio=joint_ratio.flatten(),
                    old_log_probs=old_log_probs["op"].flatten(),
                    new_log_probs=log_probs["op"].flatten(),
                    actions=valid_op_actions.flatten(),
                    max_threshold=self.ratio_explosion_threshold,
                    min_threshold=self.ratio_collapse_threshold,
                )
                metrics.setdefault("ratio_diagnostic", []).append(diag.to_dict())

        self.train_steps += 1

        if clear_buffer:
            self.buffer.reset()

        # Aggregate into typed result dict
        # TypedDict doesn't support dynamic key assignment, so we use type: ignore
        # The aggregation converts list[float] to float for most metrics
        aggregated_result: PPOUpdateMetrics = {}
        for k, v in metrics.items():
            if not v:
                aggregated_result[k] = 0.0  # type: ignore[literal-required]
                continue

            first = v[0]
            if isinstance(first, dict):
                # Diagnostic payloads (ratio_diagnostic) are not aggregated
                aggregated_result[k] = first  # type: ignore[literal-required]
            else:
                # Average across epochs (converts list[float] to float)
                aggregated_result[k] = sum(v) / len(v)  # type: ignore[literal-required]

        # Add per-head entropy tracking (P3-1)
        aggregated_result["head_entropies"] = head_entropy_history
        # Add per-head gradient norm tracking (P4-6)
        aggregated_result["head_grad_norms"] = head_grad_norm_history

        return aggregated_result

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get network state dict from policy
        # Handle torch.compile wrapper (_orig_mod)
        # getattr AUTHORIZED by Code Review 2025-12-17
        # Justification: torch.compile wraps modules - must unwrap to access state_dim
        base_net_untyped = getattr(self.policy.network, '_orig_mod', self.policy.network)
        # Type assertion: we know this is the actual network module
        from esper.tamiyo.networks import FactoredRecurrentActorCritic
        assert isinstance(base_net_untyped, FactoredRecurrentActorCritic)
        base_net = base_net_untyped
        save_dict = {
            # Version for forward compatibility
            'checkpoint_version': CHECKPOINT_VERSION,
            'network_state_dict': self.policy.state_dict(),
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
                'entropy_coef_per_head': self.entropy_coef_per_head,
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
                # Training hyperparameters
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'max_grad_norm': self.max_grad_norm,
                'weight_decay': self.weight_decay,
                # torch.compile configuration (for resume)
                'compile_mode': self.compile_mode,
            },
            # Architecture info for load-time reconstruction
            'architecture': {
                'state_dim': base_net.state_dim,
                # Slot configuration (critical for network shape)
                'slot_ids': self.slot_config.slot_ids,
                'num_slots': self.slot_config.num_slots,
            }
        }
        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda:0") -> "PPOAgent":
        """Load agent from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model onto

        Returns:
            PPOAgent with restored weights and configuration

        Raises:
            RuntimeError: If checkpoint architecture is incompatible
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        # Required checkpoint fields - fail fast if missing (no backwards compat)
        try:
            version = checkpoint['checkpoint_version']
            state_dict = checkpoint['network_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            architecture = checkpoint['architecture']
            config = checkpoint['config']
            train_steps = checkpoint['train_steps']
        except KeyError as e:
            raise RuntimeError(
                f"Incompatible checkpoint format: missing required field {e}. "
                f"This checkpoint was saved with an older version that is no longer supported. "
                f"Please retrain the model to create a compatible checkpoint."
            ) from e

        # M6: Free checkpoint memory immediately after extracting needed data.
        # Checkpoint holds a full copy of all model weights; waiting for GC to
        # free this can cause OOM when loading large models on GPU.
        del checkpoint

        # Version validation - only CHECKPOINT_VERSION is supported
        if version != CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Checkpoint version mismatch: got {version}, expected {CHECKPOINT_VERSION}. "
                f"This checkpoint is incompatible. Please retrain the model."
            )

        # === Reconstruct SlotConfig ===
        # slot_ids is a required field in architecture (saved by save() since v1)
        if 'slot_ids' not in architecture:
            raise RuntimeError(
                "Incompatible checkpoint: architecture.slot_ids is required. "
                "Please retrain the model to create a compatible checkpoint."
            )
        slot_config = SlotConfig(slot_ids=tuple(architecture['slot_ids']))

        # === Pre-load validation ===
        expected_num_slots = slot_config.num_slots
        actual_num_slots = state_dict['slot_head.2.weight'].shape[0]
        if expected_num_slots != actual_num_slots:
            raise RuntimeError(
                f"Checkpoint slot count mismatch: "
                f"slot_config has {expected_num_slots} slots, "
                f"but checkpoint weights have {actual_num_slots} slots. "
                f"Saved slot_ids: {architecture['slot_ids']}"
            )

        # === Infer state_dim ===
        # feature_net.0.weight has shape [hidden_dim, state_dim]
        state_dim = state_dict['feature_net.0.weight'].shape[1]

        # === Extract compile_mode (default "off" for old checkpoints) ===
        compile_mode = config.get('compile_mode', 'off')

        # === Create PolicyBundle (uncompiled - compile AFTER loading weights) ===
        from esper.tamiyo.policy.factory import create_policy
        policy = create_policy(
            policy_type="lstm",
            feature_dim=state_dim,
            slot_config=slot_config,
            hidden_dim=config['lstm_hidden_dim'],
            device=device,
            compile_mode="off",  # Defer compilation until weights loaded
        )

        # === Create agent with restored config ===
        # Remove config params that are now part of PolicyBundle
        agent_config = {k: v for k, v in config.items()
                       if k not in ('lstm_hidden_dim',)}
        agent = cls(
            policy=policy,
            slot_config=slot_config,
            device=device,
            **agent_config
        )

        # === Load weights ===
        agent.policy.load_state_dict(state_dict)
        agent.optimizer.load_state_dict(optimizer_state_dict)
        agent.train_steps = train_steps

        # === Apply torch.compile AFTER loading weights ===
        # Critical: Compile must happen after state_dict to ensure graph traces
        # the actual loaded weights, not random initialization.
        if compile_mode != "off":
            agent.policy.compile(mode=compile_mode, dynamic=True)

        return agent


__all__ = [
    "PPOAgent",
    "signals_to_features",
]
