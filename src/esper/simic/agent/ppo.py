"""Simic PPO Module - PPO Agent for Seed Lifecycle Control

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rollout_buffer import TamiyoRolloutBuffer
from .advantages import compute_per_head_advantages
from .types import PPOUpdateMetrics
from esper.simic.telemetry import RatioExplosionDiagnostic
from esper.leyline import (
    PolicyBundle,
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
    LifecycleOp,
    NUM_OPS,
)
from esper.leyline.slot_config import SlotConfig
import logging

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Checkpoint format version for forward compatibility
# Increment when checkpoint structure changes in backwards-incompatible ways
CHECKPOINT_VERSION = 1

# Sparse heads need higher entropy coefficients to maintain exploration
# when they receive fewer training signals due to causal masking
ENTROPY_COEF_PER_HEAD: dict[str, float] = {
    "op": 1.0,  # Always active (100% of steps)
    "slot": 1.0,  # Usually active (~60%)
    "blueprint": 1.3,  # GERMINATE only (~18%) — needs boost
    "style": 1.2,  # GERMINATE + SET_ALPHA_TARGET (~22%)
    "tempo": 1.3,  # GERMINATE only (~18%) — needs boost
    "alpha_target": 1.2,  # GERMINATE + SET_ALPHA_TARGET (~22%)
    "alpha_speed": 1.2,  # SET_ALPHA_TARGET + PRUNE (~19%)
    "alpha_curve": 1.2,  # SET_ALPHA_TARGET + PRUNE (~19%)
}
# Note: Start conservative (1.2-1.3x), tune empirically if heads collapse


# =============================================================================
# Helper Functions
# =============================================================================

def _init_q_metrics_nan() -> dict[str, list[float]]:
    """Initialize Q-value metrics with NaN when no valid states available."""
    nan = float("nan")
    return {
        "q_germinate": [nan],
        "q_advance": [nan],
        "q_fossilize": [nan],
        "q_prune": [nan],
        "q_wait": [nan],
        "q_set_alpha": [nan],
        "q_variance": [nan],
        "q_spread": [nan],
    }


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
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
        self.entropy_anneal_steps = entropy_anneal_steps
        # Per-head entropy multipliers (differential coefficients for sparse heads)
        self.entropy_coef_per_head = entropy_coef_per_head or ENTROPY_COEF_PER_HEAD
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
                list(base_net.lstm_ln.parameters()) +
                list(base_net.blueprint_embedding.parameters())  # Phase 4: blueprint embeddings
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

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient with annealing and floor.

        Returns:
            Entropy coefficient (annealed if enabled, floored at entropy_coef_min)

        Note:
            The adaptive_entropy_floor feature was removed (B11-DRL-02). It scaled
            the floor by mask density, but MaskedCategorical already normalizes
            entropy by log(num_valid), making the adaptive scaling redundant and
            potentially harmful (over-exploration in masked states).
        """
        if self.entropy_anneal_steps == 0:
            # No annealing - use fixed coefficient with base floor
            return max(self.entropy_coef, self.entropy_coef_min)

        # With annealing - interpolate and apply base floor
        progress = min(1.0, self.train_steps / self.entropy_anneal_steps)
        annealed = self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)
        return max(annealed, self.entropy_coef_min)

    def _collect_cuda_memory_metrics(self) -> dict[str, float]:
        """Collect CUDA memory statistics for infrastructure monitoring.

        Called once per PPO update (not per inner epoch) to amortize sync overhead.
        Returns empty dict if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return {}

        # Get device index from policy device string
        device_str = str(self.device)
        if device_str == "cpu":
            return {}

        # torch.cuda.memory_allocated() and friends trigger CPU-GPU sync
        # Calling once per update() is acceptable overhead (~1ms)
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
        peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)  # GB

        # Fragmentation: 1 - (allocated/reserved), >0.3 indicates memory pressure
        fragmentation = 1.0 - (allocated / reserved) if reserved > 0 else 0.0

        return {
            "cuda_memory_allocated_gb": allocated,
            "cuda_memory_reserved_gb": reserved,
            "cuda_memory_peak_gb": peak,
            "cuda_memory_fragmentation": fragmentation,
        }

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

        # Collect CUDA memory metrics once per update (amortize sync overhead)
        cuda_memory_metrics = self._collect_cuda_memory_metrics()

        # Compute advantage stats for status banner diagnostics
        # These indicate if advantage normalization is working correctly
        # PERF: Batch mean/std into single GPU→CPU transfer
        valid_advantages_for_stats = data["advantages"][valid_mask]
        if valid_advantages_for_stats.numel() > 0:
            adv_mean = valid_advantages_for_stats.mean()
            adv_std = valid_advantages_for_stats.std()
            # Compute skewness and kurtosis (both use same std threshold)
            # Use 0.05 threshold to match UI "low warning" - below this, σ^n division unstable
            if adv_std > 0.05:
                centered = valid_advantages_for_stats - adv_mean
                # Skewness: E[(X-μ)³] / σ³ - asymmetry indicator
                adv_skewness = (centered ** 3).mean() / (adv_std ** 3)
                # Excess kurtosis: E[(X-μ)⁴] / σ⁴ - 3 (0 = normal, >0 = heavy tails)
                adv_kurtosis = (centered ** 4).mean() / (adv_std ** 4) - 3.0
            else:
                # NaN signals "undefined" - std too low for meaningful higher moments
                adv_skewness = torch.tensor(float("nan"), device=adv_mean.device, dtype=adv_mean.dtype)
                adv_kurtosis = torch.tensor(float("nan"), device=adv_mean.device, dtype=adv_mean.dtype)
            adv_stats = torch.stack([adv_mean, adv_std, adv_skewness, adv_kurtosis]).cpu().tolist()
            metrics["advantage_mean"] = [adv_stats[0]]
            metrics["advantage_std"] = [adv_stats[1]]
            metrics["advantage_skewness"] = [adv_stats[2]]
            metrics["advantage_kurtosis"] = [adv_stats[3]]
            # Fraction of positive advantages (healthy: 40-60%)
            # Imbalanced ratios suggest reward design issues or easy/hard regions
            adv_positive_ratio = (valid_advantages_for_stats > 0).float().mean().item()
            metrics["advantage_positive_ratio"] = [adv_positive_ratio]
        else:
            # No valid advantages - use NaN to signal "no data" (not "balanced" or "zero")
            metrics["advantage_mean"] = [float("nan")]
            metrics["advantage_std"] = [float("nan")]
            metrics["advantage_skewness"] = [float("nan")]
            metrics["advantage_kurtosis"] = [float("nan")]
            metrics["advantage_positive_ratio"] = [float("nan")]

        # === Collect Op-Conditioned Q-Values (Policy V2) ===
        # Compute Q(s, op) for all ops using a representative state
        # Use first valid state from batch to avoid bias from terminal states
        if valid_mask.any():
            # Get first valid state
            first_valid_idx = valid_mask.nonzero(as_tuple=True)
            if len(first_valid_idx[0]) > 0:
                sample_state_idx = (int(first_valid_idx[0][0].item()), int(first_valid_idx[1][0].item()))
                sample_obs = data["states"][sample_state_idx[0], sample_state_idx[1]].unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
                sample_blueprints = data["blueprint_indices"][sample_state_idx[0], sample_state_idx[1]].unsqueeze(0).unsqueeze(0)  # [1, 1, num_slots]

                # Forward pass to get LSTM output
                with torch.no_grad():
                    forward_result = self.policy.network.forward(
                        state=sample_obs,
                        blueprint_indices=sample_blueprints,
                        hidden=None,  # Use initial hidden state for consistency
                    )
                    lstm_out = forward_result["lstm_out"]  # [1, 1, hidden_dim]

                # Compute Q(s, op) for each op
                # Build mapping from LifecycleOp to Q-values
                q_value_map: dict[LifecycleOp, float] = {}
                for op_idx in range(NUM_OPS):
                    op_tensor = torch.tensor([[op_idx]], dtype=torch.long, device=self.device)
                    q_val = self.policy.network._compute_value(lstm_out, op_tensor)  # type: ignore[operator]
                    q_value_map[LifecycleOp(op_idx)] = q_val.item()

                # Assign to metrics with correct names using actual LifecycleOp enum
                # LifecycleOp: WAIT=0, GERMINATE=1, SET_ALPHA_TARGET=2, PRUNE=3, FOSSILIZE=4, ADVANCE=5
                metrics["q_germinate"] = [q_value_map[LifecycleOp.GERMINATE]]
                metrics["q_advance"] = [q_value_map[LifecycleOp.ADVANCE]]
                metrics["q_fossilize"] = [q_value_map[LifecycleOp.FOSSILIZE]]
                metrics["q_prune"] = [q_value_map[LifecycleOp.PRUNE]]
                metrics["q_wait"] = [q_value_map[LifecycleOp.WAIT]]
                metrics["q_set_alpha"] = [q_value_map[LifecycleOp.SET_ALPHA_TARGET]]

                # Compute Q-variance and Q-spread
                q_values = list(q_value_map.values())
                q_variance = float(torch.tensor(q_values).var().item())
                q_spread = max(q_values) - min(q_values)
                metrics["q_variance"] = [q_variance]
                metrics["q_spread"] = [q_spread]
            else:
                # No valid states - use NaN
                metrics.update(_init_q_metrics_nan())
        else:
            # No valid data - use NaN
            metrics.update(_init_q_metrics_nan())

        # Initialize per-head entropy tracking (P3-1)
        head_entropy_history: dict[str, list[float]] = {head: [] for head in HEAD_NAMES}
        # Initialize per-head gradient norm tracking (P4-6)
        head_grad_norm_history: dict[str, list[float]] = {head: [] for head in HEAD_NAMES + ("value",)}
        # Initialize log prob extremes tracking (NaN predictor)
        # Very negative log probs (<-50 warning, <-100 critical) predict numerical underflow
        # Use inf/-inf for proper min/max tracking (log probs are always <= 0)
        log_prob_min_across_epochs: float = float("inf")
        log_prob_max_across_epochs: float = float("-inf")

        # Initialize per-head ratio max tracking (Policy V2 - multi-head ratio explosion detection)
        # Track max ratio per head across all epochs for telemetry
        head_ratio_max_across_epochs: dict[str, float] = {head: float("-inf") for head in HEAD_NAMES}
        joint_ratio_max_across_epochs: float = float("-inf")

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
                data["blueprint_indices"],
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

            # FINITENESS GATE: Detect NaN/Inf early to identify source of numerical instability
            # This is NOT bug-hiding - it's fail-fast on corrupted probabilities.
            # Common causes: mask mismatch between rollout and update, degenerate distributions
            nonfinite_found = False
            nonfinite_sources: list[str] = []

            # Check new log_probs
            for key in HEAD_NAMES:
                if not torch.isfinite(log_probs[key]).all():
                    nonfinite_count = (~torch.isfinite(log_probs[key])).sum().item()
                    nonfinite_sources.append(f"log_probs[{key}]: {nonfinite_count} non-finite")
                    nonfinite_found = True

            # Check old log_probs (stored as "{key}_log_probs" in data dict)
            for key in HEAD_NAMES:
                old_key = f"{key}_log_probs"
                old_lp = data[old_key][valid_mask]
                if not torch.isfinite(old_lp).all():
                    nonfinite_count = (~torch.isfinite(old_lp)).sum().item()
                    nonfinite_sources.append(f"old_log_probs[{key}]: {nonfinite_count} non-finite")
                    nonfinite_found = True

            # Check values
            if not torch.isfinite(values).all():
                nonfinite_count = (~torch.isfinite(values)).sum().item()
                nonfinite_sources.append(f"values: {nonfinite_count} non-finite")
                nonfinite_found = True

            if nonfinite_found:
                logger.warning(
                    f"FINITENESS GATE FAILED at epoch {epoch_i}: {', '.join(nonfinite_sources)}. "
                    "Skipping optimizer step to prevent NaN propagation. "
                    "Likely cause: mask mismatch between rollout and update time."
                )
                metrics.setdefault("finiteness_gate_failures", []).append({
                    "epoch": epoch_i,
                    "sources": nonfinite_sources,
                })
                continue  # Skip this epoch's update, try next epoch

            # Track log prob extremes across all heads (NaN predictor)
            # Use HEAD_NAMES for consistent ordering
            all_log_probs = torch.cat([log_probs[k] for k in HEAD_NAMES])
            if all_log_probs.numel() > 0:
                # Batch min/max into single GPU->CPU transfer
                epoch_extremes = torch.stack([all_log_probs.min(), all_log_probs.max()]).cpu().tolist()
                epoch_log_prob_min, epoch_log_prob_max = epoch_extremes
                log_prob_min_across_epochs = min(log_prob_min_across_epochs, epoch_log_prob_min)
                log_prob_max_across_epochs = max(log_prob_max_across_epochs, epoch_log_prob_max)

            # Track per-head entropy (P3-1)
            # PERF: Batch all 8 head entropies into single GPU→CPU transfer
            head_entropy_tensors = [entropy[key].mean() for key in HEAD_NAMES]
            head_entropy_values = torch.stack(head_entropy_tensors).cpu().tolist()
            for key, val in zip(HEAD_NAMES, head_entropy_values):
                head_entropy_history[key].append(val)

            valid_advantages = data["advantages"][valid_mask]
            valid_returns = data["returns"][valid_mask]

            # Compute per-head advantages with causal masking
            # Returns both advantages AND masks to avoid redundant computation
            valid_op_actions = data["op_actions"][valid_mask]
            # Use effective op (action_for_reward) for causal masks to avoid crediting invalid ops.
            valid_effective_op_actions = data["effective_op_actions"][valid_mask]
            per_head_advantages, head_masks = compute_per_head_advantages(
                valid_advantages, valid_effective_op_actions
            )
            # B4-DRL-01: Masks from leyline.causal_masks (single source of truth)

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
            log_ratios_for_joint: dict[str, torch.Tensor] = {}  # Store for joint ratio computation
            for key in HEAD_NAMES:
                # Clamp log-ratio to prevent inf/NaN from exp() when probabilities diverge
                # significantly. log(exp(20)) ≈ 4.85e8 is already extreme; log(exp(88)) overflows.
                # This provides early protection before ratio explosion detection (lines 809-821).
                log_ratio = log_probs[key] - old_log_probs[key]
                log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
                log_ratios_for_joint[key] = log_ratio_clamped
                per_head_ratios[key] = torch.exp(log_ratio_clamped)

            # Track per-head ratio max across epochs (Policy V2 telemetry)
            # PERF: Batch all 8 head ratio max values into single GPU→CPU transfer
            with torch.inference_mode():
                per_head_ratio_max_tensors = [per_head_ratios[k].max() for k in HEAD_NAMES]
                per_head_ratio_max_values = torch.stack(per_head_ratio_max_tensors).cpu().tolist()
                for key, max_val in zip(HEAD_NAMES, per_head_ratio_max_values):
                    head_ratio_max_across_epochs[key] = max(head_ratio_max_across_epochs[key], max_val)

                # Compute joint ratio using log-space summation (numerically stable)
                # joint_ratio = exp(sum(log_ratio_i)) = product(ratio_i)
                stacked_log_ratios = torch.stack([log_ratios_for_joint[k] for k in HEAD_NAMES])
                joint_log_ratio = stacked_log_ratios.sum(dim=0)  # Sum across heads per timestep
                joint_log_ratio_clamped = torch.clamp(joint_log_ratio, min=-30.0, max=30.0)
                joint_ratio = torch.exp(joint_log_ratio_clamped)
                epoch_joint_ratio_max = joint_ratio.max().item()
                joint_ratio_max_across_epochs = max(joint_ratio_max_across_epochs, epoch_joint_ratio_max)

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
                    # BUGFIX: Reuse CLAMPED log_ratio from line 602 to prevent overflow/NaN
                    # Previously this recomputed log_ratio without clamping, causing:
                    # - exp(large_value) → inf → NaN in KL
                    # - exp(-inf) → 0, but (0-1) - (-inf) → NaN
                    # The clamped version is already available in log_ratios_for_joint
                    log_ratio_clamped = log_ratios_for_joint[key]
                    kl_per_step = (torch.exp(log_ratio_clamped) - 1) - log_ratio_clamped
                    n_valid = mask.sum().float().clamp(min=1)
                    # Masked mean KL for this head
                    head_kl = (kl_per_step * mask).sum() / n_valid
                    # Weight by fraction of timesteps where head is causally relevant
                    causal_weight = n_valid / total_timesteps
                    weighted_kl_sum = weighted_kl_sum + causal_weight * head_kl
                    total_weight = total_weight + causal_weight
                # Normalize to proper weighted average (weights sum to 1)
                approx_kl = (weighted_kl_sum / total_weight.clamp(min=1e-8)).item()
                metrics["approx_kl"].append(approx_kl)

                # Clip fraction: how often clipping was active
                # PERF: Batch all 3 clip metrics into single GPU→CPU transfer
                # BUG FIX: Use TRUE joint_ratio (product of all heads) computed at line 619
                # Previously this was overwritten with just per_head_ratios["op"], hiding
                # divergence in other heads (slot, blueprint, style, etc.)
                clip_fraction_t = ((joint_ratio - 1.0).abs() > self.clip_ratio).float().mean()
                # Directional clip fractions (per DRL expert recommendation)
                # Tracks WHERE clipping occurs: asymmetry indicates directional policy drift
                clip_pos = (joint_ratio > 1.0 + self.clip_ratio).float().mean()
                clip_neg = (joint_ratio < 1.0 - self.clip_ratio).float().mean()
                # Single GPU→CPU sync for all 3 clip metrics
                clip_metrics = torch.stack([clip_fraction_t, clip_pos, clip_neg]).cpu().tolist()
                metrics["clip_fraction"].append(clip_metrics[0])
                metrics["clip_fraction_positive"].append(clip_metrics[1])
                metrics["clip_fraction_negative"].append(clip_metrics[2])

                # Early stopping: if KL exceeds threshold, skip this update entirely
                # 1.5x multiplier is standard (OpenAI baselines, Stable-Baselines3)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    metrics["early_stop_epoch"] = [epoch_i]
                    # PERF: Batch ratio metrics into single GPU→CPU transfer
                    ratio_stats = torch.stack([
                        joint_ratio.mean(), joint_ratio.max(), joint_ratio.min(), joint_ratio.std()
                    ]).cpu().tolist()
                    metrics["ratio_mean"].append(ratio_stats[0])
                    metrics["ratio_max"].append(ratio_stats[1])
                    metrics["ratio_min"].append(ratio_stats[2])
                    metrics["ratio_std"].append(ratio_stats[3])
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
                head_loss = -(clipped_surr * mask).sum() / n_valid
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
                masked_ent = (ent * mask).sum() / n_valid
                entropy_loss = entropy_loss - head_coef * masked_ent

            entropy_coef = self.get_entropy_coef()

            loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]

            # DEBUG: Check gradient flow to heads (one-time probe)
            if epoch_i == 0 and not hasattr(self, "_gradient_debug_done"):
                self._gradient_debug_done = True
                network = self.policy.network
                logger.warning("=== GRADIENT DEBUG PROBE ===")

                # Check loss computation graph
                logger.warning(f"loss: requires_grad={loss.requires_grad}, grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}")
                logger.warning(f"policy_loss: requires_grad={policy_loss.requires_grad}, grad_fn={type(policy_loss.grad_fn).__name__ if policy_loss.grad_fn else None}")
                logger.warning(f"value_loss: requires_grad={value_loss.requires_grad}, grad_fn={type(value_loss.grad_fn).__name__ if value_loss.grad_fn else None}")

                # Check if log_probs require grad
                for key in ["op", "slot", "blueprint"]:
                    if key in log_probs:
                        lp = log_probs[key]
                        logger.warning(f"log_probs[{key}]: requires_grad={lp.requires_grad}, grad_fn={type(lp.grad_fn).__name__ if lp.grad_fn else None}")

                # Check MASK COLLAPSE - if heads have only 1 valid action, no gradient flows
                logger.warning("=== MASK COLLAPSE CHECK (action validity) ===")
                for key in ["op", "slot", "blueprint", "style"]:
                    if key in masks:
                        mask = masks[key]  # [batch, seq, num_actions] or similar
                        valid_counts = mask.sum(dim=-1).float()  # How many valid per timestep
                        min_valid = valid_counts.min().item()
                        max_valid = valid_counts.max().item()
                        mean_valid = valid_counts.mean().item()
                        pct_single = (valid_counts == 1).float().mean().item() * 100
                        logger.warning(f"{key}_mask: valid_actions min={min_valid:.0f} max={max_valid:.0f} mean={mean_valid:.1f} single_action={pct_single:.1f}%")

                # Check CAUSAL MASKS - whether head contributes to loss at all
                logger.warning("=== CAUSAL MASK CHECK (head relevance) ===")
                for key in ["op", "slot", "blueprint", "style"]:
                    if key in head_masks:
                        causal_mask = head_masks[key]  # [batch*seq] or similar - True if head is relevant
                        pct_relevant = causal_mask.float().mean().item() * 100
                        n_relevant = causal_mask.sum().item()
                        n_total = causal_mask.numel()
                        logger.warning(f"{key}_causal: {n_relevant}/{n_total} timesteps relevant ({pct_relevant:.1f}%)")

                # Check ratio computation
                for key in ["op", "slot"]:
                    if key in per_head_ratios:
                        ratio = per_head_ratios[key]
                        logger.warning(f"per_head_ratios[{key}]: requires_grad={ratio.requires_grad}, grad_fn={type(ratio.grad_fn).__name__ if ratio.grad_fn else None}")

                # Check network type (compiled vs original)
                logger.warning(f"Network type: {type(network).__name__}")
                has_orig_mod = hasattr(network, '_orig_mod')
                logger.warning(f"Is compiled (has _orig_mod): {has_orig_mod}")

                # If compiled, also check original module's params
                base_network = getattr(network, '_orig_mod', network)
                logger.warning(f"Base network type: {type(base_network).__name__}")

                # Check which params have grads (on compiled wrapper)
                debug_params_with_grad = sum(1 for p in network.parameters() if p.grad is not None)
                total_params = sum(1 for _ in network.parameters())
                logger.warning(f"Compiled network params with grad: {debug_params_with_grad}/{total_params}")

                # Check which params have grads (on original)
                if has_orig_mod:
                    orig_params_with_grad = sum(1 for p in base_network.parameters() if p.grad is not None)
                    orig_total_params = sum(1 for _ in base_network.parameters())
                    logger.warning(f"Original network params with grad: {orig_params_with_grad}/{orig_total_params}")

                # Check head params specifically - use BASE network for attribute access
                # Distinguish: None (not in graph) vs NaN (numerical instability) vs finite (healthy)
                for name in ["slot_head", "op_head", "value_head"]:
                    head = getattr(base_network, name)
                    head_params = list(head.parameters())
                    none_count = sum(1 for p in head_params if p.grad is None)
                    nan_count = sum(1 for p in head_params if p.grad is not None and not torch.isfinite(p.grad).all())
                    finite_count = sum(1 for p in head_params if p.grad is not None and torch.isfinite(p.grad).all())
                    logger.warning(f"{name}: None={none_count}, NaN/Inf={nan_count}, Finite={finite_count} (of {len(head_params)} params)")
                    # Show grad norm if any grads exist
                    grads_exist = [p for p in head_params if p.grad is not None]
                    if grads_exist:
                        norm = grads_exist[0].grad.norm().item()
                        logger.warning(f"  First grad norm: {norm:.6f} ({'NaN!' if not torch.isfinite(torch.tensor(norm)).item() else 'finite'})")

                # Check if optimizer has these params
                opt_params = set()
                for pg in self.optimizer.param_groups:
                    for p in pg['params']:
                        opt_params.add(id(p))
                first_slot_param = list(base_network.slot_head.parameters())[0]
                logger.warning(f"slot_head param in optimizer: {id(first_slot_param) in opt_params}")

                logger.warning("=== END DEBUG PROBE ===")

            # Collect per-head gradient norms BEFORE clipping (P4-6)
            # Measures raw gradients to diagnose head dominance
            # PERF: Batch all norm computations, then single .tolist() at end
            with torch.inference_mode():
                # Use the BASE network to access head modules. torch.compile shares
                # parameters with the original module, so gradients are on the same
                # Parameter objects. Using base_network ensures consistency with how
                # the optimizer was created (which also uses base_network params when
                # weight_decay > 0).
                network = self.policy.network
                base_network = getattr(network, '_orig_mod', network)

                head_names = ["slot", "blueprint", "style", "tempo", "alpha_target",
                              "alpha_speed", "alpha_curve", "op", "value"]
                head_modules = [
                    base_network.slot_head, base_network.blueprint_head, base_network.style_head,
                    base_network.tempo_head, base_network.alpha_target_head, base_network.alpha_speed_head,
                    base_network.alpha_curve_head, base_network.op_head, base_network.value_head,
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
                        # BUG FIX: Use NaN to signal "no gradient data" instead of 0.0
                        # 0.0 would hide the bug (No Bug-Hiding Patterns rule)
                        # NaN signals missing data and will surface in telemetry
                        norm_t = torch.tensor(float("nan"), device=self.device)
                    head_norm_tensors.append(norm_t)

                # Single GPU→CPU sync: stack all norms, then .tolist()
                all_norms = torch.stack(head_norm_tensors).cpu().tolist()
                for head_name, grad_norm in zip(head_names, all_norms):
                    head_grad_norm_history[head_name].append(grad_norm)

                # Gradient CV: coefficient of variation = std/|mean| (per DRL expert)
                # Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy gradients
                # PERF: Compute on CPU using already-transferred all_norms (avoids extra sync)
                n = len(all_norms)
                if n > 1 and any(v > 0 for v in all_norms):
                    grad_mean = sum(all_norms) / n
                    grad_var = sum((x - grad_mean) ** 2 for x in all_norms) / (n - 1)
                    grad_std = grad_var ** 0.5
                    grad_cv = grad_std / max(abs(grad_mean), 1e-8)
                else:
                    grad_cv = 0.0
                metrics["gradient_cv"].append(grad_cv)

            # Capture pre-clip gradient norm (BUG FIX: was discarding this value)
            # The return value of clip_grad_norm_ is the total norm BEFORE clipping.
            # This is critical for detecting gradient explosion vs healthy gradients.
            pre_clip_norm = nn.utils.clip_grad_norm_(
                self.policy.network.parameters(), self.max_grad_norm
            )
            metrics["pre_clip_grad_norm"].append(float(pre_clip_norm))
            self.optimizer.step()

            # Track metrics
            # PERF: Batch all 10 logging metrics into single GPU→CPU transfer
            # BUG FIX: Use TRUE joint_ratio (product of all heads) computed at line 619
            # Previously this used per_head_ratios["op"] which hid divergence in other heads
            logging_tensors = torch.stack([
                policy_loss,
                value_loss,
                -entropy_loss,  # negate here to match original semantics
                joint_ratio.mean(),
                joint_ratio.max(),
                joint_ratio.min(),
                joint_ratio.std(),  # BUG FIX: ratio_std was missing, hidden by .get() default
                # Value function stats (single GPU sync with rest)
                values.mean(),
                values.std(),
                values.min(),
                values.max(),
            ]).cpu().tolist()
            # NOTE: logging_tensors is now a Python list[float]. Indexed access below
            # avoids 10 separate GPU→CPU syncs (one per .item() call).
            metrics["policy_loss"].append(logging_tensors[0])
            metrics["value_loss"].append(logging_tensors[1])
            metrics["entropy"].append(logging_tensors[2])
            metrics["ratio_mean"].append(logging_tensors[3])
            metrics["ratio_max"].append(logging_tensors[4])
            metrics["ratio_min"].append(logging_tensors[5])
            metrics["ratio_std"].append(logging_tensors[6])  # BUG FIX: was missing
            metrics["value_mean"].append(logging_tensors[7])
            metrics["value_std"].append(logging_tensors[8])
            metrics["value_min"].append(logging_tensors[9])
            metrics["value_max"].append(logging_tensors[10])
            # PERF: Reuse already-transferred ratio stats (indices 4,5) instead of
            # re-computing on GPU which would trigger 2 redundant syncs
            ratio_max_val = logging_tensors[4]
            ratio_min_val = logging_tensors[5]
            if (
                ratio_max_val > self.ratio_explosion_threshold
                or ratio_min_val < self.ratio_collapse_threshold
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
        # Add log prob extremes (NaN predictor)
        # Guard against no valid data (inf values indicate no updates occurred)
        # Use NaN (not 0.0) to signal "no data" - 0.0 means "probability=1" which is misleading
        if log_prob_min_across_epochs == float("inf"):
            log_prob_min_across_epochs = float("nan")
        if log_prob_max_across_epochs == float("-inf"):
            log_prob_max_across_epochs = float("nan")
        aggregated_result["log_prob_min"] = log_prob_min_across_epochs
        aggregated_result["log_prob_max"] = log_prob_max_across_epochs

        # Add per-head ratio max tracking (Policy V2 - multi-head ratio explosion detection)
        # Guard against no valid data (-inf values indicate no updates occurred)
        # Use 1.0 (neutral ratio) as default for missing data
        for key in HEAD_NAMES:
            ratio_key = f"head_{key}_ratio_max"
            max_val = head_ratio_max_across_epochs[key]
            aggregated_result[ratio_key] = max_val if max_val != float("-inf") else 1.0  # type: ignore[literal-required]
        aggregated_result["joint_ratio_max"] = (
            joint_ratio_max_across_epochs if joint_ratio_max_across_epochs != float("-inf") else 1.0
        )

        # Add CUDA memory metrics (collected once per update, not averaged)
        if cuda_memory_metrics:
            for k, v in cuda_memory_metrics.items():
                aggregated_result[k] = v  # type: ignore[literal-required]

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

        # === Extract state_dim ===
        # Required since checkpoint v1; base_net.state_dim excludes blueprint embeddings.
        if 'state_dim' not in architecture:
            raise RuntimeError(
                "Incompatible checkpoint: architecture.state_dim is required. "
                "Please retrain the model to create a compatible checkpoint."
            )
        state_dim = int(architecture['state_dim'])

        # === Extract compile_mode (default "off" for old checkpoints) ===
        compile_mode = config.get('compile_mode', 'off')

        # === Create PolicyBundle (uncompiled - compile AFTER loading weights) ===
        from esper.tamiyo.policy.factory import create_policy
        policy = create_policy(
            policy_type="lstm",
            slot_config=slot_config,
            state_dim=state_dim,
            lstm_hidden_dim=config['lstm_hidden_dim'],
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
]
