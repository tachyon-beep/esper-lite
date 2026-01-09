"""Simic PPO agent for seed lifecycle control.

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.amp as torch_amp
import torch.nn as nn

from .rollout_buffer import TamiyoRolloutBuffer
from .advantages import compute_per_head_advantages
from .ppo_metrics import PPOUpdateMetricsBuilder
from .ppo_update import PPOUpdateResult, compute_losses, compute_ratio_metrics
from .types import PPOUpdateMetrics
from esper.simic.telemetry import RatioExplosionDiagnostic
from esper.simic.telemetry.lstm_health import compute_lstm_health
from esper.simic.control import ValueNormalizer
from esper.leyline.value_metrics import (
    ValueFunctionMetricsDict,
    compute_value_function_metrics,
)
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
    DEFAULT_RATIO_COLLAPSE_THRESHOLD,
    DEFAULT_RATIO_EXPLOSION_THRESHOLD,
    DEFAULT_VALUE_CLIP,
    DEFAULT_VALUE_COEF,
    ENTROPY_FLOOR_PER_HEAD,
    ENTROPY_FLOOR_PENALTY_COEF,
    HEAD_NAMES,
    NUM_OPS,
)
from esper.leyline.slot_config import SlotConfig
import logging

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Checkpoint format version for forward compatibility
# Increment when checkpoint structure changes in backwards-incompatible ways
# Version 2: value_normalizer_state_dict and compile_mode are now required fields
CHECKPOINT_VERSION = 2

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

def _init_q_metrics_nan(device: torch.device | str) -> dict[str, list[torch.Tensor]]:
    """Initialize Q-value metrics with NaN when no valid states available."""
    nan = torch.tensor(float("nan"), device=device)
    op_q_values = torch.full((NUM_OPS,), float("nan"), device=device)
    op_valid_mask = torch.zeros(NUM_OPS, dtype=torch.bool, device=device)
    return {
        "op_q_values": [op_q_values],
        "op_valid_mask": [op_valid_mask],
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
        clip_value: bool = False,  # Disabled: return normalization provides stability
        # Separate clip range for value function (larger than policy clip_ratio)
        # Note: Research (Engstrom et al., 2020) suggests value clipping often hurts.
        # With return normalization (std-only), clipping conflicts with normalized scale.
        value_clip: float = DEFAULT_VALUE_CLIP,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        # P2 FIX: n_epochs parameter REMOVED - it was dead code (never used in update loop).
        # The actual epoch count is controlled by:
        #   - recurrent_n_epochs: INTERNAL epochs within a single update() call (default 1 for LSTM safety)
        #   - ppo_updates_per_batch: EXTERNAL updates in vectorized.py training loop
        # Standard PPO "epochs" are configured via ppo_updates_per_batch in train_ppo_vectorized().
        recurrent_n_epochs: int | None = None,  # Default 1 for recurrent (hidden state safety)
        batch_size: int = DEFAULT_BATCH_SIZE,
        target_kl: float | None = 0.015,
        weight_decay: float = 0.0,  # Applied to critic only (RL best practice)
        device: str = "cuda:0",
        chunk_length: int = DEFAULT_EPISODE_LENGTH,  # Must match max_epochs (from leyline)
        num_envs: int = DEFAULT_N_ENVS,  # For TamiyoRolloutBuffer
        max_steps_per_env: int = DEFAULT_EPISODE_LENGTH,  # For TamiyoRolloutBuffer (from leyline)
        compile_mode: str = "off",  # For checkpoint persistence (policy may already be compiled)
        # Per-head entropy floor penalty (prevents sparse head collapse)
        entropy_floor: dict[str, float] | None = None,
        entropy_floor_penalty_coef: dict[str, float] | None = None,
        total_train_steps: int | None = None,  # For late-training decay schedule
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
        # MED-01 fix: Use is not None - empty dict {} is falsy but valid
        self.entropy_coef_per_head = dict(ENTROPY_COEF_PER_HEAD)
        if entropy_coef_per_head is not None:
            self.entropy_coef_per_head.update(entropy_coef_per_head)
        # Per-head entropy floor penalty (prevents sparse head collapse)
        # Uses ENTROPY_FLOOR_PER_HEAD from leyline as defaults
        self.entropy_floor = entropy_floor if entropy_floor is not None else dict(ENTROPY_FLOOR_PER_HEAD)
        self.entropy_floor_penalty_coef = (
            entropy_floor_penalty_coef if entropy_floor_penalty_coef is not None
            else dict(ENTROPY_FLOOR_PENALTY_COEF)
        )
        self.total_train_steps = total_train_steps if total_train_steps is not None else 1_000_000
        self.value_coef = value_coef
        self.clip_value = clip_value
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.lstm_hidden_dim = policy.hidden_dim
        # P2 FIX: self.n_epochs removed - was dead code (see __init__ comment)
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

        # P1 BUG FIX: Value normalization for GAE consistency
        # The critic is trained on normalized returns (std~1), but GAE needs
        # denormalized values to compute δ = r + γV(s') - V(s) correctly.
        # ValueNormalizer tracks running stats and provides denormalize() for GAE.
        self.value_normalizer = ValueNormalizer(device=device)

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

    def _get_penalty_schedule(self, progress: float) -> float:
        """Schedule entropy floor penalty coefficient across training phases.

        DRL Expert recommendation: Early-training boost + late-training decay.

        Schedule:
        - 0-25% (early): 1.5x boost - establish diverse exploration habits
        - 25-75% (mid): 1.0x baseline
        - 75-100% (late): decay from 1.0x to 0.5x - allow natural convergence

        PyTorch Expert Note: Apply as scalar multiplier OUTSIDE compute_losses
        to maintain clean separation of concerns.

        Args:
            progress: Training progress [0, 1] (train_steps / total_train_steps)

        Returns:
            Schedule factor [0.5, 1.5]
        """
        if progress < 0.25:
            # Early training: 1.5x boost to establish exploration
            return 1.5
        elif progress < 0.75:
            # Mid training: baseline
            return 1.0
        else:
            # Late training: decay from 1.0 at 75% to 0.5 at 100%
            decay_progress = (progress - 0.75) / 0.25
            return 1.0 - 0.5 * decay_progress

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

    def _compute_value_function_metrics(self) -> ValueFunctionMetricsDict:
        """Compute value function metrics from buffer data.

        Called after compute_advantages_and_returns() to extract
        TELE-220 to TELE-228 metrics.
        """
        # Collect valid data from all environments
        all_td_errors = []
        all_values = []
        all_returns = []

        for env_id in range(self.buffer.num_envs):
            num_steps = self.buffer.step_counts[env_id]
            if num_steps > 0:
                all_td_errors.append(self.buffer.td_errors[env_id, :num_steps])
                all_values.append(self.buffer.values[env_id, :num_steps])
                all_returns.append(self.buffer.returns[env_id, :num_steps])

        if not all_td_errors:
            return {
                "v_return_correlation": 0.0,
                "td_error_mean": 0.0,
                "td_error_std": 0.0,
                "bellman_error": 0.0,
                "return_p10": 0.0,
                "return_p50": 0.0,
                "return_p90": 0.0,
                "return_variance": 0.0,
                "return_skewness": 0.0,
            }

        td_errors = torch.cat(all_td_errors)
        values = torch.cat(all_values)
        returns = torch.cat(all_returns)

        return compute_value_function_metrics(td_errors, values, returns)

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

        # P1 BUG FIX: Pass value_normalizer to GAE so it can denormalize critic outputs
        # The critic learns normalized values (std~1), but GAE needs raw scale values
        # to compute delta = r + γV' - V correctly. Without this, scale mismatch
        # corrupts advantages and breaks training.
        self.buffer.compute_advantages_and_returns(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            value_normalizer=self.value_normalizer,
        )
        pre_norm_adv_mean, pre_norm_adv_std, std_floored = self.buffer.normalize_advantages()

        # D5: Compute forced step ratio from buffer data
        # Forced steps are timesteps where agent had no choice (only WAIT valid)
        forced_actions = self.buffer.forced_actions
        total_timesteps = sum(self.buffer.step_counts)
        if total_timesteps > 0:
            forced_count = 0
            for env_id in range(self.buffer.num_envs):
                num_steps = self.buffer.step_counts[env_id]
                if num_steps > 0:
                    forced_count += int(forced_actions[env_id, :num_steps].sum().item())
            forced_step_ratio = forced_count / total_timesteps
            usable_actor_timesteps = total_timesteps - forced_count
        else:
            forced_step_ratio = 0.0
            usable_actor_timesteps = 0

        # Compute value function metrics (TELE-220 to TELE-228)
        value_func_metrics = self._compute_value_function_metrics()

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
        #
        # SCALE FIX: Values from buffer are on normalized scale (critic trained on normalized
        # returns), but returns are on raw scale. Denormalize values to match returns scale.
        valid_values = data["values"][valid_mask]
        valid_returns = data["returns"][valid_mask]
        raw_values = self.value_normalizer.denormalize(valid_values)
        var_returns = valid_returns.var()
        explained_variance: torch.Tensor
        if var_returns > 1e-8:
            ev_tensor = 1.0 - (valid_returns - raw_values).var() / var_returns
            explained_variance = ev_tensor
        else:
            explained_variance = torch.tensor(0.0, device=valid_returns.device)

        metrics: dict[str, Any] = defaultdict(list)
        metrics["explained_variance"] = [explained_variance]

        # Return statistics for diagnosing value loss scale
        return_mean = valid_returns.mean()
        return_std = valid_returns.std()
        if not torch.isfinite(torch.stack([return_mean, return_std])).all():
            raise RuntimeError(
                f"Non-finite returns detected: mean={return_mean}, std={return_std}. "
                "This is a hard bug: investigate reward/value/GAE plumbing."
            )
        metrics["return_mean"] = [return_mean]
        metrics["return_std"] = [return_std]

        # P1 BUG FIX: Use running value normalizer instead of batch std
        # 1. Update normalizer with current batch returns (tracks running distribution)
        # 2. Normalize returns using running std (consistent scale across batches)
        # This ensures GAE denormalization and critic training use the SAME scale.
        #
        # Old (broken): batch_std varies each update, GAE uses stale scale
        # New (fixed): running_std is consistent, GAE denormalizes with same scale
        self.value_normalizer.update(valid_returns)
        normalized_returns = self.value_normalizer.normalize(valid_returns).detach()
        value_target_scale = torch.tensor(self.value_normalizer.get_scale(), device=valid_returns.device)
        metrics["value_target_scale"] = [value_target_scale]

        # Pre-normalization advantage stats for diagnosing advantage collapse
        # If pre_norm_adv_std is tiny but pre_clip_grad_norm is huge, it indicates
        # normalization is amplifying noise. If std is healthy but grad is huge,
        # it's more likely raw return scale or value mismatch.
        metrics["pre_norm_advantage_mean"] = [
            torch.tensor(pre_norm_adv_mean, device=valid_returns.device)
        ]
        metrics["pre_norm_advantage_std"] = [
            torch.tensor(pre_norm_adv_std, device=valid_returns.device)
        ]

        # D4/D5: Slot saturation diagnostics
        metrics["forced_step_ratio"] = [
            torch.tensor(forced_step_ratio, device=valid_returns.device)
        ]
        metrics["usable_actor_timesteps"] = [
            torch.tensor(usable_actor_timesteps, device=valid_returns.device, dtype=torch.long)
        ]
        metrics["advantage_std_floored"] = [
            torch.tensor(std_floored, device=valid_returns.device, dtype=torch.bool)
        ]
        metrics["d5_pre_norm_advantage_std"] = [
            torch.tensor(pre_norm_adv_std, device=valid_returns.device)
        ]

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
            metrics["advantage_mean"] = [adv_mean]
            metrics["advantage_std"] = [adv_std]
            metrics["advantage_skewness"] = [adv_skewness]
            metrics["advantage_kurtosis"] = [adv_kurtosis]
            # Fraction of positive advantages (healthy: 40-60%)
            # Imbalanced ratios suggest reward design issues or easy/hard regions
            adv_positive_ratio = (valid_advantages_for_stats > 0).float().mean()
            metrics["advantage_positive_ratio"] = [adv_positive_ratio]
        else:
            # No valid advantages - use NaN to signal "no data" (not "balanced" or "zero")
            nan_metric = torch.tensor(float("nan"), device=valid_mask.device)
            metrics["advantage_mean"] = [nan_metric]
            metrics["advantage_std"] = [nan_metric]
            metrics["advantage_skewness"] = [nan_metric]
            metrics["advantage_kurtosis"] = [nan_metric]
            metrics["advantage_positive_ratio"] = [nan_metric]

        # === Collect Op-Conditioned Q-Values (Policy V2) ===
        # Compute Q(s, op) vector using a representative state.
        # Use first valid state from batch to avoid bias from terminal states.
        if valid_mask.any():
            # Get first valid state
            first_valid_idx = valid_mask.nonzero(as_tuple=True)
            if len(first_valid_idx[0]) > 0:
                sample_row = first_valid_idx[0][0]
                sample_col = first_valid_idx[1][0]
                sample_obs = data["states"][sample_row, sample_col].unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
                sample_blueprints = data["blueprint_indices"][sample_row, sample_col].unsqueeze(0).unsqueeze(0)  # [1, 1, num_slots]
                op_mask = data["op_masks"][sample_row, sample_col].to(dtype=torch.bool)  # [num_ops]
                if op_mask.numel() != NUM_OPS:
                    raise ValueError(
                        f"Expected op mask length {NUM_OPS}, got {op_mask.numel()}."
                    )
                if not op_mask.any():
                    raise ValueError("Op mask has no valid ops - state machine bug.")

                # Forward pass to get LSTM output
                with torch.no_grad():
                    forward_result = self.policy.network.forward(
                        state=sample_obs,
                        blueprint_indices=sample_blueprints,
                        hidden=None,  # Use initial hidden state for consistency
                    )
                    lstm_out = forward_result["lstm_out"]  # [1, 1, hidden_dim]

                # Compute Q(s, op) vector in LifecycleOp order (NUM_OPS indices).
                op_q_values = torch.empty(NUM_OPS, device=self.device)
                for op_idx in range(NUM_OPS):
                    op_tensor = torch.tensor([[op_idx]], dtype=torch.long, device=self.device)
                    q_val = self.policy.network._compute_value(lstm_out, op_tensor)
                    op_q_values[op_idx] = q_val.squeeze()

                # Compute Q-variance and Q-spread over valid ops only.
                valid_q_values = op_q_values[op_mask]
                if valid_q_values.numel() >= 2:
                    q_variance = valid_q_values.var()
                    q_spread = valid_q_values.max() - valid_q_values.min()
                else:
                    q_variance = torch.tensor(float("nan"), device=self.device)
                    q_spread = torch.tensor(float("nan"), device=self.device)

                metrics["op_q_values"] = [op_q_values]
                metrics["op_valid_mask"] = [op_mask]
                metrics["q_variance"] = [q_variance]
                metrics["q_spread"] = [q_spread]
            else:
                # No valid states - use NaN
                metrics.update(_init_q_metrics_nan(self.device))
        else:
            # No valid data - use NaN
            metrics.update(_init_q_metrics_nan(self.device))

        # Initialize per-head entropy tracking (P3-1)
        head_entropy_history: dict[str, list[torch.Tensor]] = {head: [] for head in HEAD_NAMES}

        # LSTM health tracking (TELE-340)
        lstm_health_history: dict[str, list[float | bool]] = defaultdict(list)
        # Initialize per-head gradient norm tracking (P4-6)
        head_grad_norm_history: dict[str, list[torch.Tensor]] = {
            head: [] for head in HEAD_NAMES + ("value",)
        }
        # Initialize log prob extremes tracking (NaN predictor)
        # Very negative log probs (<-50 warning, <-100 critical) predict numerical underflow
        # Use inf/-inf for proper min/max tracking (log probs are always <= 0)
        log_prob_min_across_epochs = torch.tensor(float("inf"), device=self.device)
        log_prob_max_across_epochs = torch.tensor(float("-inf"), device=self.device)

        # Initialize per-head ratio max tracking (Policy V2 - multi-head ratio explosion detection)
        # Track max ratio per head across all epochs for telemetry
        head_ratio_max_across_epochs: dict[str, torch.Tensor] = {
            head: torch.tensor(float("-inf"), device=self.device) for head in HEAD_NAMES
        }
        joint_ratio_max_across_epochs = torch.tensor(float("-inf"), device=self.device)

        # Per-head NaN/Inf tracking (for indicator lights)
        # OR across all epochs - once detected, stays detected for this update
        head_nan_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}
        head_inf_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}

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
            # B11-PT-01 FIX: Run evaluate_actions outside autocast context.
            # When PPO update runs under AMP autocast, the network forward pass
            # produces float16 logits and LSTM states. Even though MaskedCategorical
            # upcasts logits to float32 for the distribution math, this is not
            # sufficient - the full forward pass must run in float32 for stable
            # gradient computation. This is standard practice in RL with AMP:
            # run the policy evaluation in float32, keep backbone/feature extraction in AMP.
            # Device can be str ("cpu", "cuda:0") or torch.device - normalize to type string
            device_type = str(self.device).split(":")[0]
            with torch_amp.autocast(device_type=device_type, enabled=False):  # type: ignore[attr-defined]
                # Cast inputs to float32 to ensure entire forward pass is float32
                # NOTE: initial_hidden_h/c are detached tensors from rollout collection.
                # This is CORRECT for recurrent PPO:
                # 1. We use them as starting points for LSTM reconstruction
                # 2. The LSTM forward pass produces new, gradient-enabled hidden states
                # 3. BPTT happens within the reconstructed sequence, not through initial_hidden
                # See rollout_buffer.py lines 377-378 for detach() calls.
                hidden_h = data["initial_hidden_h"].float()
                hidden_c = data["initial_hidden_c"].float()
                result = self.policy.evaluate_actions(
                    data["states"].float(),
                    data["blueprint_indices"],
                    actions,
                    masks,
                    hidden=(hidden_h, hidden_c),
                )
            log_probs = result.log_prob
            values = result.value
            entropy = result.entropy

            # Track LSTM hidden state health (TELE-340)
            if result.hidden is not None:
                lstm_health = compute_lstm_health(result.hidden)
                if lstm_health is not None:
                    lstm_health_history["lstm_h_rms"].append(lstm_health.h_rms)
                    lstm_health_history["lstm_c_rms"].append(lstm_health.c_rms)
                    lstm_health_history["lstm_h_env_rms_mean"].append(lstm_health.h_env_rms_mean)
                    lstm_health_history["lstm_h_env_rms_max"].append(lstm_health.h_env_rms_max)
                    lstm_health_history["lstm_c_env_rms_mean"].append(lstm_health.c_env_rms_mean)
                    lstm_health_history["lstm_c_env_rms_max"].append(lstm_health.c_env_rms_max)
                    lstm_health_history["lstm_h_max"].append(lstm_health.h_max)
                    lstm_health_history["lstm_c_max"].append(lstm_health.c_max)
                    lstm_health_history["lstm_has_nan"].append(lstm_health.has_nan)
                    lstm_health_history["lstm_has_inf"].append(lstm_health.has_inf)

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

            # Check new log_probs - separate NaN from Inf per head
            # Fast path: only drill down when isfinite fails (preserves 0 syncs in happy path)
            for key in HEAD_NAMES:
                lp = log_probs[key]
                if not torch.isfinite(lp).all():
                    # Slow path: distinguish NaN from Inf
                    if torch.isnan(lp).any():
                        head_nan_detected[key] = True
                        nonfinite_sources.append(f"log_probs[{key}]: NaN detected")
                    if torch.isinf(lp).any():
                        head_inf_detected[key] = True
                        nonfinite_sources.append(f"log_probs[{key}]: Inf detected")
                    nonfinite_found = True

            # Check old log_probs (stored as "{key}_log_probs" in data dict)
            for key in HEAD_NAMES:
                old_key = f"{key}_log_probs"
                old_lp = data[old_key][valid_mask]
                if not torch.isfinite(old_lp).all():
                    nonfinite_count = (~torch.isfinite(old_lp)).sum()
                    nonfinite_sources.append(f"old_log_probs[{key}]: {nonfinite_count} non-finite")
                    nonfinite_found = True

            # Check values
            if not torch.isfinite(values).all():
                nonfinite_count = (~torch.isfinite(values)).sum()
                nonfinite_sources.append(f"values: {nonfinite_count} non-finite")
                nonfinite_found = True

            if nonfinite_found:
                logger.warning(
                    f"FINITENESS GATE FAILED at epoch {epoch_i}: {', '.join(nonfinite_sources)}. "
                    "Skipping optimizer step to prevent NaN propagation. "
                    "Likely cause: mask mismatch between rollout and update time."
                )
                # RD-01 fix: metrics is defaultdict(list), no need for setdefault
                metrics["finiteness_gate_failures"].append({
                    "epoch": epoch_i,
                    "sources": nonfinite_sources,
                })
                continue  # Skip this epoch's update, try next epoch

            # Track log prob extremes across all heads (NaN predictor)
            # Use HEAD_NAMES for consistent ordering
            all_log_probs = torch.cat([log_probs[k] for k in HEAD_NAMES])
            if all_log_probs.numel() > 0:
                epoch_log_prob_min = all_log_probs.min()
                epoch_log_prob_max = all_log_probs.max()
                log_prob_min_across_epochs = torch.minimum(
                    log_prob_min_across_epochs, epoch_log_prob_min
                )
                log_prob_max_across_epochs = torch.maximum(
                    log_prob_max_across_epochs, epoch_log_prob_max
                )

            # Track per-head entropy (P3-1)
            # PERF: Batch all 8 head entropies into single GPU→CPU transfer
            head_entropy_values = torch.stack([entropy[key].mean() for key in HEAD_NAMES])
            for idx, key in enumerate(HEAD_NAMES):
                head_entropy_history[key].append(head_entropy_values[idx])

            valid_advantages = data["advantages"][valid_mask]
            valid_returns = data["returns"][valid_mask]

            # D1: Extract forced mask for loss weighting
            # Forced steps have no agency (only WAIT valid) - exclude from actor loss
            forced_mask = data["forced_actions"][valid_mask]

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
            batch_timesteps = valid_mask.sum().float().clamp(min=1)
            ratio_metrics = compute_ratio_metrics(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                head_masks=head_masks,
                clip_ratio=self.clip_ratio,
                target_kl=self.target_kl,
                head_names=HEAD_NAMES,
                total_timesteps=batch_timesteps,
            )
            update_result = PPOUpdateResult(ratio_metrics=ratio_metrics, loss_metrics=None)
            for key, max_val in update_result.ratio_metrics.per_head_ratio_max.items():
                head_ratio_max_across_epochs[key] = torch.maximum(
                    head_ratio_max_across_epochs[key], max_val
                )
            joint_ratio_max_across_epochs = torch.maximum(
                joint_ratio_max_across_epochs, update_result.ratio_metrics.joint_ratio_max
            )

            joint_ratio = update_result.ratio_metrics.joint_ratio
            metrics["approx_kl"].append(update_result.ratio_metrics.approx_kl)
            metrics["clip_fraction"].append(update_result.ratio_metrics.clip_fraction)
            metrics["clip_fraction_positive"].append(update_result.ratio_metrics.clip_fraction_positive)
            metrics["clip_fraction_negative"].append(update_result.ratio_metrics.clip_fraction_negative)

            if update_result.ratio_metrics.early_stop:
                early_stopped = True
                metrics["early_stop_epoch"] = [torch.tensor(epoch_i, device=self.device)]
                ratio_stats = update_result.ratio_metrics.ratio_stats
                metrics["ratio_mean"].append(ratio_stats[0])
                metrics["ratio_max"].append(ratio_stats[1])
                metrics["ratio_min"].append(ratio_stats[2])
                metrics["ratio_std"].append(ratio_stats[3])
                break  # Skip loss computation, backward, and optimizer step

            # Compute policy loss per head and sum
            # Use masked mean to avoid bias from averaging zeros with real values
            valid_old_values = data["values"][valid_mask]
            entropy_coef = self.get_entropy_coef()

            # Compute training progress for penalty schedule
            # Schedule: 1.5x boost (0-25%), 1.0x baseline (25-75%), decay to 0.5x (75-100%)
            # This prevents early entropy collapse while allowing late-training convergence
            progress = self.train_steps / self.total_train_steps
            schedule_factor = self._get_penalty_schedule(progress)

            # Apply schedule to ALL per-head coefficients uniformly
            scheduled_coef = {
                head: coef * schedule_factor
                for head, coef in self.entropy_floor_penalty_coef.items()
            }

            losses = compute_losses(
                per_head_ratios=ratio_metrics.per_head_ratios,
                per_head_advantages=per_head_advantages,
                head_masks=head_masks,
                forced_mask=forced_mask,  # D1: Exclude forced steps from actor loss
                values=values,
                normalized_returns=normalized_returns,
                old_values=valid_old_values,
                entropy=entropy,
                entropy_coef_per_head=self.entropy_coef_per_head,
                entropy_coef=entropy_coef,
                clip_ratio=self.clip_ratio,
                clip_value=self.clip_value,
                value_clip=self.value_clip,
                value_coef=self.value_coef,
                head_names=HEAD_NAMES,
                entropy_floor=self.entropy_floor,
                entropy_floor_penalty_coef=scheduled_coef,  # Scheduled coefficients
            )
            update_result = PPOUpdateResult(ratio_metrics=ratio_metrics, loss_metrics=losses)
            loss_metrics = update_result.loss_metrics
            assert loss_metrics is not None
            loss = loss_metrics.total_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]

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

                all_norms = torch.stack(head_norm_tensors)
                for head_name, grad_norm in zip(head_names, all_norms):
                    head_grad_norm_history[head_name].append(grad_norm)

                # Gradient CV: coefficient of variation = std/|mean| (per DRL expert)
                # Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy gradients
                n = all_norms.numel()
                if n > 1 and torch.any(all_norms > 0):
                    grad_mean = all_norms.mean()
                    grad_var = ((all_norms - grad_mean) ** 2).sum() / (n - 1)
                    grad_std = torch.sqrt(grad_var)
                    grad_cv = grad_std / torch.clamp(grad_mean.abs(), min=1e-8)
                else:
                    grad_cv = torch.tensor(0.0, device=all_norms.device)
                metrics["gradient_cv"].append(grad_cv)

            # Capture pre-clip gradient norm (BUG FIX: was discarding this value)
            # The return value of clip_grad_norm_ is the total norm BEFORE clipping.
            # This is critical for detecting gradient explosion vs healthy gradients.
            pre_clip_norm = nn.utils.clip_grad_norm_(
                self.policy.network.parameters(), self.max_grad_norm
            )
            metrics["pre_clip_grad_norm"].append(pre_clip_norm)
            self.optimizer.step()

            # Track metrics
            # PERF: Batch all logging metrics into single GPU→CPU transfer
            # BUG FIX: Use TRUE joint_ratio (product of all heads) computed at line 619
            # Previously this used per_head_ratios["op"] which hid divergence in other heads
            logging_tensors = torch.stack([
                losses.policy_loss,
                losses.value_loss,
                -losses.entropy_loss,  # negate here to match original semantics
                joint_ratio.mean(),
                joint_ratio.max(),
                joint_ratio.min(),
                joint_ratio.std(),  # BUG FIX: ratio_std was missing, hidden by .get() default
                # Value function stats (single GPU sync with rest)
                values.mean(),
                values.std(),
                values.min(),
                values.max(),
                losses.entropy_floor_penalty,  # DRL Expert: Track for calibration debugging
            ])
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
            metrics["entropy_floor_penalty"].append(logging_tensors[11])
            # PERF: Reuse already-transferred ratio stats (indices 4,5) instead of
            # re-computing on GPU which would trigger 2 redundant syncs
            ratio_max_val = logging_tensors[4]
            ratio_min_val = logging_tensors[5]
            ratio_exploded = (
                (ratio_max_val > self.ratio_explosion_threshold)
                | (ratio_min_val < self.ratio_collapse_threshold)
            )
            if ratio_exploded:
                diag = RatioExplosionDiagnostic.from_batch(
                    ratio=joint_ratio.flatten(),
                    old_log_probs=old_log_probs["op"].flatten(),
                    new_log_probs=log_probs["op"].flatten(),
                    actions=valid_op_actions.flatten(),
                    max_threshold=self.ratio_explosion_threshold,
                    min_threshold=self.ratio_collapse_threshold,
                )
                # RD-01 fix: metrics is defaultdict(list), no need for setdefault
                metrics["ratio_diagnostic"].append(diag.to_dict())

        # NOTE: train_steps increment is deferred until after finiteness gate check.
        # If all epochs fail finiteness checks, we should NOT advance train_steps
        # because entropy annealing and other schedules depend on actual updates.

        if clear_buffer:
            self.buffer.reset()

        # Aggregate into typed result dict (metrics aggregation owns list->scalar logic)
        finiteness_failures = metrics["finiteness_gate_failures"]
        epochs_completed = len(metrics["ratio_max"])

        builder = PPOUpdateMetricsBuilder(
            metrics=metrics,
            finiteness_failures=finiteness_failures,
            epochs_completed=epochs_completed,
            head_entropies=head_entropy_history,
            head_grad_norms=head_grad_norm_history,
            head_nan_detected=head_nan_detected,
            head_inf_detected=head_inf_detected,
            lstm_health_history=lstm_health_history,
            log_prob_min_across_epochs=log_prob_min_across_epochs,
            log_prob_max_across_epochs=log_prob_max_across_epochs,
            head_ratio_max_across_epochs=head_ratio_max_across_epochs,
            joint_ratio_max_across_epochs=joint_ratio_max_across_epochs,
            value_func_metrics=value_func_metrics,
            cuda_memory_metrics=cuda_memory_metrics,
            head_names=HEAD_NAMES,
        )
        result = builder.finalize()

        if result.update_performed:
            # Only increment train_steps when an actual update occurred.
            # This ensures entropy annealing and other schedules track real updates,
            # not skipped finiteness-gate failures.
            self.train_steps += 1

        return result.metrics

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
            'value_normalizer_state_dict': self.value_normalizer.state_dict(),
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
                'entropy_floor': self.entropy_floor,
                'entropy_floor_penalty_coef': self.entropy_floor_penalty_coef,
                'total_train_steps': self.total_train_steps,
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
                # P2 FIX: n_epochs removed from checkpoint - was dead code
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
    def load(
        cls,
        path: str | Path,
        device: str = "cuda:0",
        compile_mode: str | None = None,  # P3 FIX: Runtime override for debugging
    ) -> "PPOAgent":
        """Load agent from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model onto
            compile_mode: Override checkpoint's torch.compile mode. Use "off" for debugging
                         or to run on incompatible hardware. If None, uses checkpoint's
                         saved mode. This is the recommended pattern for torch.compile
                         portability - compiled graphs are hardware-specific and may
                         not transfer across GPU architectures.

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

        # Extract value_normalizer state (required since CHECKPOINT_VERSION 2)
        try:
            value_normalizer_state = checkpoint['value_normalizer_state_dict']
        except KeyError as e:
            raise RuntimeError(
                f"Incompatible checkpoint format: missing required field {e}. "
                f"This checkpoint was saved with an older version (before v2). "
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

        # === Extract compile_mode (required since CHECKPOINT_VERSION 2) ===
        # P3 FIX: Allow runtime override of compile_mode for debugging/portability
        # torch.compile graphs are hardware-specific - compiled on A100 may fail on H100
        if 'compile_mode' not in config:
            raise RuntimeError(
                "Incompatible checkpoint: config.compile_mode is required. "
                "Please retrain the model to create a compatible checkpoint."
            )
        checkpoint_compile_mode = config['compile_mode']
        effective_compile_mode = compile_mode if compile_mode is not None else checkpoint_compile_mode

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
        # Remove config params that are now part of PolicyBundle or removed
        # P2 FIX: Also filter 'n_epochs' - removed dead parameter (old checkpoints may have it)
        agent_config = {k: v for k, v in config.items()
                       if k not in ('lstm_hidden_dim', 'n_epochs')}
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

        # P1 FIX: Restore value_normalizer state if present
        # This ensures consistent normalization scale across training resumption
        if value_normalizer_state is not None:
            agent.value_normalizer.load_state_dict(value_normalizer_state)

        # === Apply torch.compile AFTER loading weights ===
        # Critical: Compile must happen after state_dict to ensure graph traces
        # the actual loaded weights, not random initialization.
        # P3 FIX: Use effective_compile_mode (may be overridden by parameter)
        if effective_compile_mode != "off":
            agent.policy.compile(mode=effective_compile_mode, dynamic=True)

        # Update agent's stored compile_mode to reflect what we actually used
        agent.compile_mode = effective_compile_mode

        return agent


__all__ = [
    "PPOAgent",
]
