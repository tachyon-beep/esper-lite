"""Simic PPO agent for seed lifecycle control.

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .rollout_buffer import TamiyoRolloutBuffer
from .advantages import compute_per_head_advantages
from .ppo_metrics import PPOUpdateMetricsBuilder
from .ppo_update import PPOUpdateResult, compute_contribution_aux_loss, compute_losses, compute_ratio_metrics
from .types import PPOUpdateMetrics
from esper.simic.telemetry import RatioExplosionDiagnostic
from esper.simic.telemetry.lstm_health import compute_lstm_health
from esper.simic.control import ValueNormalizer
from esper.leyline.value_metrics import (
    ValueFunctionMetricsDict,
    compute_floored_aux_explained_variance,
    compute_floored_explained_variance,
    compute_value_function_metrics,
)
from esper.leyline import (
    PolicyBundle,
    compute_availability_masks,
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
    PROBABILITY_FLOOR_PER_HEAD,
    VALUE_HEAD_SCHEMA_VERSION,
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
        # Value coefficient warmup: start low, ramp up to value_coef over warmup_steps.
        # This prevents critic collapse when early returns have low variance (before
        # the policy discovers fossilization). The critic learns "be patient" by having
        # less influence early, then gradually taking over as variance appears.
        value_coef_start: float | None = None,  # Default: 0.1 * value_coef
        value_warmup_steps: int = 0,  # 0 = no warmup, use fixed value_coef
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
        # Per-head probability floor (HARD constraint - guarantees gradient flow)
        # Unlike entropy penalties, this clamps action probabilities to a minimum,
        # ensuring gradients can always flow even when entropy would collapse.
        probability_floor: dict[str, float] | None = None,
        total_train_steps: int | None = None,  # For late-training decay schedule
        # Auxiliary contribution supervision (Expert-reviewed defaults)
        aux_contribution_coef: float = 0.05,  # DRL Expert: reduced from 0.1
        aux_warmup_steps: int = 1000,  # DRL + PyTorch Expert: ramp from 0 → full
        # EV-telemetry-robustness: floor the EV denominator (valid_returns.var()) on the RAW
        # return scale. Step-0 calibration: healthy-run return std is 7-13 -> var_returns ~49-169
        # (the normal operating band). A floor of 1.0 (std 1.0) excludes only batches whose return
        # spread is <~1/7th of typical -- the pathological low-variance tail that manufactures a
        # -8 EV outlier -- while leaving every normal batch's EV numerically unchanged (the clamp
        # is a no-op above the floor). Config-exposed so a relative/EMA floor can be swapped in.
        ev_return_variance_floor: float = 1.0,
        # EV-telemetry-robustness (SLICE C): aux (contribution-target) EV floor. The
        # contribution-target scale is NOT calibratable from persisted telemetry
        # (contribution_targets are not serialized), so the aux floor is DATA-RELATIVE,
        # recomputed each update: floor = max(floor_min, floor_fraction * Var0(target)).
        # The 0.05 fraction mirrors the main EV floor being ~3-5% of the healthy
        # var_returns median; floor_min is the old bare clamp(min=0.01) value. DISTINCT
        # from ev_return_variance_floor (different scale). DIAGNOSTIC-ONLY: feeds the
        # aux_ev_low_return_variance flag + aux value_nrmse denominator; NEVER a gate trigger.
        aux_ev_return_variance_floor_fraction: float = 0.05,
        aux_ev_return_variance_floor_min: float = 0.01,
        aux_stop_gradient: bool = True,  # DRL Expert: prevent representation collapse
        contribution_loss_clip: float = 10.0,  # Clip targets to prevent outliers
        enable_contribution_aux: bool = True,  # Can disable for ablation
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
        # RECURRENT MULTI-EPOCH INVARIANT (anchored reference pass, full-recompute TBPTT):
        # recurrent_n_epochs is the number of INTERNAL PPO epochs per update() call. Multi-
        # epoch recurrent PPO is mathematically EXACT under this design — there is no hidden-
        # state staleness, because:
        #   - Every scored forward in the epoch loop is a FULL-RECOMPUTE TBPTT unroll of each
        #     buffer row as one contiguous LSTM sequence (chunk_length) from the rollout's
        #     timestep-0 hidden state. Hidden trajectories are recomputed every epoch from the
        #     CURRENT weights — nothing is reused across epochs, so nothing goes stale.
        #   - Advantages/returns are computed ONCE by the rollout buffer's GAE (at episode-end)
        #     and read here as fixed data; the value normalizer's update()/normalize() runs ONCE
        #     in the pre-loop. Both outputs (advantages, normalized return targets) are fixed for
        #     the whole update — never recomputed per epoch.
        #   - The ONLY live per-epoch baseline is old_log_probs for the PPO importance ratio,
        #     and it is ANCHORED at theta0 via the no_grad reference pass (PR1): ref_log_probs
        #     are computed once from the pre-update weights, so ratio = exp(scored - ref) is
        #     identically 1.0 at epoch 0 and drifts only with the genuine policy update.
        #   - clip_value STAYS False under K>1 (ENFORCED by a ValueError in __init__ below).
        #     Value clipping would compare against rollout values that are NOT anchored at theta0;
        #     re-enabling it under multi-epoch needs an anchored ref_values pass (a separate task).
        #   - early_stop_epoch (metrics, set to epoch_i) counts epochs that RAN, not wall
        #     epochs: the finiteness guard in the epoch loop can `continue` past a bad epoch,
        #     desyncing the loop index from the count of executed updates.
        # The per-step hidden buffer (rollout_buffer.py pre_step_hiddens, hidden_h/hidden_c)
        # is TELEMETRY-ONLY — it conditions the Q(s, op) diagnostic forward (see below) and
        # is NOT used by the TBPTT loss path (which always unrolls from timestep-0). Do not
        # delete it.
        self.recurrent_n_epochs = recurrent_n_epochs if recurrent_n_epochs is not None else 1
        if self.recurrent_n_epochs < 1:
            raise ValueError(
                f"recurrent_n_epochs must be >= 1, got {self.recurrent_n_epochs}. "
                "K=1 is single-epoch PPO; K>1 is the internal multi-epoch path."
            )
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
        # Per-head probability floor (HARD constraint - guarantees gradient flow)
        # Uses PROBABILITY_FLOOR_PER_HEAD from leyline as defaults
        self.probability_floor = (
            probability_floor if probability_floor is not None
            else dict(PROBABILITY_FLOOR_PER_HEAD)
        )
        if total_train_steps is not None and total_train_steps <= 0:
            raise ValueError(f"total_train_steps must be positive, got {total_train_steps}")
        self.total_train_steps = total_train_steps if total_train_steps is not None else 1_000_000
        self.value_coef = value_coef
        # Value warmup: start at 10% of target by default, ramp up over warmup_steps
        self.value_coef_start = value_coef_start if value_coef_start is not None else 0.1 * value_coef
        self.value_warmup_steps = value_warmup_steps
        self.clip_value = clip_value
        if self.recurrent_n_epochs > 1 and clip_value:
            raise ValueError(
                "clip_value=True is incompatible with recurrent_n_epochs > 1: value clipping "
                "anchors on rollout old_values that are NOT recomputed at theta0, so under "
                "multi-epoch recurrent PPO it would clip against a stale reference. Re-enabling "
                "value clipping under K>1 requires an anchored ref_values pass (a separate task). "
                "Keep clip_value=False for recurrent multi-epoch."
            )
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.lstm_hidden_dim = policy.hidden_dim
        # P2 FIX: self.n_epochs removed - was dead code (see __init__ comment)
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.weight_decay = weight_decay
        self.device = device

        # Auxiliary contribution supervision config
        self.aux_contribution_coef = aux_contribution_coef
        self.aux_warmup_steps = aux_warmup_steps
        self.aux_stop_gradient = aux_stop_gradient
        self.contribution_loss_clip = contribution_loss_clip
        self.enable_contribution_aux = enable_contribution_aux
        self._aux_training_step = 0  # Track training steps for warmup

        # EV-telemetry-robustness floor (raw return-variance scale; see ctor docstring above).
        # ev_var_floor_std is stored for diagnostics/docstring parity only -- the W7 helper
        # recomputes its own std floor (math.sqrt(floor)) internally and does NOT read this.
        self.ev_return_variance_floor = ev_return_variance_floor
        self.ev_var_floor_std = math.sqrt(self.ev_return_variance_floor)

        # EV-telemetry-robustness (SLICE C): aux floor params (contribution-target scale;
        # see ctor docstring). Data-relative, recomputed each update in the aux EV block.
        self.aux_ev_return_variance_floor_fraction = aux_ev_return_variance_floor_fraction
        self.aux_ev_return_variance_floor_min = aux_ev_return_variance_floor_min

        # BPTT INVARIANT (recurrent-PPO correctness):
        # The PPO update unrolls each buffer row as ONE contiguous LSTM sequence
        # of length chunk_length, starting from the rollout's timestep-0 hidden
        # state, and does NOT reset hidden state at done boundaries inside the
        # sequence (see update() -> evaluate_actions). Rollout collection, by
        # contrast, zeroes hidden state across episode boundaries. Episodes
        # terminate only at done == (epoch == max_epochs), and chunk_length is
        # pinned to max_epochs == episode_length. If a single buffer row could
        # hold more than one episode (max_steps_per_env > chunk_length), the
        # recurrent gradient would leak hidden state across the done boundary in
        # evaluate_actions but not in rollout, producing a SILENT recurrent-
        # gradient bias. Guard the invariant loudly at construction time.
        if max_steps_per_env > chunk_length:
            raise ValueError(
                "BPTT INVARIANT VIOLATED: max_steps_per_env "
                f"({max_steps_per_env}) > chunk_length ({chunk_length}). "
                "chunk_length == episode_length == max_epochs is the LSTM "
                "sequence length the PPO update unrolls per buffer row. A row "
                "longer than chunk_length would pack multiple episodes into one "
                "BPTT sequence; evaluate_actions does not reset hidden state at "
                "done boundaries, so recurrent gradients would leak across "
                "episode boundaries (present in evaluate_actions but absent in "
                "rollout collection) -> silent recurrent-gradient bias. Require "
                "max_steps_per_env <= chunk_length."
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
            # P0-1: BOTH value heads are critics. state_value_head is the op-INDEPENDENT
            # PPO baseline V(s); q_head is the op-conditioned telemetry/aux head. Both
            # regress on returns, so weight_decay applies to both (and both MUST be in
            # the optimizer or a head silently never trains).
            critic_params = (
                list(base_net.state_value_head.parameters()) +
                list(base_net.q_head.parameters())
            )
            shared_params = (
                list(base_net.feature_net.parameters()) +
                list(base_net.lstm.parameters()) +
                list(base_net.lstm_ln.parameters()) +
                list(base_net.blueprint_embedding.parameters()) +  # Phase 4: blueprint embeddings
                list(base_net.contribution_predictor.parameters())  # Phase 3: auxiliary contribution predictor
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

    def get_value_coef(self) -> float:
        """Get current value coefficient with warmup.

        Value warmup prevents critic collapse when early returns have low variance.
        Before the policy discovers fossilization, all returns look similar and the
        critic learns a constant (value collapse). By starting with low value_coef,
        we tell the critic "be patient, don't overfit to early uniform returns."

        Returns:
            Value coefficient (ramped up if warmup enabled)
        """
        if self.value_warmup_steps == 0:
            # No warmup - use fixed coefficient
            return self.value_coef

        # With warmup - ramp from value_coef_start to value_coef
        progress = min(1.0, self.train_steps / self.value_warmup_steps)
        return self.value_coef_start + progress * (self.value_coef - self.value_coef_start)

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
        progress = min(1.0, max(0.0, progress))
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
            # P1-VEC: one GPU->CPU sync instead of num_envs. Build the per-env validity
            # mask (step t valid iff t < step_counts[env]) on device, AND it with the
            # forced flags, and reduce once. Equivalent to summing forced_actions[env, :n].
            max_steps = forced_actions.shape[1]
            step_counts_t = torch.as_tensor(
                self.buffer.step_counts, device=forced_actions.device, dtype=torch.long
            )
            step_valid = (
                torch.arange(max_steps, device=forced_actions.device)[None, :]
                < step_counts_t[:, None]
            )
            forced_count = int((forced_actions & step_valid).sum().item())  # 1 sync, was num_envs
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
        # W7 helper seam: variance-floored EV + floor-stabilized value_nrmse companion + flag.
        # Single path (no dual 1e-8/0.0 branch). Degenerate (numel<=1) masks are NOT handled
        # here -- they produce NaN var()/std() and fall through to the non-finite return-stat
        # raise below (B3 hard bug), never a NaN-by-convention EV.
        (
            explained_variance,
            value_nrmse,
            ev_low_return_variance,
            ev_return_variance,
        ) = compute_floored_explained_variance(
            raw_values, valid_returns, self.ev_return_variance_floor
        )

        metrics: dict[str, Any] = defaultdict(list)
        metrics["explained_variance"] = [explained_variance]
        metrics["ev_return_variance"] = [ev_return_variance]
        metrics["value_nrmse"] = [value_nrmse]
        metrics["ev_low_return_variance"] = [ev_low_return_variance]
        metrics["ev_low_return_variance_count"] = [1 if ev_low_return_variance else 0]

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

                # TPD-002: Recover the ROLLOUT row's stored LSTM hidden state.
                # The buffer records, for every transition, the hidden state that was
                # the recurrent INPUT to the network when that observation was consumed
                # (pre_step_hiddens during rollout collection). For the recurrent policy,
                # Q(s, op) telemetry is only meaningful if it conditions on that same
                # recurrent context. Passing hidden=None would force the network's
                # initial (zero) hidden state and produce a fake, context-free diagnostic.
                #
                # Stored shape: data["hidden_h"]/["hidden_c"] are
                # [num_envs, max_steps, lstm_layers, hidden_dim]. The network's forward()
                # expects (h, c) each [lstm_layers, batch, hidden_dim]. We slice the chosen
                # (env, step) row to [lstm_layers, hidden_dim] and insert a batch dim of 1.
                #
                # Every row that exists in the buffer carries a real stored hidden state
                # (the rollout always supplies pre_step_hiddens; the first step of an
                # episode stores the network's genuine initial hidden, which is a real
                # zero-state, NOT a missing value). There is therefore no "no hidden"
                # case to silently substitute None for — the contract is that a valid row
                # always has a real recurrent state, and we pass it through unconditionally.
                sample_hidden_h = (
                    data["hidden_h"][sample_row, sample_col]  # [lstm_layers, hidden_dim]
                    .unsqueeze(1)  # [lstm_layers, 1, hidden_dim]
                    .to(device=self.device, dtype=sample_obs.dtype)
                    .contiguous()
                )
                sample_hidden_c = (
                    data["hidden_c"][sample_row, sample_col]  # [lstm_layers, hidden_dim]
                    .unsqueeze(1)  # [lstm_layers, 1, hidden_dim]
                    .to(device=self.device, dtype=sample_obs.dtype)
                    .contiguous()
                )

                # Forward pass to get LSTM output, conditioned on the rollout row's
                # actual recurrent state (not the initial hidden state).
                #
                # AMP-SAFETY (CRITICAL): update() may run under an outer autocast context
                # (policy_amp_context -> autocast(bf16) for BF16 PPO; the AMP test legs use
                # autocast(fp16) directly). This no_grad telemetry forward touches EVERY head
                # Linear (slot/blueprint/.../op AND state_value_head AND q_head -- forward()
                # computes both V(s) and Q(s, op)) BEFORE the gradient-carrying
                # evaluate_actions forward below. Under autocast, the FIRST forward through a
                # Linear populates autocast's per-parameter cast-weight CACHE; when that first
                # touch happens inside no_grad, the cached low-precision weight carries NO
                # autograd linkage. evaluate_actions then reuses the cached graph-less cast,
                # so autograd never connects the head logits back to the head nn.Parameters
                # and the policy/value heads receive None gradients (Sanctum then shows NaN
                # head grad norms; in production the policy silently stops learning). Running
                # this pure-FP32, no_grad diagnostic with autocast DISABLED keeps it out of
                # the training-dtype cast cache, so the real forward establishes the cache
                # under grad and the head gradients flow. autocast_cache parity is restored.
                autocast_device_type = torch.device(self.device).type
                with torch.autocast(device_type=autocast_device_type, enabled=False), torch.no_grad():
                    forward_result = self.policy.network.forward(
                        state=sample_obs,
                        blueprint_indices=sample_blueprints,
                        hidden=(sample_hidden_h, sample_hidden_c),
                    )
                    lstm_out = forward_result["lstm_out"]  # [1, 1, hidden_dim]

                    # Compute Q(s, op) vector in LifecycleOp order (NUM_OPS indices).
                    # P1-QLOOP: one batched _compute_q call over all ops instead of NUM_OPS
                    # serial launches. lstm_out is [1, 1, hidden]; broadcast over the op axis.
                    # Kept inside the autocast(enabled=False) region so the q_head Linear
                    # is also cached graph-free (the q_head is part of the same forward()
                    # above and trains via the detached Q-aux loss below). Note _compute_q
                    # detaches lstm_out internally; harmless here under no_grad.
                    op_indices = torch.arange(
                        NUM_OPS, device=self.device, dtype=torch.long
                    ).reshape(NUM_OPS, 1)
                    lstm_out_rep = lstm_out.expand(NUM_OPS, 1, -1).contiguous()  # [NUM_OPS, 1, hidden]
                    op_q_values = self.policy.network._compute_q(
                        lstm_out_rep, op_indices
                    ).reshape(NUM_OPS)

                # Compute Q-variance and Q-spread over valid ops only.
                valid_q_values = op_q_values[op_mask]
                if valid_q_values.numel() >= 2:
                    q_variance = valid_q_values.var()
                    q_spread = valid_q_values.max() - valid_q_values.min()
                elif valid_q_values.numel() == 1:
                    q_variance = torch.zeros((), device=self.device, dtype=valid_q_values.dtype)
                    q_spread = torch.zeros((), device=self.device, dtype=valid_q_values.dtype)
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
        # Initialize conditional entropy tracking (P3-2: entropy only when head is causally relevant)
        # This is the true exploration signal for sparse heads like blueprint/tempo
        conditional_head_entropy_history: dict[str, list[torch.Tensor]] = {head: [] for head in HEAD_NAMES}
        head_learnable_fraction_history: dict[str, list[torch.Tensor]] = {head: [] for head in HEAD_NAMES}
        # P0-1: "value" tracks the op-INDEPENDENT V(s) baseline (state_value_head);
        # "q" tracks the op-conditioned telemetry/aux q_head. Both are surfaced so
        # neither value-like head's grad is dropped from telemetry.
        head_gradient_state_history: dict[str, list[str]] = {head: [] for head in HEAD_NAMES + ("value", "q")}

        # LSTM health tracking (TELE-340)
        lstm_health_history: dict[str, list[float | bool]] = defaultdict(list)
        # Initialize per-head gradient norm tracking (P4-6)
        head_grad_norm_history: dict[str, list[torch.Tensor]] = {
            head: [] for head in HEAD_NAMES + ("value", "q")
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

        # Per-head clip-fraction history (factored-PPO trust-region telemetry).
        # Each entry is the fraction of samples where that head's ratio left
        # [1-clip, 1+clip] in one epoch; the builder means them across epochs into
        # head_{name}_clip_fraction metrics. Companion to the joint clip_fraction.
        head_clip_fraction_history: dict[str, list[torch.Tensor]] = {
            head: [] for head in HEAD_NAMES
        }

        # Per-head NaN/Inf tracking (for indicator lights)
        # OR across all epochs - once detected, stays detected for this update
        head_nan_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}
        head_inf_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}

        # Anchored reference pass: frozen-theta0 baseline for the PPO importance ratio.
        #
        # PRECISION: the anchor runs under plain no_grad and INHERITS the caller's autocast
        # exactly as the epoch-0 scored forward does (BF16 under policy_amp_context at
        # vectorized.py:447; FP32 on the CPU/no-AMP path). Sharing the precision decision is
        # what makes the epoch-0 ratio = exp(scored - ref) identically 1.0 -- anchor and
        # epoch-0 are the SAME forward at theta_0, same dtype, differing only by no_grad+detach.
        #
        # AMP-SAFETY (CRITICAL -- see the discovery gate
        # test_epoch0_per_head_grad_norms_nonzero_under_amp): this no_grad anchor is the FIRST
        # forward through every head Linear under the caller's autocast. Under autocast, the
        # first touch of a Linear's weight populates autocast's per-parameter cast-weight CACHE;
        # because that first touch is inside no_grad, the cached BF16 weight carries NO autograd
        # linkage. If the cache were left intact, the grad-enabled epoch-0 forward would REUSE
        # those graph-less casts, severing the path from the head logits back to the head
        # nn.Parameters -- the head .grad comes back None and the policy silently stops learning
        # (the gate test sees every action head's grad_state == "missing"; only the value head,
        # which the surrogate loss does not route through these heads, survives). The fix is to
        # EVICT the poisoned casts immediately after the anchor: torch.clear_autocast_cache()
        # drops the no_grad entries so the epoch-0 forward re-casts each weight fresh UNDER GRAD,
        # restoring full autograd linkage. The cache holds nothing else at this point (the
        # telemetry forward at ~:777 ran with autocast disabled, writing nothing), so the clear
        # is local in effect. This keeps the anchor at the caller's exact precision -- so the
        # ratio stays 1.0 to the bit under AMP -- while leaving the cast cache un-poisoned.
        anchor_actions = {
            "slot": data["slot_actions"],
            "blueprint": data["blueprint_actions"],
            "style": data["style_actions"],
            "tempo": data["tempo_actions"],
            "alpha_target": data["alpha_target_actions"],
            "alpha_speed": data["alpha_speed_actions"],
            "alpha_curve": data["alpha_curve_actions"],
            "op": data["op_actions"],
        }
        anchor_masks = {
            "slot": data["slot_masks"],
            "blueprint": data["blueprint_masks"],
            "style": data["style_masks"],
            "tempo": data["tempo_masks"],
            "alpha_target": data["alpha_target_masks"],
            "alpha_speed": data["alpha_speed_masks"],
            "alpha_curve": data["alpha_curve_masks"],
            "op": data["op_masks"],
        }
        with torch.no_grad():
            anchor_result = self.policy.evaluate_actions(
                data["states"],
                data["blueprint_indices"],
                anchor_actions,
                anchor_masks,
                hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
                probability_floor=self.probability_floor,
                aux_stop_gradient=self.aux_stop_gradient,
            )
        # Evict the no_grad anchor's graph-less cast-weight cache entries so the epoch-0
        # grad forward re-casts each head/value weight fresh under autograd (see AMP-SAFETY
        # note above). No-op when the caller is not under autocast (FP32/CPU path).
        torch.clear_autocast_cache()
        ref_log_probs = {head: anchor_result.log_prob[head].detach() for head in HEAD_NAMES}

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
            # P1-BF16: evaluate_actions runs UNDER the BF16 autocast established by
            # _run_ppo_updates (vectorized.py). The LSTM + heads compute in BF16; the
            # log_softmax / log_prob / entropy / floor are upcast to FP32 at the masked-
            # logits seam (P1-EVAL), so ratios/KL stay precision-stable and SYMMETRIC with
            # the rollout's old_log_probs (which apply the identical FP32 seam). BF16 has
            # FP32 exponent range, so no GradScaler is required.
            # NOTE: initial_hidden_h/c are detached rollout tensors used as BPTT starting
            # points; BPTT happens within the reconstructed sequence, not through them
            # (see rollout_buffer.py detach() calls).
            result = self.policy.evaluate_actions(
                data["states"],
                data["blueprint_indices"],
                actions,
                masks,
                hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
                probability_floor=self.probability_floor,
                aux_stop_gradient=self.aux_stop_gradient,
            )
            log_probs = result.log_prob
            values = result.value  # V(s): op-INDEPENDENT PPO baseline (current epoch)
            # P0-1: Q-aux uses the CURRENT epoch's q_value (NOT the anchor's). Like the
            # value loss, the Q-aux regression target (normalized_returns) is fixed for
            # the whole update while the predictor moves each epoch -- there is no
            # anchoring requirement on value/q targets (only ref_log_probs is anchored).
            # P2-a: PPO-trained policies MUST populate q_value (P0-1 always builds a
            # q_head). The Q-aux regression and op_q telemetry depend on it; fail loud
            # at the boundary rather than crashing opaquely downstream (e.g. on the
            # q_values[valid_mask] index below).
            assert result.q_value is not None, (
                "PPO requires a q_value; the policy must populate EvalResult.q_value "
                "(P0-1: PPO-trained policies always expose an op-conditioned q_head)."
            )
            q_values = result.q_value  # Q(s, stored_op): detached telemetry/aux
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
            q_values = q_values[valid_mask]  # P0-1: filter Q-aux like the value path
            for key in entropy:
                entropy[key] = entropy[key][valid_mask]

            # FINITENESS GATE: Detect NaN/Inf early to identify source of numerical instability
            # This is NOT bug-hiding - it's fail-fast on corrupted probabilities.
            # Common causes: mask mismatch between rollout and update, degenerate distributions
            nonfinite_found = False
            nonfinite_sources: list[str] = []

            # P1-SYNC: ONE fused finiteness reduction across all heads' new+old log_probs and
            # values (was up to 17 separate .all() GPU->CPU syncs on the happy path). new
            # log_probs are FP32 (the P1-EVAL seam); old log_probs are FP32 because the rollout
            # buffer is FP32-allocated (rollout_buffer.py) regardless of rollout autocast. We
            # .float() both stacks defensively so a future dtype change cannot desync the stack.
            # On failure we drill into the per-head attribution slow path VERBATIM, so
            # head_nan_detected / head_inf_detected / nonfinite_sources are byte-identical.
            new_lp_stack = torch.stack([log_probs[k].float() for k in HEAD_NAMES])
            old_lp_stack = torch.stack(
                [ref_log_probs[k][valid_mask].float() for k in HEAD_NAMES]
            )
            all_finite = (
                torch.isfinite(new_lp_stack).all()
                & torch.isfinite(old_lp_stack).all()
                & torch.isfinite(values).all()
                & torch.isfinite(q_values).all()
            )
            if not bool(all_finite):  # single sync; per-head drill-down only on failure
                # Check new log_probs - separate NaN from Inf per head
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

                # Check anchored reference log_probs (frozen-theta0 baseline)
                for key in HEAD_NAMES:
                    old_lp = ref_log_probs[key][valid_mask]
                    if not torch.isfinite(old_lp).all():
                        nonfinite_count = (~torch.isfinite(old_lp)).sum()
                        nonfinite_sources.append(f"ref_log_probs[{key}]: {nonfinite_count} non-finite")
                        nonfinite_found = True

                # Check values
                if not torch.isfinite(values).all():
                    nonfinite_count = (~torch.isfinite(values)).sum()
                    nonfinite_sources.append(f"values: {nonfinite_count} non-finite")
                    nonfinite_found = True

                # Check q_values (op-conditioned Q-aux): a non-finite q feeds q_aux_loss
                # into total_loss and would corrupt the optimizer step. Gate it on the
                # same skip-epoch path as values.
                if not torch.isfinite(q_values).all():
                    nonfinite_count = (~torch.isfinite(q_values)).sum()
                    nonfinite_sources.append(f"q_values: {nonfinite_count} non-finite")
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

            # DRL Expert (2026-01): Compute AVAILABILITY masks for entropy regularization
            # These track which heads COULD HAVE mattered (action was VALID), not which
            # heads DID matter (action was CHOSEN). Critical for sparse head entropy floor
            # penalty - prevents death spiral where heads collapse once agent stops exploring.
            valid_op_masks = data["op_masks"][valid_mask]
            availability_masks = compute_availability_masks(valid_op_masks)

            # Track conditional entropy (P3-2): entropy only when head is causally relevant
            # This is the true exploration signal for sparse heads like blueprint/tempo.
            # head_masks[key] indicates whether that head affects the gradient for each timestep.
            unforced_mask = ~forced_mask
            learnable_denominator = valid_mask.sum().float().clamp(min=1)
            for key in HEAD_NAMES:
                mask = head_masks[key].float()
                n_relevant = mask.sum().clamp(min=1)
                # Conditional mean: sum(entropy * mask) / count(mask)
                conditional_ent = (entropy[key] * mask).sum() / n_relevant
                conditional_head_entropy_history[key].append(conditional_ent)
                action_choice_mask = masks[key][valid_mask].sum(dim=-1) > 1
                learnable_mask = head_masks[key] & action_choice_mask & unforced_mask
                head_learnable_fraction_history[key].append(
                    learnable_mask.sum().float() / learnable_denominator
                )

            # Compute per-head ratios
            # data['hidden_h'][:,1:] is never indexed here; the loss path uses only
            # data['initial_hidden_h'] (telemetry/Q(s,op) uses the per-step hidden buffer at ~:728-758).
            old_log_probs = {
                "slot": ref_log_probs["slot"][valid_mask],
                "blueprint": ref_log_probs["blueprint"][valid_mask],
                "style": ref_log_probs["style"][valid_mask],
                "tempo": ref_log_probs["tempo"][valid_mask],
                "alpha_target": ref_log_probs["alpha_target"][valid_mask],
                "alpha_speed": ref_log_probs["alpha_speed"][valid_mask],
                "alpha_curve": ref_log_probs["alpha_curve"][valid_mask],
                "op": ref_log_probs["op"][valid_mask],
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
            for key, head_clip in update_result.ratio_metrics.per_head_clip_fraction.items():
                head_clip_fraction_history[key].append(head_clip)

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
                value_coef=self.get_value_coef(),  # Use warmup-aware getter
                q_values=q_values,  # P0-1: current-epoch Q(s, op) for detached aux
                q_aux_coef=0.5 * self.get_value_coef(),  # P0-1: 0.5 * value_coef
                head_names=HEAD_NAMES,
                entropy_floor=self.entropy_floor,
                entropy_floor_penalty_coef=scheduled_coef,  # Scheduled coefficients
                availability_masks=availability_masks,  # DRL Expert: Use availability not causal for entropy
            )
            update_result = PPOUpdateResult(ratio_metrics=ratio_metrics, loss_metrics=losses)
            loss_metrics = update_result.loss_metrics
            assert loss_metrics is not None
            loss = loss_metrics.total_loss

            # Compute auxiliary contribution loss (Phase 3.2)
            # DRL Expert: Only compute when enabled and batch has fresh contribution data
            if self.enable_contribution_aux and "has_fresh_contribution" in data:
                aux_loss = compute_contribution_aux_loss(
                    pred_contributions=result.pred_contributions,
                    target_contributions=data["contribution_targets"],
                    active_mask=data["contribution_mask"],
                    has_fresh_target=data["has_fresh_contribution"],
                    clip=self.contribution_loss_clip,
                )
                # Warmup coefficient (DRL Expert + PyTorch Expert recommendation)
                # Ramp from 0 to full coefficient over aux_warmup_steps
                warmup_progress = min(1.0, self._aux_training_step / max(1, self.aux_warmup_steps))
                effective_aux_coef = self.aux_contribution_coef * warmup_progress
                loss = loss + effective_aux_coef * aux_loss
                metrics["aux_contribution_loss"].append(aux_loss.detach())
                metrics["effective_aux_coef"].append(torch.tensor(effective_aux_coef, device=aux_loss.device))

                # Phase 4.1: Track prediction quality for collapse detection
                # DRL Expert: Monitor variance, explained variance, and correlation
                with torch.no_grad():
                    fresh_mask = data["has_fresh_contribution"]
                    if fresh_mask.any():
                        # Flatten for quality metrics - pred_contributions is [batch, seq_len, num_slots]
                        # valid_mask is [batch, seq_len], fresh_mask is also [batch, seq_len]
                        # We want timesteps where valid_mask AND fresh_mask are True
                        combined_mask = valid_mask & fresh_mask
                        # B3 / no-bug-hiding: numel<2 yields NaN var()/std(). The aux-EV
                        # floor is for the legitimate low (but finite, numel>=2) variance
                        # regime ONLY -- a degenerate (numel<2) selection SKIPS the aux-EV
                        # emission rather than NaN-defaulting or flagging-on-degenerate.
                        if combined_mask.any() and result.pred_contributions[combined_mask].numel() >= 2:
                            pred_flat = result.pred_contributions[combined_mask].flatten()
                            target_flat = data["contribution_targets"][combined_mask].flatten()

                            # Variance (should NOT be ~0 - indicates collapse)
                            pred_variance = pred_flat.var()

                            # EV-telemetry-robustness (SLICE C): variance-floored aux EV +
                            # floor-stabilized aux value_nrmse + low-variance flag, on the
                            # contribution-target scale. Replaces the bare clamp(min=0.01).
                            # DIAGNOSTIC-ONLY (flag + value_nrmse denominator; NEVER a gate).
                            (
                                explained_var,
                                aux_value_nrmse,
                                aux_ev_low_return_variance,
                                _aux_ev_return_variance,
                            ) = compute_floored_aux_explained_variance(
                                pred_flat=pred_flat,
                                target_flat=target_flat,
                                floor_fraction=self.aux_ev_return_variance_floor_fraction,
                                floor_min=self.aux_ev_return_variance_floor_min,
                            )

                            # Correlation (should be > 0.5 eventually)
                            if pred_flat.numel() > 2:
                                corr = torch.corrcoef(
                                    torch.stack([pred_flat, target_flat])
                                )[0, 1]
                                # Handle NaN from corrcoef when variance is zero
                                if not torch.isfinite(corr):
                                    corr = torch.tensor(0.0, device=pred_flat.device)
                            else:
                                corr = torch.tensor(0.0, device=pred_flat.device)

                            metrics["aux_pred_variance"].append(pred_variance)
                            metrics["aux_explained_variance"].append(explained_var)
                            metrics["aux_value_nrmse"].append(aux_value_nrmse)
                            metrics["aux_ev_low_return_variance"].append(aux_ev_low_return_variance)
                            metrics["aux_pred_target_correlation"].append(corr)

                            # Phase 4.2: Collapse detection warnings
                            # DRL Expert: Detect prediction collapse after warmup period
                            # Rate-limit warnings to avoid log spam (every 100 updates)
                            if (
                                self._aux_training_step > self.aux_warmup_steps
                                and self._aux_training_step % 100 == 0
                            ):
                                pred_var_val = pred_variance.item()
                                corr_val = corr.item()

                                if pred_var_val < 0.01:
                                    logger.warning(
                                        "Contribution predictor may have collapsed (variance=%.4f). "
                                        "Consider increasing aux_contribution_coef or disabling stop_gradient.",
                                        pred_var_val,
                                    )
                                if corr_val < 0.2 and corr_val >= 0:  # Skip if NaN (corr_val < 0 can be valid)
                                    logger.warning(
                                        "Contribution predictor correlation low (%.3f). "
                                        "Aux task may not be learning.",
                                        corr_val,
                                    )
                        else:
                            # No valid+fresh (numel>=2) timesteps in this epoch
                            zero_t = torch.tensor(0.0, device=loss.device)
                            metrics["aux_pred_variance"].append(zero_t)
                            metrics["aux_explained_variance"].append(zero_t)
                            metrics["aux_value_nrmse"].append(zero_t)
                            metrics["aux_ev_low_return_variance"].append(False)
                            metrics["aux_pred_target_correlation"].append(zero_t)
                    else:
                        # No fresh measurements in this epoch
                        zero_t = torch.tensor(0.0, device=loss.device)
                        metrics["aux_pred_variance"].append(zero_t)
                        metrics["aux_explained_variance"].append(zero_t)
                        metrics["aux_value_nrmse"].append(zero_t)
                        metrics["aux_ev_low_return_variance"].append(False)
                        metrics["aux_pred_target_correlation"].append(zero_t)
            else:
                # No contribution data in batch - skip aux loss and quality metrics
                metrics["aux_contribution_loss"].append(torch.tensor(0.0, device=loss.device))
                metrics["effective_aux_coef"].append(torch.tensor(0.0, device=loss.device))
                zero_t = torch.tensor(0.0, device=loss.device)
                metrics["aux_pred_variance"].append(zero_t)
                metrics["aux_explained_variance"].append(zero_t)
                metrics["aux_value_nrmse"].append(zero_t)
                metrics["aux_ev_low_return_variance"].append(False)
                metrics["aux_pred_target_correlation"].append(zero_t)

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

                # P0-1: "value" -> state_value_head (op-INDEPENDENT V(s) baseline);
                # "q" -> q_head (op-conditioned telemetry/aux). Both emitted so the
                # detached q_head's grad (it trains only its own params) stays visible.
                head_names = ["slot", "blueprint", "style", "tempo", "alpha_target",
                              "alpha_speed", "alpha_curve", "op", "value", "q"]
                head_modules = [
                    base_network.slot_head, base_network.blueprint_head, base_network.style_head,
                    base_network.tempo_head, base_network.alpha_target_head, base_network.alpha_speed_head,
                    base_network.alpha_curve_head, base_network.op_head,
                    base_network.state_value_head, base_network.q_head,
                ]

                # Collect all head norms as tensors (no .item() yet)
                head_norm_tensors: list[torch.Tensor] = []
                for head_name, head_module in zip(head_names, head_modules):
                    head_params = list(head_module.parameters())
                    head_has_trainable_params = any(p.requires_grad for p in head_params)
                    params_with_grad = [p for p in head_params if p.grad is not None]
                    if params_with_grad:
                        has_nonfinite_grad = any(
                            not torch.isfinite(p.grad).all() for p in params_with_grad
                        )
                        norm_t = torch.linalg.vector_norm(
                            torch.stack([torch.linalg.vector_norm(p.grad) for p in params_with_grad])
                        )
                        grad_state = "nonfinite" if has_nonfinite_grad else "finite"
                    else:
                        # BUG FIX: Use NaN to signal "no gradient data" instead of 0.0
                        # 0.0 would hide the bug (No Bug-Hiding Patterns rule)
                        # NaN signals missing data and will surface in telemetry
                        norm_t = torch.tensor(float("nan"), device=self.device)
                        grad_state = "missing" if head_has_trainable_params else "not_learnable"
                    head_norm_tensors.append(norm_t)
                    head_gradient_state_history[head_name].append(grad_state)

                all_norms = torch.stack(head_norm_tensors)
                # P1-SYNC: batch the per-head learnable-fraction == 0 compare into ONE sync
                # (was one .item() per head). Stack each head's latest fraction, materialize
                # once, then index. Preserves the "not_learnable" relabel bit-for-bit.
                lf_heads = [hn for hn in head_names if hn in head_learnable_fraction_history]
                if lf_heads:
                    lf_is_zero = (
                        torch.stack([head_learnable_fraction_history[hn][-1] for hn in lf_heads])
                        == 0.0
                    ).tolist()  # single sync
                    lf_zero = dict(zip(lf_heads, lf_is_zero))
                else:
                    lf_zero = {}
                for head_name, grad_norm in zip(head_names, all_norms):
                    if head_name in lf_zero and lf_zero[head_name]:
                        head_gradient_state_history[head_name][-1] = "not_learnable"
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
                losses.q_aux_loss,  # P0-1: detached Q telemetry regression loss
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
            metrics["q_aux_loss"].append(logging_tensors[12])
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
        # Count actual optimizer steps, not epochs that only reached ratio diagnostics.
        # KL early-stop at epoch 0 records ratio stats before breaking, but it does not
        # execute backward(), gradient clipping, or optimizer.step().
        epochs_completed = len(metrics["pre_clip_grad_norm"])

        builder = PPOUpdateMetricsBuilder(
            metrics=metrics,
            finiteness_failures=finiteness_failures,
            epochs_completed=epochs_completed,
            head_entropies=head_entropy_history,
            conditional_head_entropies=conditional_head_entropy_history,
            head_grad_norms=head_grad_norm_history,
            head_learnable_fractions=head_learnable_fraction_history,
            head_gradient_states=head_gradient_state_history,
            head_nan_detected=head_nan_detected,
            head_inf_detected=head_inf_detected,
            lstm_health_history=lstm_health_history,
            log_prob_min_across_epochs=log_prob_min_across_epochs,
            log_prob_max_across_epochs=log_prob_max_across_epochs,
            head_ratio_max_across_epochs=head_ratio_max_across_epochs,
            joint_ratio_max_across_epochs=joint_ratio_max_across_epochs,
            head_clip_fraction_history=head_clip_fraction_history,
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
            self._aux_training_step += 1

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
            # P0-1: value-head topology version (leyline single source of truth). Bumped
            # to 2 when the op-conditioned value_head was split into an op-INDEPENDENT
            # state_value_head (PPO baseline V(s)) + a renamed q_head (telemetry/aux).
            # Load asserts equality; pre-v2 checkpoints fail strict load BY DESIGN.
            'value_head_schema_version': VALUE_HEAD_SCHEMA_VERSION,
            'network_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_normalizer_state_dict': self.value_normalizer.state_dict(),
            'train_steps': self.train_steps,
            'aux_training_step': self._aux_training_step,
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
                'value_coef_start': self.value_coef_start,
                'value_warmup_steps': self.value_warmup_steps,
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
                # Auxiliary contribution supervision config
                'aux_contribution_coef': self.aux_contribution_coef,
                'aux_warmup_steps': self.aux_warmup_steps,
                'aux_stop_gradient': self.aux_stop_gradient,
                'contribution_loss_clip': self.contribution_loss_clip,
                'enable_contribution_aux': self.enable_contribution_aux,
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
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls.load_from_checkpoint_dict(
            checkpoint,
            device=device,
            compile_mode=compile_mode,
        )

    @classmethod
    def load_from_checkpoint_dict(
        cls,
        checkpoint: dict[str, Any],
        *,
        device: str = "cuda:0",
        compile_mode: str | None = None,
        expected_slot_config: SlotConfig | None = None,
    ) -> "PPOAgent":
        """Load agent from an already materialized checkpoint dictionary.

        Args:
            checkpoint: Checkpoint dictionary loaded on CPU.
            device: Device to place the reconstructed agent on.
            compile_mode: Override checkpoint's torch.compile mode.
            expected_slot_config: Runtime slot config that must match the checkpoint.

        Returns:
            PPOAgent with restored weights and configuration.

        Raises:
            RuntimeError: If checkpoint architecture is incompatible.
        """

        # Required checkpoint fields - fail fast if missing (no backwards compat)
        try:
            version = checkpoint['checkpoint_version']
            value_head_schema_version = checkpoint['value_head_schema_version']
            state_dict = checkpoint['network_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            architecture = checkpoint['architecture']
            config = checkpoint['config']
            train_steps = checkpoint['train_steps']
            aux_training_step = checkpoint['aux_training_step']
        except KeyError as e:
            raise RuntimeError(
                f"Incompatible checkpoint format: missing required field {e}. "
                f"This checkpoint was saved with an older version that is no longer supported. "
                f"Please retrain the model to create a compatible checkpoint."
            ) from e

        # P0-1 CHECKPOINT BREAK (intended, No-Legacy): the value-head topology changed.
        # The PPO baseline is now an op-INDEPENDENT state_value_head (V(s)); the old
        # op-conditioned value_head was renamed q_head (telemetry/aux). There is NO
        # remap/shim -- a pre-v2 checkpoint has value_head.* (no state_value_head.* /
        # q_head.*) and would fail strict load anyway; this assert names the break first.
        if value_head_schema_version != VALUE_HEAD_SCHEMA_VERSION:
            raise RuntimeError(
                f"Value-head schema mismatch: checkpoint has "
                f"value_head_schema_version={value_head_schema_version}, but this build "
                f"expects {VALUE_HEAD_SCHEMA_VERSION}. The value head was split into an "
                f"op-INDEPENDENT state_value_head (PPO baseline V(s)) plus an op-conditioned "
                f"q_head (telemetry/aux). Old checkpoints (single op-conditioned value_head) "
                f"are incompatible by design -- there is no remap. Please retrain."
            )

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
        checkpoint_slot_ids = tuple(architecture['slot_ids'])
        if (
            expected_slot_config is not None
            and checkpoint_slot_ids != expected_slot_config.slot_ids
        ):
            raise RuntimeError(
                "Checkpoint slot_ids do not match runtime slot_ids: "
                f"checkpoint={checkpoint_slot_ids}, "
                f"runtime={expected_slot_config.slot_ids}"
            )
        slot_config = SlotConfig(slot_ids=checkpoint_slot_ids)

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
        if "n_epochs" in config:
            raise RuntimeError(
                "Incompatible checkpoint: config.n_epochs is no longer supported. "
                "Please retrain the model to create a compatible checkpoint."
            )
        # Remove config params that are now part of PolicyBundle.
        agent_config = {k: v for k, v in config.items() if k != 'lstm_hidden_dim'}
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
        agent._aux_training_step = aux_training_step

        # P1 FIX: Restore value_normalizer state if present
        # This ensures consistent normalization scale across training resumption
        if value_normalizer_state is not None:
            agent.value_normalizer.load_state_dict(value_normalizer_state)

        # === Apply torch.compile AFTER loading weights ===
        # Critical: Compile must happen after state_dict to ensure graph traces
        # the actual loaded weights, not random initialization.
        # P3 FIX: Use effective_compile_mode (may be overridden by parameter)
        if effective_compile_mode != "off":
            # P3-DYN: dynamic=False (static rollout shapes); see factory.py rationale.
            agent.policy.compile(mode=effective_compile_mode, dynamic=False)

        # Update agent's stored compile_mode to reflect what we actually used
        agent.compile_mode = effective_compile_mode

        return agent


__all__ = [
    "PPOAgent",
]
