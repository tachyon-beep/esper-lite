"""Tamiyo Policy Features - HOT PATH Feature Extraction

CRITICAL: This module is on the HOT PATH for vectorized training.
ONLY import from leyline. NO imports from kasmina, tamiyo, or nissa!

This module extracts features from observations for RL training.
It must be FAST and have minimal dependencies to avoid bottlenecks
in the vectorized PPO training loop.

Note: Moved from simic.control to tamiyo.policy as part of the
Policy Migration (PR #31). Re-exported from simic.control for compatibility.
"""

from __future__ import annotations

import math
import threading
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.slot_config import SlotConfig
from esper.leyline.stages import SeedStage
# Phase 2 imports: Constants needed for Obs V3 feature extraction
from esper.leyline import (
    DEFAULT_GAMMA,
    NUM_OPS,
    NUM_STAGES,
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_NON_BLUEPRINT_DIM,
    OBS_V3_SLOT_FEATURE_SIZE,
    TaskConfig,  # Cross-subsystem task configuration
    safe,  # Cross-subsystem safe value conversion
)
# Stage schema for validation and one-hot encoding
# NOTE: Imported at module level since these are fast O(1) lookups used in hot path
from esper.leyline.stage_schema import (
    stage_to_one_hot as _stage_to_one_hot,
    STAGE_TO_INDEX,
)

# HOT PATH: ONLY leyline imports allowed!


# =============================================================================
# Symlog Transform for LSTM Saturation Prevention
# =============================================================================
# LSTM hidden/cell states were saturating (h=70.9, c=337.6) due to high-magnitude
# inputs like gradient norms (can be 100+). Symlog compresses these while preserving
# sign and relative ordering. Used in DreamerV3/MuZero for exactly this purpose.
#
# Compression examples: 1→0.69, 10→2.4, 100→4.6, 1000→6.9
# Gradient: d/dx = 1/(|x|+1) - dampens large inputs, never zero.

# Symlog normalization constant: symlog(1000) ≈ 6.91, divide by 7 for ~[0,1] range
_SYMLOG_NORM = 7.0


def symlog(x: float) -> float:
    """Symmetric log transform for magnitude compression (scalar version).

    Compresses large telemetry values to prevent LSTM saturation.
    Applied to gradient norms and other potentially unbounded metrics.

    Properties:
        - symlog(0) = 0
        - Monotonic, bijective
        - Preserves sign
        - Dampens gradients for large |x|: d/dx = 1/(|x|+1)
    """
    if x >= 0:
        return math.log1p(x)
    else:
        return -math.log1p(-x)


def symlog_tensor(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform for magnitude compression (tensor version).

    torch.compile-friendly formulation using log1p for better kernel fusion.
    """
    return torch.where(x >= 0, torch.log1p(x), -torch.log1p(-x))


if TYPE_CHECKING:
    # Type hints only - not imported at runtime
    pass


__all__ = [
    "safe",
    "symlog",
    "symlog_tensor",
    "batch_obs_to_features",
    "MULTISLOT_FEATURE_SIZE",
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
]


# safe() now lives in leyline/utils.py (re-exported above via leyline import)


# =============================================================================
# Observation V3 Feature Extraction (Phase 2)
# =============================================================================

def _pad_history(history: list[float], length: int = 5) -> list[float]:
    """Left-pad history to fixed length with zeros.

    Args:
        history: Raw history values (may be shorter than length at episode start)
        length: Target length (default 5)

    Returns:
        List of exactly `length` values, left-padded with 0.0 if needed
    """
    if len(history) >= length:
        return history[-length:]
    return [0.0] * (length - len(history)) + history


def _extract_base_features_v3(
    signal: Any,  # TrainingSignals
    env_state: Any,  # ParallelEnvState
    num_training: int,
    num_blending: int,
    num_holding: int,
    slot_config: SlotConfig,
    max_epochs: int,
) -> torch.Tensor:
    """Extract base features (23 dims) for Obs V3.

    Base features:
    - Current epoch (1 dim): normalized to [0, 1] using runtime max_epochs
    - Current val_loss (1 dim): log-normalized
    - Current val_accuracy (1 dim): normalized to [0, 1]
    - Raw loss history (5 dims): log-normalized, left-padded
    - Raw accuracy history (5 dims): normalized to [0, 1], left-padded
    - Stage distribution (3 dims): num_training_norm, num_blending_norm, num_holding_norm
    - Action feedback (7 dims): last_action_success (1) + last_action_op one-hot (6)

    Total: 23 base features

    Note: Removed host_stabilized - Tamiyo learns stability from raw telemetry.

    Args:
        signal: TrainingSignals with metrics and history
        env_state: ParallelEnvState with action feedback
        num_training: Number of seeds in TRAINING stage
        num_blending: Number of seeds in BLENDING stage
        num_holding: Number of seeds in HOLDING stage
        slot_config: SlotConfig for normalization
        max_epochs: Runtime episode length for normalization

    Returns:
        torch.Tensor with shape (23,)
    """
    # Current metrics (3 dims)
    # Normalize epoch to [0, 1] range using runtime max_epochs
    max_epochs_den = float(max_epochs)
    epoch_norm = float(signal.metrics.epoch) / max_epochs_den
    # Loss normalization: symlog to prevent LSTM saturation
    # Old: log(1+loss)/log(16) allowed loss=100 → 1.67 (exceeded 1.0)
    # New: symlog(loss)/7 keeps all values in ~[0, 1] range
    # CIFAR-10 baseline 2.3 → 0.17, loss=100 → 0.66, loss=1000 → 0.99
    val_loss_norm = symlog(signal.metrics.val_loss) / _SYMLOG_NORM
    val_accuracy_norm = signal.metrics.val_accuracy / 100.0

    # Extract and normalize loss history (5 dims) - symlog normalization
    loss_history_padded = _pad_history(signal.loss_history, 5)
    loss_history_norm = [symlog(x) / _SYMLOG_NORM for x in loss_history_padded]

    # Extract and normalize accuracy history (5 dims)
    acc_history_padded = _pad_history(signal.accuracy_history, 5)
    acc_history_norm = [x / 100.0 for x in acc_history_padded]

    # Stage distribution (3 dims) - normalize by actual slot count
    # CRITICAL: Must use slot_config.num_slots for correctness with arbitrary grid sizes
    max_slots = float(slot_config.num_slots)
    num_training_norm = num_training / max_slots
    num_blending_norm = num_blending / max_slots
    num_holding_norm = num_holding / max_slots

    # Action feedback (7 dims): last_action_success + last_action_op one-hot (6 dims)
    last_action_success = 1.0 if env_state.last_action_success else 0.0

    # Validate last_action_op is in valid range before one-hot encoding
    assert 0 <= env_state.last_action_op < NUM_OPS, \
        f"Invalid last_action_op: {env_state.last_action_op} (expected 0-{NUM_OPS-1})"

    last_op_one_hot = F.one_hot(
        torch.tensor(env_state.last_action_op),
        num_classes=NUM_OPS
    ).float()

    # Combine all base features (23 dims total)
    base_features = (
        [epoch_norm] +  # 1 dim - normalized to [0, 1]
        [val_loss_norm] +  # 1 dim
        [val_accuracy_norm] +  # 1 dim
        loss_history_norm +  # 5 dims
        acc_history_norm +   # 5 dims
        [num_training_norm, num_blending_norm, num_holding_norm] +  # 3 dims
        [last_action_success] +  # 1 dim
        last_op_one_hot.tolist()  # 6 dims
    )

    return torch.tensor(base_features, dtype=torch.float32)


def _extract_slot_features_v3(
    slot_report: Any,  # SeedStateReport
    env_state: Any,  # ParallelEnvState
    slot_id: str,
    max_epochs: int,
) -> torch.Tensor:
    """Extract per-slot features (31 dims) for Obs V3.

    Features:
    - is_active (1 dim) - always 1.0 since this function is called for active slots
    - Stage one-hot (10 dims) - from SeedStage enum
    - Current alpha (1 dim)
    - Improvement (1 dim) - counterfactual contribution
    - Contribution velocity (1 dim)
    - Blend tempo epochs (1 dim)
    - Alpha scaffolding (8 dims): target, mode, steps_total, steps_done, time_to_target, velocity, algorithm, interaction_sum
    - Telemetry merged (4 dims): gradient_norm, gradient_health, has_vanishing, has_exploding
    - gradient_health_prev (1 dim) - enables LSTM trend detection
    - epochs_in_stage_norm (1 dim) - normalized epochs in current stage
    - counterfactual_fresh (1 dim) - DEFAULT_GAMMA ** epochs_since_cf
    - seed_age_norm (1 dim) - normalized epochs since germination (metrics.epochs_total)

    Total: 31 dims

    Note: boost_received, upstream_alpha_sum, and downstream_alpha_sum from V2 are REMOVED in V3.

    Args:
        slot_report: SeedStateReport with current slot state
        env_state: ParallelEnvState with gradient health tracking
        slot_id: Slot identifier for looking up tracked state
        max_epochs: Runtime episode length for normalization

    Returns:
        torch.Tensor with shape (30,)
    """
    # Stage one-hot encoding (10 dims)
    # _stage_to_one_hot raises ValueError for invalid stages - fail fast, don't mask bugs
    stage_val = slot_report.stage.value
    stage_one_hot = _stage_to_one_hot(stage_val)

    # Current alpha (1 dim)
    current_alpha = slot_report.metrics.current_alpha

    # Improvement - use counterfactual contribution if available, else improvement_since_stage_start (1 dim)
    contribution = slot_report.metrics.counterfactual_contribution
    if contribution is None:
        contribution = slot_report.metrics.improvement_since_stage_start
    contribution_norm = max(-1.0, min(contribution / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))

    # Contribution velocity (1 dim) - raw velocity, not fossilize lookahead
    velocity = slot_report.metrics.contribution_velocity
    velocity_norm = max(-1.0, min(velocity / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))

    # Blend tempo epochs (1 dim) - normalized to [0, 1] range (max ~12 epochs)
    blend_tempo_norm = float(slot_report.blend_tempo_epochs) / 12.0

    # Alpha scaffolding (8 dims)
    alpha_target = slot_report.alpha_target
    alpha_mode_norm = float(slot_report.alpha_mode) / max(_ALPHA_MODE_MAX, 1)

    # Normalize alpha schedule steps by runtime episode length for multi-seed chaining visibility.
    # The LSTM needs to distinguish:
    # - Seed A at epoch 50 (fossilization decision)
    # - Seed B at epoch 75 (mid-training)
    # - Seed C at epoch 145 (end-of-episode)
    # Using shorter-than-runtime denominators would saturate these to 1.0,
    # breaking sequential scaffolding credit assignment.
    max_epochs_den = float(max_epochs)
    alpha_steps_total_norm = min(float(slot_report.alpha_steps_total), max_epochs_den) / max_epochs_den
    alpha_steps_done_norm = min(float(slot_report.alpha_steps_done), max_epochs_den) / max_epochs_den
    time_to_target_norm = min(float(slot_report.time_to_target), max_epochs_den) / max_epochs_den

    alpha_velocity = max(-1.0, min(slot_report.alpha_velocity, 1.0))
    alpha_algorithm_norm = float(slot_report.alpha_algorithm - _ALPHA_ALGO_MIN) / _ALPHA_ALGO_RANGE
    interaction_sum_norm = max(
        -1.0, min(slot_report.metrics.interaction_sum / 10.0, 1.0)
    )

    # Telemetry merged (4 dims)
    # Access telemetry fields from slot_report.telemetry if available
    if slot_report.telemetry is not None:
        # gradient_norm: Apply symlog to compress high magnitudes (100+ → ~4.6)
        grad_norm_raw = slot_report.telemetry.gradient_norm
        if grad_norm_raw is not None and math.isfinite(grad_norm_raw):
            gradient_norm = symlog(grad_norm_raw) / _SYMLOG_NORM
        else:
            gradient_norm = 0.0
        gradient_health = safe(slot_report.telemetry.gradient_health, 1.0, max_val=1.0)
        has_vanishing = 1.0 if slot_report.telemetry.has_vanishing else 0.0
        has_exploding = 1.0 if slot_report.telemetry.has_exploding else 0.0
    else:
        # Default values when telemetry not available (shouldn't happen in Obs V3)
        gradient_norm = 0.0
        gradient_health = 1.0  # Assume healthy
        has_vanishing = 0.0
        has_exploding = 0.0

    # gradient_health_prev (1 dim) - from env_state tracking
    # Default to 1.0 (healthy) if not yet tracked for this slot
    gradient_health_prev = env_state.gradient_health_prev.get(slot_id, 1.0)
    # Fail-fast if NaN was stored in gradient_health_prev (indicates upstream bug)
    if not math.isfinite(gradient_health_prev):
        raise ValueError(
            f"NaN/inf in gradient_health_prev for slot {slot_id}. "
            f"Value: {gradient_health_prev}. This indicates a bug in the gradient "
            f"telemetry pipeline - check vectorized.py where gradient_health is stored."
        )

    # epochs_in_stage_norm (1 dim) - normalize to [0, 1] using runtime max_epochs.
    # Must use full episode length to preserve temporal resolution for multi-seed chaining.
    # The LSTM needs to distinguish stages at different episode points (early/mid/late fossilization).
    epochs_in_stage = slot_report.metrics.epochs_in_current_stage
    epochs_in_stage_norm = min(float(epochs_in_stage), max_epochs_den) / max_epochs_den

    # counterfactual_fresh (1 dim) - gamma-matched decay
    # DEFAULT_GAMMA ** epochs_since_counterfactual
    # With DEFAULT_GAMMA=0.995, signal stays >0.5 for ~138 epochs
    epochs_since_cf = env_state.epochs_since_counterfactual.get(slot_id, 0)
    counterfactual_fresh = DEFAULT_GAMMA ** epochs_since_cf

    # seed_age_norm (1 dim) - normalize to [0, 1] using runtime max_epochs.
    # This exposes prune-age gating and distinguishes "new vs old" seeds even
    # after multiple stage transitions (epochs_in_stage resets on transitions).
    seed_age = slot_report.metrics.epochs_total
    seed_age_norm = min(float(seed_age), max_epochs_den) / max_epochs_den

    # Combine all slot features (31 dims total)
    slot_features = (
        [1.0] +  # is_active (1 dim) - always 1.0 for active slots
        stage_one_hot +  # 10 dims
        [current_alpha] +  # 1 dim
        [contribution_norm] +  # 1 dim
        [velocity_norm] +  # 1 dim
        [blend_tempo_norm] +  # 1 dim
        # Alpha scaffolding (8 dims)
        [
            alpha_target,
            alpha_mode_norm,
            alpha_steps_total_norm,
            alpha_steps_done_norm,
            time_to_target_norm,
            alpha_velocity,
            alpha_algorithm_norm,
            interaction_sum_norm,
        ] +
        # Telemetry merged (4 dims)
        [
            gradient_norm,
            gradient_health,
            has_vanishing,
            has_exploding,
        ] +
        # New Obs V3 fields (3 dims)
        [
            gradient_health_prev,
            epochs_in_stage_norm,
            counterfactual_fresh,
            seed_age_norm,
        ]
    )

    return torch.tensor(slot_features, dtype=torch.float32)


# =============================================================================
# Vectorized Construction (Optimized for high n_envs)
# =============================================================================
# DONE: Vectorized one-hot and device-cached tables are now wired into
# batch_obs_to_features(). Key optimizations:
#   1. Pre-allocates output tensors directly on target device
#   2. Uses integer enum comparisons (SeedStage.value) instead of strings
#   3. Uses _get_cached_stage_table()/_get_cached_op_table() for one-hot lookups
#   4. Fills features in-place via tensor indexing
#
# The Python loops over envs/slots remain (data comes from Python objects),
# but allocation overhead is eliminated - single tensor allocation per call
# instead of O(n_envs * num_slots) small tensor allocations.

# Module-level one-hot tables for stage and op encoding
_STAGE_ONE_HOT_TABLE = torch.eye(NUM_STAGES, dtype=torch.float32)
_OP_ONE_HOT_TABLE = torch.eye(NUM_OPS, dtype=torch.float32)

# Device-keyed caches to avoid per-step .to(device) allocations
_STAGE_DEVICE_CACHE: dict[torch.device, torch.Tensor] = {}
_OP_DEVICE_CACHE: dict[torch.device, torch.Tensor] = {}
_DEVICE_CACHE_LOCK = threading.Lock()

# Stage value constants for fast integer comparison (avoid string comparisons in hot path)
_TRAINING_VAL = SeedStage.TRAINING.value
_BLENDING_VAL = SeedStage.BLENDING.value
_HOLDING_VAL = SeedStage.HOLDING.value
_FOSSILIZED_VAL = SeedStage.FOSSILIZED.value


def _normalize_device(device: torch.device | str) -> torch.device:
    """Normalize device to canonical form.

    Handles:
    - "cuda" or torch.device("cuda") → torch.device("cuda", current_device)
    - "cpu" or torch.device("cpu") → torch.device("cpu")
    - "cuda:0" → torch.device("cuda", 0)

    This prevents duplicate cache entries for the same physical device.
    torch.device("cuda") != torch.device("cuda:0") but they're the same GPU.
    """
    # Convert string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _get_cached_stage_table(device: torch.device) -> torch.Tensor:
    """Get stage one-hot table for device, caching to avoid repeated transfers.

    Thread-safe via double-checked locking pattern.
    """
    device = _normalize_device(device)
    if device not in _STAGE_DEVICE_CACHE:
        with _DEVICE_CACHE_LOCK:
            if device not in _STAGE_DEVICE_CACHE:  # Double-check after acquiring lock
                _STAGE_DEVICE_CACHE[device] = _STAGE_ONE_HOT_TABLE.to(device)
    return _STAGE_DEVICE_CACHE[device]


def _get_cached_op_table(device: torch.device) -> torch.Tensor:
    """Get op one-hot table for device, caching to avoid repeated transfers.

    Thread-safe via double-checked locking pattern.
    """
    device = _normalize_device(device)
    if device not in _OP_DEVICE_CACHE:
        with _DEVICE_CACHE_LOCK:
            if device not in _OP_DEVICE_CACHE:  # Double-check after acquiring lock
                _OP_DEVICE_CACHE[device] = _OP_ONE_HOT_TABLE.to(device)
    return _OP_DEVICE_CACHE[device]


def _vectorized_stage_one_hot(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Vectorized stage one-hot encoding with device-cached lookup table.

    Args:
        indices: Stage indices of shape [..., num_slots] where -1 marks inactive.
                 Supports arbitrary leading batch dimensions.
        device: Target device for output tensor.

    Returns:
        One-hot encoded tensor of shape [..., num_slots, NUM_STAGES].
        Inactive slots (index -1) are all zeros.
    """
    table = _get_cached_stage_table(device)
    valid_mask = indices >= 0  # -1 marks inactive slots
    safe_indices = indices.clamp(min=0)
    one_hot = table[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive


def _vectorized_op_one_hot(op_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Vectorized operation one-hot encoding with device-cached lookup table.

    Args:
        op_indices: Operation indices of shape [n_envs] (values 0 to NUM_OPS-1).
        device: Target device for output tensor.

    Returns:
        One-hot encoded tensor of shape [n_envs, NUM_OPS].
    """
    table = _get_cached_op_table(device)
    return table[op_indices]


# =============================================================================
# Multi-Slot Features (V4 - dynamic dimensions)
# =============================================================================

# Obs V3 Constants
# Base features: 23 dims (epoch, val_loss, val_accuracy, history_5 × 2, stage distribution,
#                         action feedback: last_action_success + last_action_op one-hot)
# NOTE: Includes 7 action feedback dims (last_action_success + 6-dim one-hot for last_action_op)
# NOTE: Removed host_stabilized - Tamiyo learns stability from raw telemetry
BASE_FEATURE_SIZE = OBS_V3_BASE_FEATURE_SIZE

# Per-slot features (Obs V3 - NO blueprint one-hot, moved to embedding):
# 1 is_active
# + 10 stage one-hot (SeedStage categorical encoding via StageSchema)
# + 1 current_alpha
# + 1 improvement (counterfactual contribution)
# + 1 contribution_velocity
# + 1 blend_tempo_epochs
# + 8 alpha scaffolding (target, mode, steps_total, steps_done, time_to_target, velocity, algorithm, interaction_sum)
# + 4 telemetry merged (gradient_norm, gradient_health, has_vanishing, has_exploding)
# + 1 gradient_health_prev
# + 1 epochs_in_stage_norm
# + 1 counterfactual_fresh
# + 1 seed_age_norm (metrics.epochs_total / max_epochs)
# Total: 1 + 10 + 1 + 1 + 1 + 1 + 8 + 4 + 1 + 1 + 1 + 1 = 31 dims per slot (blueprint moved to embedding)
SLOT_FEATURE_SIZE = OBS_V3_SLOT_FEATURE_SIZE

# Obs V3 total for 3 slots: 23 base + 3 slots × 31 features = 116 dims (excludes blueprint embeddings)
# Blueprint embeddings (4 dims × 3 slots = 12) are added inside the network, making total network input 128
# NOTE: Default for 3-slot configuration. Use get_feature_size(slot_config) for dynamic slot counts.
MULTISLOT_FEATURE_SIZE = OBS_V3_NON_BLUEPRINT_DIM

# Observation normalization bounds (kept local to avoid heavier imports on the HOT PATH)
_IMPROVEMENT_CLAMP_PCT_PTS: float = 10.0  # Clamp improvement to ±10 percentage points → [-1, 1]
_ALPHA_MODE_MAX: int = max(mode.value for mode in AlphaMode)
_ALPHA_ALGO_MIN: int = min(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_MAX: int = max(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_RANGE: int = max(_ALPHA_ALGO_MAX - _ALPHA_ALGO_MIN, 1)


def get_feature_size(slot_config: SlotConfig) -> int:
    """Feature size excluding blueprint embeddings (added by network).

    Obs V3 Breakdown:
        Base features:     23 dims (epoch, val_loss, val_accuracy,
                                    loss_history_5, accuracy_history_5,
                                    stage distribution,
                                    action feedback: last_action_success + last_action_op one-hot)
        Per-slot features: 31 dims × num_slots (stage one-hot, alpha, gradient health,
                                                 contribution, telemetry, etc.)

    For 3 slots: 23 + (31 × 3) = 116 dims

    Note: Blueprint embeddings (4 dims × num_slots) are added inside the network
    via BlueprintEmbedding module, making total network input 116 + 12 = 128 for 3 slots.

    Args:
        slot_config: Slot configuration defining number of slots

    Returns:
        Total observation feature size (excluding blueprint embeddings)
    """
    return OBS_V3_BASE_FEATURE_SIZE + (OBS_V3_SLOT_FEATURE_SIZE * slot_config.num_slots)


def batch_obs_to_features(
    batch_signals: list[Any],  # TrainingSignals
    batch_slot_reports: list[dict[str, Any]],  # dict[str, SeedStateReport]
    batch_env_states: list[Any],  # list[ParallelEnvState]
    slot_config: SlotConfig,
    device: torch.device,
    *,
    max_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for batch of environments (Obs V3).

    OPTIMIZED for high n_envs (64-256+):
    - Pre-allocates single output tensor on CPU, fills in-place, then single H2D transfer
      (vs old: many small tensor allocations + stack + transfer)
    - Uses integer enum comparisons (stage.value) instead of string comparisons
    - Uses cached one-hot lookup tables instead of F.one_hot() per-element

    Note: We fill on CPU because Python-loop element writes to GPU tensors are slow
    (each indexed write triggers a CUDA kernel launch). Single H2D transfer at end
    is efficient; the optimization is reducing allocation count, not transfer count.

    Returns tuple of (obs, blueprint_indices):
    - obs: [batch, obs_dim] - base features + slot features (NO blueprint one-hot)
    - blueprint_indices: [batch, num_slots] - int64 indices for nn.Embedding lookup

    Feature breakdown:
    - Base features: 23 dims (epoch, loss, accuracy, raw history, stage counts)
    - Action feedback: 7 dims (last_action_success + last_action_op one-hot) - INCLUDED in base
    - Per-slot features: 31 dims × num_slots (stage, alpha, gradient health, contribution, etc.)

    For 3 slots: 23 + (31 × 3) = 116 dims

    Args:
        batch_signals: List of TrainingSignals with metrics and history
        batch_slot_reports: List of dict[str, SeedStateReport] for each env
        batch_env_states: List of ParallelEnvState with action feedback and tracking
        slot_config: Slot configuration defining num_slots and slot_ids
        device: Target device for output tensors
        max_epochs: Runtime episode length for normalization

    Returns:
        obs: [batch, obs_dim] - observation features (base + slots, no blueprint)
        blueprint_indices: [batch, num_slots] - blueprint indices for embedding (int64)
    """
    n_envs = len(batch_signals)
    num_slots = slot_config.num_slots
    obs_dim = OBS_V3_BASE_FEATURE_SIZE + OBS_V3_SLOT_FEATURE_SIZE * num_slots

    # Pre-allocate on CPU for Python-loop filling, then transfer once to device.
    # CRITICAL: Do NOT pre-allocate on GPU and fill element-by-element - each indexed
    # write to a GPU tensor triggers a CUDA kernel launch, making it 10-100x slower!
    # The old code built many small CPU tensors then stack+transfer. The new code
    # builds one CPU tensor then transfers - same H2D copy count, fewer allocations.
    obs = torch.zeros((n_envs, obs_dim), dtype=torch.float32)
    blueprint_indices = torch.full((n_envs, num_slots), -1, dtype=torch.int64)

    # Get cached one-hot lookup tables (on CPU for indexing, will transfer with obs)
    op_table = _OP_ONE_HOT_TABLE  # CPU table - transferred with obs at end
    stage_table = _STAGE_ONE_HOT_TABLE  # CPU table

    # Pre-compute normalization constants
    max_epochs_den = float(max_epochs)
    max_slots = float(num_slots)

    # Fill features for each environment
    for env_idx in range(n_envs):
        signal = batch_signals[env_idx]
        env_state = batch_env_states[env_idx]
        reports = batch_slot_reports[env_idx]

        # Count stage distribution using integer enum values (not strings)
        num_training = 0
        num_blending = 0
        num_holding = 0
        for report in reports.values():
            stage_val = report.stage.value
            if stage_val == _TRAINING_VAL:
                num_training += 1
            elif stage_val == _BLENDING_VAL:
                num_blending += 1
            elif stage_val == _HOLDING_VAL or stage_val == _FOSSILIZED_VAL:
                num_holding += 1

        # === BASE FEATURES (23 dims) ===
        # Fill directly into pre-allocated tensor

        # Current metrics (3 dims: epoch, val_loss, val_accuracy)
        epoch_norm = float(signal.metrics.epoch) / max_epochs_den
        # Loss normalization: symlog to prevent LSTM saturation
        val_loss_norm = symlog(signal.metrics.val_loss) / _SYMLOG_NORM
        val_accuracy_norm = signal.metrics.val_accuracy / 100.0

        obs[env_idx, 0] = epoch_norm
        obs[env_idx, 1] = val_loss_norm
        obs[env_idx, 2] = val_accuracy_norm

        # Loss history (5 dims) - symlog normalized
        loss_history_padded = _pad_history(signal.loss_history, 5)
        for i, loss_val in enumerate(loss_history_padded):
            obs[env_idx, 3 + i] = symlog(loss_val) / _SYMLOG_NORM

        # Accuracy history (5 dims) - normalized to [0, 1]
        acc_history_padded = _pad_history(signal.accuracy_history, 5)
        for i, acc_val in enumerate(acc_history_padded):
            obs[env_idx, 8 + i] = acc_val / 100.0

        # Stage distribution (3 dims)
        obs[env_idx, 13] = num_training / max_slots
        obs[env_idx, 14] = num_blending / max_slots
        obs[env_idx, 15] = num_holding / max_slots

        # Action feedback (7 dims: last_action_success + 6-dim one-hot)
        obs[env_idx, 16] = 1.0 if env_state.last_action_success else 0.0

        # Validate last_action_op range before indexing
        assert 0 <= env_state.last_action_op < NUM_OPS, \
            f"Invalid last_action_op: {env_state.last_action_op} (expected 0-{NUM_OPS-1})"
        obs[env_idx, 17:23] = op_table[env_state.last_action_op]

        # === SLOT FEATURES (30 dims per slot) ===
        for slot_idx, slot_id in enumerate(slot_config.slot_ids):
            report = reports.get(slot_id)
            slot_offset = OBS_V3_BASE_FEATURE_SIZE + slot_idx * OBS_V3_SLOT_FEATURE_SIZE

            if report is None:
                # Empty slot (`state is None`) - already zeros from initialization
                continue

            # Set blueprint index
            blueprint_indices[env_idx, slot_idx] = report.blueprint_index

            # is_active (1 dim) - always 1.0 for active slots
            obs[env_idx, slot_offset] = 1.0

            # Stage one-hot (10 dims) - use cached lookup table
            # STAGE_TO_INDEX maps sparse enum values (0-10, skip 5) to contiguous indices (0-9)
            stage_idx = STAGE_TO_INDEX[report.stage.value]
            obs[env_idx, slot_offset + 1:slot_offset + 11] = stage_table[stage_idx]

            # Current alpha (1 dim)
            obs[env_idx, slot_offset + 11] = report.metrics.current_alpha

            # Improvement (1 dim)
            contribution = report.metrics.counterfactual_contribution
            if contribution is None:
                contribution = report.metrics.improvement_since_stage_start
            contribution_norm = max(-1.0, min(contribution / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))
            obs[env_idx, slot_offset + 12] = contribution_norm

            # Contribution velocity (1 dim)
            velocity = report.metrics.contribution_velocity
            velocity_norm = max(-1.0, min(velocity / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))
            obs[env_idx, slot_offset + 13] = velocity_norm

            # Blend tempo epochs (1 dim)
            blend_tempo_norm = float(report.blend_tempo_epochs) / 12.0
            obs[env_idx, slot_offset + 14] = blend_tempo_norm

            # Alpha scaffolding (8 dims)
            obs[env_idx, slot_offset + 15] = report.alpha_target
            obs[env_idx, slot_offset + 16] = float(report.alpha_mode) / max(_ALPHA_MODE_MAX, 1)
            obs[env_idx, slot_offset + 17] = min(float(report.alpha_steps_total), max_epochs_den) / max_epochs_den
            obs[env_idx, slot_offset + 18] = min(float(report.alpha_steps_done), max_epochs_den) / max_epochs_den
            obs[env_idx, slot_offset + 19] = min(float(report.time_to_target), max_epochs_den) / max_epochs_den
            obs[env_idx, slot_offset + 20] = max(-1.0, min(report.alpha_velocity, 1.0))
            obs[env_idx, slot_offset + 21] = float(report.alpha_algorithm - _ALPHA_ALGO_MIN) / _ALPHA_ALGO_RANGE
            obs[env_idx, slot_offset + 22] = max(
                -1.0, min(report.metrics.interaction_sum / 10.0, 1.0)
            )

            # Telemetry merged (4 dims)
            if report.telemetry is not None:
                # gradient_norm: Apply symlog to compress high magnitudes (100+ → ~4.6)
                # Old: clipped to 10.0 then /10 - lost information for large gradients
                # New: symlog preserves relative ordering, divided by 7 for ~[0,1] range
                grad_norm_raw = report.telemetry.gradient_norm
                if grad_norm_raw is not None and math.isfinite(grad_norm_raw):
                    obs[env_idx, slot_offset + 23] = symlog(grad_norm_raw) / _SYMLOG_NORM
                else:
                    obs[env_idx, slot_offset + 23] = 0.0
                obs[env_idx, slot_offset + 24] = safe(report.telemetry.gradient_health, 1.0, max_val=1.0)
                obs[env_idx, slot_offset + 25] = 1.0 if report.telemetry.has_vanishing else 0.0
                obs[env_idx, slot_offset + 26] = 1.0 if report.telemetry.has_exploding else 0.0
            else:
                # Default values when telemetry not available
                obs[env_idx, slot_offset + 23] = 0.0
                obs[env_idx, slot_offset + 24] = 1.0  # Assume healthy
                obs[env_idx, slot_offset + 25] = 0.0
                obs[env_idx, slot_offset + 26] = 0.0

            # gradient_health_prev (1 dim)
            gradient_health_prev = env_state.gradient_health_prev.get(slot_id, 1.0)
            if not math.isfinite(gradient_health_prev):
                raise ValueError(
                    f"NaN/inf in gradient_health_prev for slot {slot_id}. "
                    f"Value: {gradient_health_prev}. This indicates a bug in the gradient "
                    f"telemetry pipeline - check vectorized.py where gradient_health is stored."
                )
            obs[env_idx, slot_offset + 27] = gradient_health_prev

            # epochs_in_stage_norm (1 dim)
            epochs_in_stage = report.metrics.epochs_in_current_stage
            obs[env_idx, slot_offset + 28] = min(float(epochs_in_stage), max_epochs_den) / max_epochs_den

            # counterfactual_fresh (1 dim)
            epochs_since_cf = env_state.epochs_since_counterfactual.get(slot_id, 0)
            obs[env_idx, slot_offset + 29] = DEFAULT_GAMMA ** epochs_since_cf

            # seed_age_norm (1 dim)
            seed_age = report.metrics.epochs_total
            obs[env_idx, slot_offset + 30] = min(float(seed_age), max_epochs_den) / max_epochs_den

    # Single H2D transfer at end (after all Python-loop filling is complete)
    return obs.to(device), blueprint_indices.to(device)


# =============================================================================
# Task Configuration - now imported from leyline
# =============================================================================
# TaskConfig is imported from esper.leyline at the top of this file.
# This keeps the hot path clean while providing cross-subsystem access.
