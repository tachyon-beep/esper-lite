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

from dataclasses import dataclass
import math
import os
import threading
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.slot_config import SlotConfig
# Phase 2 imports: Constants needed for Obs V3 feature extraction
from esper.leyline import NUM_OPS, NUM_STAGES, DEFAULT_GAMMA, NUM_BLUEPRINTS
# Stage schema for validation and one-hot encoding
# NOTE: Imported at module level since these are fast O(1) lookups used in hot path
from esper.leyline.stage_schema import (
    VALID_STAGE_VALUES as _VALID_STAGE_VALUES,
    NUM_STAGES as _NUM_STAGE_DIMS,
    stage_to_one_hot as _stage_to_one_hot,
    STAGE_TO_INDEX as _STAGE_TO_INDEX,
)

# HOT PATH: ONLY leyline imports allowed!

# Debug flag for paranoia stage validation (set ESPER_DEBUG_STAGE=1 to enable)
_DEBUG_STAGE_VALIDATION = os.environ.get("ESPER_DEBUG_STAGE", "").lower() in ("1", "true", "yes")

if TYPE_CHECKING:
    # Type hints only - not imported at runtime
    from esper.leyline import SeedStateReport


__all__ = [
    "safe",
    "obs_to_multislot_features",
    "batch_obs_to_features",
    "MULTISLOT_FEATURE_SIZE",
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
]


# =============================================================================
# Safe Value Conversion
# =============================================================================

def safe(v: float | int | None, default: float = 0.0, max_val: float = 100.0) -> float:
    """Safely convert value to float, handling None/inf/nan.

    Args:
        v: Value to convert (can be None, float, int, etc.)
        default: Default value for None/inf/nan
        max_val: Maximum absolute value (clips to [-max_val, max_val])

    Returns:
        Safe float value
    """
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return max(-max_val, min(float(v), max_val))


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
    host_stabilized: bool,
) -> torch.Tensor:
    """Extract base features (24 dims) for Obs V3.

    Base features:
    - Current epoch (1 dim): normalized to [0, 1] (max 150 epochs)
    - Current val_loss (1 dim): log-normalized
    - Current val_accuracy (1 dim): normalized to [0, 1]
    - Raw loss history (5 dims): log-normalized, left-padded
    - Raw accuracy history (5 dims): normalized to [0, 1], left-padded
    - Stage distribution (3 dims): num_training_norm, num_blending_norm, num_holding_norm
    - Host stabilized flag (1 dim): 0.0 or 1.0
    - Action feedback (7 dims): last_action_success (1) + last_action_op one-hot (6)

    Total: 24 base features

    Args:
        signal: TrainingSignals with metrics and history
        env_state: ParallelEnvState with action feedback
        num_training: Number of seeds in TRAINING stage
        num_blending: Number of seeds in BLENDING stage
        num_holding: Number of seeds in HOLDING stage
        host_stabilized: Whether host has stabilized (boolean)

    Returns:
        torch.Tensor with shape (24,)
    """
    # Current metrics (3 dims)
    # Normalize epoch to [0, 1] range using 150 as max (typical training length)
    MAX_EPOCHS_NORM = 150.0
    epoch_norm = float(signal.metrics.epoch) / MAX_EPOCHS_NORM
    # Loss normalization: log(1 + loss) / log(16)
    # Range: [0.0, 1.0] supporting loss values up to 15 before saturation
    # CIFAR-10 baseline 2.3 → 0.4311, TinyStories baseline 10.8 → 0.8954
    val_loss_norm = math.log(1 + signal.metrics.val_loss) / math.log(16)
    val_accuracy_norm = signal.metrics.val_accuracy / 100.0

    # Extract and normalize loss history (5 dims) - log-scale normalization
    # Same normalization as current val_loss: log(1 + loss) / log(16)
    loss_history_padded = _pad_history(signal.loss_history, 5)
    loss_history_norm = [math.log(1 + x) / math.log(16) for x in loss_history_padded]

    # Extract and normalize accuracy history (5 dims)
    acc_history_padded = _pad_history(signal.accuracy_history, 5)
    acc_history_norm = [x / 100.0 for x in acc_history_padded]

    # Stage distribution (3 dims) - normalize by max slots (3)
    num_training_norm = num_training / 3.0
    num_blending_norm = num_blending / 3.0
    num_holding_norm = num_holding / 3.0

    # Host stabilized flag (1 dim) - already boolean (0.0 or 1.0)
    host_stabilized_float = 1.0 if host_stabilized else 0.0

    # Action feedback (7 dims): last_action_success + last_action_op one-hot (6 dims)
    last_action_success = 1.0 if env_state.last_action_success else 0.0

    # Validate last_action_op is in valid range before one-hot encoding
    assert 0 <= env_state.last_action_op < NUM_OPS, \
        f"Invalid last_action_op: {env_state.last_action_op} (expected 0-{NUM_OPS-1})"

    last_op_one_hot = F.one_hot(
        torch.tensor(env_state.last_action_op),
        num_classes=NUM_OPS
    ).float()

    # Combine all base features (24 dims total)
    base_features = (
        [epoch_norm] +  # 1 dim - normalized to [0, 1]
        [val_loss_norm] +  # 1 dim
        [val_accuracy_norm] +  # 1 dim
        loss_history_norm +  # 5 dims
        acc_history_norm +   # 5 dims
        [num_training_norm, num_blending_norm, num_holding_norm] +  # 3 dims
        [host_stabilized_float] +  # 1 dim
        [last_action_success] +  # 1 dim
        last_op_one_hot.tolist()  # 6 dims
    )

    return torch.tensor(base_features, dtype=torch.float32)


def _extract_slot_features_v3(
    slot_report: Any,  # SeedStateReport
    env_state: Any,  # ParallelEnvState
    slot_id: str,
) -> torch.Tensor:
    """Extract per-slot features (30 dims) for Obs V3.

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

    Total: 30 dims

    Note: boost_received, upstream_alpha_sum, and downstream_alpha_sum from V2 are REMOVED in V3.

    Args:
        slot_report: SeedStateReport with current slot state
        env_state: ParallelEnvState with gradient health tracking
        slot_id: Slot identifier for looking up tracked state

    Returns:
        torch.Tensor with shape (30,)
    """
    # Stage one-hot encoding (10 dims)
    stage_val = slot_report.stage.value
    if _DEBUG_STAGE_VALIDATION:
        assert stage_val in _VALID_STAGE_VALUES, (
            f"Invalid stage value {stage_val} for slot {slot_id}; "
            f"valid values are {sorted(_VALID_STAGE_VALUES)}"
        )

    if stage_val in _VALID_STAGE_VALUES:
        stage_one_hot = _stage_to_one_hot(stage_val)
    else:
        # Fallback: all zeros (should not happen after Phase 0 validation)
        stage_one_hot = [0.0] * _NUM_STAGE_DIMS

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

    # Normalize alpha schedule steps (use 25 epochs as default max)
    max_epochs_den = 25.0
    alpha_steps_total_norm = min(float(slot_report.alpha_steps_total), max_epochs_den) / max_epochs_den
    alpha_steps_done_norm = min(float(slot_report.alpha_steps_done), max_epochs_den) / max_epochs_den
    time_to_target_norm = min(float(slot_report.time_to_target), max_epochs_den) / max_epochs_den

    alpha_velocity = max(-1.0, min(slot_report.alpha_velocity, 1.0))
    alpha_algorithm_norm = float(slot_report.alpha_algorithm - _ALPHA_ALGO_MIN) / _ALPHA_ALGO_RANGE
    interaction_sum_norm = min(slot_report.metrics.interaction_sum / 10.0, 1.0)

    # Telemetry merged (4 dims)
    # Access telemetry fields from slot_report.telemetry if available
    if slot_report.telemetry is not None:
        gradient_norm = safe(slot_report.telemetry.gradient_norm, 0.0, max_val=10.0) / 10.0
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

    # epochs_in_stage_norm (1 dim) - normalize to [0, 1] using max 25 epochs
    epochs_in_stage = slot_report.metrics.epochs_in_current_stage
    epochs_in_stage_norm = min(float(epochs_in_stage), max_epochs_den) / max_epochs_den

    # counterfactual_fresh (1 dim) - gamma-matched decay
    # DEFAULT_GAMMA ** epochs_since_counterfactual
    # With DEFAULT_GAMMA=0.995, signal stays >0.5 for ~138 epochs
    epochs_since_cf = env_state.epochs_since_counterfactual.get(slot_id, 0)
    counterfactual_fresh = DEFAULT_GAMMA ** epochs_since_cf

    # Combine all slot features (30 dims total)
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
        ]
    )

    return torch.tensor(slot_features, dtype=torch.float32)


# =============================================================================
# Vectorized Construction (Task 2d)
# =============================================================================

# Module-level one-hot table for stage encoding
_STAGE_ONE_HOT_TABLE = torch.eye(NUM_STAGES, dtype=torch.float32)

# Device-keyed cache to avoid per-step .to(device) allocations
_DEVICE_CACHE: dict[torch.device, torch.Tensor] = {}
_DEVICE_CACHE_LOCK = threading.Lock()


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


def _get_cached_table(device: torch.device) -> torch.Tensor:
    """Get stage one-hot table for device, caching to avoid repeated transfers.

    Thread-safe via double-checked locking pattern.
    """
    device = _normalize_device(device)
    if device not in _DEVICE_CACHE:
        with _DEVICE_CACHE_LOCK:
            if device not in _DEVICE_CACHE:  # Double-check after acquiring lock
                _DEVICE_CACHE[device] = _STAGE_ONE_HOT_TABLE.to(device)
    return _DEVICE_CACHE[device]


def _vectorized_one_hot(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Vectorized one-hot encoding with device-cached lookup table.

    Args:
        indices: Stage indices of shape [..., num_slots] where -1 marks inactive.
                 Supports arbitrary leading batch dimensions.
        device: Target device for output tensor.

    Returns:
        One-hot encoded tensor of shape [..., num_slots, NUM_STAGES].
        Inactive slots (index -1) are all zeros.
    """
    table = _get_cached_table(device)
    valid_mask = indices >= 0  # -1 marks inactive slots
    safe_indices = indices.clamp(min=0)
    one_hot = table[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive


# =============================================================================
# Multi-Slot Features (V4 - dynamic dimensions)
# =============================================================================

# Obs V3 Constants
# Base features: 24 dims (epoch, val_loss, val_accuracy, history_5 × 2, stage distribution,
#                         host_stabilized, action feedback: last_action_success + last_action_op one-hot)
# NOTE: Includes 7 action feedback dims (last_action_success + 6-dim one-hot for last_action_op)
BASE_FEATURE_SIZE = 24

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
# Total: 1 + 10 + 1 + 1 + 1 + 1 + 8 + 4 + 1 + 1 + 1 = 30 dims per slot (blueprint moved to embedding)
SLOT_FEATURE_SIZE = 30

# Obs V3 total for 3 slots: 24 base + 3 slots × 30 features = 114 dims (excludes blueprint embeddings)
# Blueprint embeddings (4 dims × 3 slots = 12) are added inside the network, making total network input 126
# NOTE: Default for 3-slot configuration. Use get_feature_size(slot_config) for dynamic slot counts.
MULTISLOT_FEATURE_SIZE = 114  # Obs V3: 24 + (30 × 3)

# Observation normalization bounds (kept local to avoid heavier imports on the HOT PATH)
_IMPROVEMENT_CLAMP_PCT_PTS: float = 10.0  # Clamp improvement to ±10 percentage points → [-1, 1]
_DEFAULT_MAX_EPOCHS_DEN: float = 25.0  # Fallback when obs['max_epochs'] not provided
_ALPHA_MODE_MAX: int = max(mode.value for mode in AlphaMode)
_ALPHA_ALGO_MIN: int = min(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_MAX: int = max(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_RANGE: int = max(_ALPHA_ALGO_MAX - _ALPHA_ALGO_MIN, 1)


def get_feature_size(slot_config: SlotConfig) -> int:
    """Feature size excluding blueprint embeddings (added by network).

    Obs V3 Breakdown:
        Base features:     24 dims (epoch, val_loss, val_accuracy,
                                    loss_history_5, accuracy_history_5,
                                    stage distribution, host_stabilized,
                                    action feedback: last_action_success + last_action_op one-hot)
        Per-slot features: 30 dims × num_slots (stage one-hot, alpha, gradient health,
                                                 contribution, telemetry, etc.)

    For 3 slots: 24 + (30 × 3) = 114 dims

    Note: Blueprint embeddings (4 dims × num_slots) are added inside the network
    via BlueprintEmbedding module, making total network input 114 + 12 = 126 for 3 slots.

    Args:
        slot_config: Slot configuration defining number of slots

    Returns:
        Total observation feature size (excluding blueprint embeddings)
    """
    BASE_FEATURES = 24  # Includes 7 action feedback dims
    SLOT_FEATURES = 30  # Per-slot, excluding blueprint (moved to embedding)
    return BASE_FEATURES + (SLOT_FEATURES * slot_config.num_slots)


# Blueprint string ID to index mapping (matches BlueprintAction enum in leyline.factored_actions)
# WARNING: Must stay synchronized with BlueprintAction enum values.
# P2-7 Design Decision: Duplicated for performance - pre-computed dict lookup in hot path
# is faster than enum operations. This module is on the HOT PATH (see module docstring).
_BLUEPRINT_TO_INDEX = {
    "noop": 0,
    "conv_light": 1,
    "attention": 2,
    "norm": 3,
    "depthwise": 4,
    "bottleneck": 5,
    "conv_small": 6,
    "conv_heavy": 7,
    "lora": 8,
    "lora_large": 9,
    "mlp_small": 10,
    "mlp": 11,
    "flex_attention": 12,
}
_NUM_BLUEPRINT_TYPES = 13

# Drift guard: Ensure mapping stays synchronized with NUM_BLUEPRINTS constant
assert len(_BLUEPRINT_TO_INDEX) == NUM_BLUEPRINTS, (
    f"Blueprint mapping drift: {len(_BLUEPRINT_TO_INDEX)} entries in _BLUEPRINT_TO_INDEX "
    f"vs NUM_BLUEPRINTS={NUM_BLUEPRINTS}"
)
# Ensure indices are contiguous 0..NUM_BLUEPRINTS-1
assert set(_BLUEPRINT_TO_INDEX.values()) == set(range(NUM_BLUEPRINTS)), (
    f"Blueprint indices must cover 0..{NUM_BLUEPRINTS-1}, "
    f"got {sorted(_BLUEPRINT_TO_INDEX.values())}"
)


def obs_to_multislot_features(
    obs: dict[str, Any],
    total_seeds: int = 0,
    max_seeds: int = 1,
    slot_config: SlotConfig | None = None,
) -> list[float]:
    """Extract features including per-slot state (dynamic dims based on slot count).

    Base features (23 dims) - training state without seed telemetry:
    - Timing: epoch, global_step (2)
    - Losses: train_loss, val_loss, loss_delta (3)
    - Accuracies: train_accuracy, val_accuracy, accuracy_delta (3)
    - Trends: plateau_epochs, best_val_accuracy, best_val_loss (3)
    - History: loss_history_5 (5), accuracy_history_5 (5)
    - Total params: total_params (1)
    - Resource management: seed_utilization (1)

    Per-slot features (39 dims each, schema v2 - one-hot stage + scaffolding):
    - is_active: 1.0 if seed active, 0.0 otherwise (1 dim)
    - stage_one_hot: categorical encoding via StageSchema (10 dims)
    - alpha: blending alpha (0.0-1.0)
    - improvement: counterfactual contribution delta (normalized to [-1, 1])
    - contribution_velocity: EMA of contribution delta (for trend detection, normalized to [-1, 1])
    - tempo: blend tempo epochs normalized (0-1)
    - alpha_target: controller target alpha (0.0-1.0)
    - alpha_mode: controller mode normalized to [0, 1] (AlphaMode enum)
    - alpha_steps_total: schedule length normalized to [0, 1]
    - alpha_steps_done: schedule progress normalized to [0, 1]
    - time_to_target: remaining controller steps normalized to [0, 1]
    - alpha_velocity: schedule velocity (clamped to [-1, 1])
    - alpha_algorithm: composition/gating mode normalized to [0, 1] (AlphaAlgorithm enum)
    - interaction_sum: total synergy with other seeds (normalized to [0, 1])
    - boost_received: strongest single interaction (normalized to [0, 1])
    - upstream_alpha_sum: alpha of seeds in earlier slots (normalized to [0, 1])
    - downstream_alpha_sum: alpha of seeds in later slots (normalized to [0, 1])
    - blueprint_id: one-hot encoding of blueprint type (13 dims)

    This keeps each slot's local state visible, while still giving Tamiyo
    a single flat observation vector that standard PPO implementations
    can consume without custom architecture changes.

    Feature Layout (for default 3-slot config, total 140 dims):
    [0-1]   Timing (epoch, global_step)
    [2-4]   Losses (train, val, delta)
    [5-7]   Accuracies (train, val, delta)
    [8-10]  Trends (plateau_epochs, best_val_acc, best_val_loss)
    [11-15] Loss history (5 values)
    [16-20] Accuracy history (5 values)
    [21]    Total params
    [22]    Seed utilization
    [23-61]  Slot 0 (1 is_active + 10 stage + 15 state + 13 blueprint)
    [62-100] Slot 1 (1 is_active + 10 stage + 15 state + 13 blueprint)
    [101-139] Slot 2 (1 is_active + 10 stage + 15 state + 13 blueprint)

    Args:
        obs: Observation dictionary with optional 'slots' key
        total_seeds: Current total seeds across all slots (default 0)
        max_seeds: Maximum allowed seeds (default 1)
        slot_config: Slot configuration (default: 3-slot config)

    Returns:
        List of floats: 23 base + num_slots * 39 slot features

    Note (P2-9 Design Decision):
        Returns list[float] instead of torch.Tensor for flexibility during
        construction. The caller (signals_to_features) may append telemetry
        features. Batched conversion to tensor happens once at line 1502 in
        vectorized.py, which is more efficient than converting each env
        separately. Cost is ~5-10µs per step, negligible vs forward pass.
    """
    if slot_config is None:
        slot_config = SlotConfig.default()
    # Compute seed utilization
    seed_utilization = total_seeds / max_seeds if max_seeds > 0 else 0.0

    # Base features (23 dims) - simplified from V3, no seed state
    base = [
        float(obs['epoch']),
        float(obs['global_step']),
        safe(obs['train_loss'], 10.0),
        safe(obs['val_loss'], 10.0),
        safe(obs['loss_delta'], 0.0),
        obs['train_accuracy'],
        obs['val_accuracy'],
        safe(obs['accuracy_delta'], 0.0),
        float(obs['plateau_epochs']),
        obs['best_val_accuracy'],
        safe(obs['best_val_loss'], 10.0),
        *[safe(v, 10.0) for v in obs['loss_history_5']],
        *obs['accuracy_history_5'],
        float(obs.get('total_params', 0)),  # Total model params
        float(seed_utilization),  # New: resource management
    ]

    # Slot field access patterns (P3 Audit 2025-12-25):
    # REQUIRED (via direct access): obs["slots"], slots[slot_id]
    # OPTIONAL (via .get() with defaults):
    #   - alpha: 0.0 for inactive slots (safe default)
    #   - alpha_target: defaults to current alpha (no transition)
    #   - alpha_mode: HOLD (AlphaMode.HOLD.value) for no scheduled transition
    #   - alpha_steps_total: 0 for no transition scheduled
    #   - alpha_steps_done: 0 for no transition progress
    #   - time_to_target: 0 for no transition
    #   - alpha_velocity: 0.0 for static alpha
    #   - alpha_algorithm: defaults to AlphaAlgorithm.ADD (min value) for composition mode
    #   - interaction_sum: 0.0 for no topology/scaffolding
    #   - boost_received: 0.0 for no topology/scaffolding
    #   - upstream_alpha_sum: 0.0 for no topology/scaffolding
    #   - downstream_alpha_sum: 0.0 for no topology/scaffolding
    #   - stage: 0 for inactive (NOTE: active slots MUST have explicit stage from SeedStage enum)
    #   - improvement: 0.0 for no contribution history
    #   - contribution_velocity: 0.0 for no velocity history
    #   - is_active: 0 for inactive slot
    #   - blend_tempo_epochs: 5 default (safe fallback tempo)
    #   - blueprint_id: None for no blueprint assignment
    #
    # DESIGN NOTE: All .get() defaults are safe for inactive/new slots. Active slots
    # with missing required fields (stage, is_active=1) would indicate a bug in the
    # slot state construction in kasmina/tamiyo, not a feature extraction issue.

    # Per-slot features (25 dims per slot, num_slots determined by slot_config)
    # 12 state features + 13 blueprint one-hot
    max_epochs_den = max(float(obs.get("max_epochs") or _DEFAULT_MAX_EPOCHS_DEN), 1.0)
    slot_features = []
    slots = obs["slots"]  # KeyError if missing 'slots'
    for slot_id in slot_config.slot_ids:
        slot = slots[slot_id]  # KeyError if slot_id missing
        alpha = safe(slot.get("alpha", 0.0), 0.0, max_val=1.0)
        alpha = max(0.0, alpha)  # safe() clamps symmetrically; alpha should be >= 0
        alpha_target = safe(slot.get("alpha_target", alpha), alpha, max_val=1.0)
        alpha_target = max(0.0, alpha_target)

        alpha_mode_raw = int(slot.get("alpha_mode", AlphaMode.HOLD.value) or AlphaMode.HOLD.value)
        alpha_mode_raw = max(0, min(alpha_mode_raw, _ALPHA_MODE_MAX))
        alpha_mode_norm = alpha_mode_raw / max(_ALPHA_MODE_MAX, 1)

        alpha_steps_total = max(0.0, float(slot.get("alpha_steps_total", 0) or 0))
        alpha_steps_done = max(0.0, float(slot.get("alpha_steps_done", 0) or 0))
        time_to_target = max(0.0, float(slot.get("time_to_target", 0) or 0))

        alpha_steps_total_norm = min(alpha_steps_total, max_epochs_den) / max_epochs_den
        alpha_steps_done_norm = min(alpha_steps_done, max_epochs_den) / max_epochs_den
        time_to_target_norm = min(time_to_target, max_epochs_den) / max_epochs_den

        alpha_velocity = safe(slot.get("alpha_velocity", 0.0), 0.0, max_val=1.0)

        alpha_algorithm_raw = int(slot.get("alpha_algorithm", _ALPHA_ALGO_MIN) or _ALPHA_ALGO_MIN)
        alpha_algorithm_raw = max(_ALPHA_ALGO_MIN, min(alpha_algorithm_raw, _ALPHA_ALGO_MAX))
        alpha_algorithm_norm = (alpha_algorithm_raw - _ALPHA_ALGO_MIN) / _ALPHA_ALGO_RANGE

        # Interaction and topology features (scaffolding support)
        interaction_sum = safe(slot.get("interaction_sum", 0.0), 0.0, max_val=10.0) / 10.0
        boost_received = safe(slot.get("boost_received", 0.0), 0.0, max_val=5.0) / 5.0
        upstream_alpha = safe(slot.get("upstream_alpha_sum", 0.0), 0.0, max_val=3.0) / 3.0
        downstream_alpha = safe(slot.get("downstream_alpha_sum", 0.0), 0.0, max_val=3.0) / 3.0

        # Extract and optionally validate stage value
        stage_val = int(slot.get('stage', 0))
        if _DEBUG_STAGE_VALIDATION:
            assert stage_val in _VALID_STAGE_VALUES, (
                f"Invalid stage value {stage_val} for slot {slot_id}; "
                f"valid values are {sorted(_VALID_STAGE_VALUES)}"
            )

        # Stage one-hot encoding (10 dims)
        if stage_val in _VALID_STAGE_VALUES:
            stage_one_hot = _stage_to_one_hot(stage_val)
        else:
            # Fallback: all zeros (should not happen after Phase 0 validation)
            stage_one_hot = [0.0] * _NUM_STAGE_DIMS

        # Compute normalized contribution and velocity (raw, not lookahead)
        contribution = safe(slot.get("improvement", 0.0), 0.0, max_val=_IMPROVEMENT_CLAMP_PCT_PTS)
        contribution_velocity = safe(slot.get("contribution_velocity", 0.0), 0.0, max_val=_IMPROVEMENT_CLAMP_PCT_PTS)

        # is_active (1 dim)
        slot_features.append(float(slot.get('is_active', 0)))
        # Stage one-hot (10 dims)
        slot_features.extend(stage_one_hot)
        # Other state features (15 dims)
        slot_features.extend([
            alpha,
            contribution / _IMPROVEMENT_CLAMP_PCT_PTS,
            contribution_velocity / _IMPROVEMENT_CLAMP_PCT_PTS,  # Raw velocity, not fossilize_value
            # Tempo normalized to 0-1 range (max is ~12 epochs)
            float(slot.get('blend_tempo_epochs', 5)) / 12.0,
            alpha_target,
            alpha_mode_norm,
            alpha_steps_total_norm,
            alpha_steps_done_norm,
            time_to_target_norm,
            alpha_velocity,
            alpha_algorithm_norm,
            # Scaffolding features
            interaction_sum,
            boost_received,
            upstream_alpha,
            downstream_alpha,
        ])
        # Blueprint one-hot (13 dims)
        blueprint_id = slot.get('blueprint_id', None)
        blueprint_idx = _BLUEPRINT_TO_INDEX.get(blueprint_id, -1) if blueprint_id else -1
        blueprint_one_hot = [0.0] * _NUM_BLUEPRINT_TYPES
        if 0 <= blueprint_idx < _NUM_BLUEPRINT_TYPES:
            blueprint_one_hot[blueprint_idx] = 1.0
        slot_features.extend(blueprint_one_hot)

    return base + slot_features


def batch_obs_to_features(
    batch_signals: list[Any],  # TrainingSignals
    batch_slot_reports: list[dict[str, Any]],  # dict[str, SeedStateReport]
    batch_env_states: list[Any],  # list[ParallelEnvState]
    slot_config: SlotConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for batch of environments (Obs V3).

    Returns tuple of (obs, blueprint_indices):
    - obs: [batch, obs_dim] - base features + slot features (NO blueprint one-hot)
    - blueprint_indices: [batch, num_slots] - int64 indices for nn.Embedding lookup

    Feature breakdown:
    - Base features: 24 dims (epoch, loss, accuracy, raw history, stage counts, host stabilized)
    - Action feedback: 7 dims (last_action_success + last_action_op one-hot) - INCLUDED in base
    - Per-slot features: 30 dims × num_slots (stage, alpha, gradient health, contribution, etc.)

    For 3 slots: 24 + (30 × 3) = 114 dims

    Note: The spec mentions 121 dims (24 + 7 + 90) suggesting action feedback is separate.
    However, _extract_base_features_v3() returns 24 dims which INCLUDES the 7 action feedback
    dims (see lines 120-121). Therefore, the actual dimension is 24 + 90 = 114 for 3 slots.

    Args:
        batch_signals: List of TrainingSignals with metrics and history
        batch_slot_reports: List of dict[str, SeedStateReport] for each env
        batch_env_states: List of ParallelEnvState with action feedback and tracking
        slot_config: Slot configuration defining num_slots and slot_ids
        device: Target device for output tensors

    Returns:
        obs: [batch, obs_dim] - observation features (base + slots, no blueprint)
        blueprint_indices: [batch, num_slots] - blueprint indices for embedding (int64)
    """
    n_envs = len(batch_signals)
    num_slots = slot_config.num_slots

    # Calculate stage distribution for each environment
    stage_distributions = []
    for reports in batch_slot_reports:
        num_training = 0
        num_blending = 0
        num_holding = 0
        for report in reports.values():
            stage_name = report.stage.name
            if stage_name == "TRAINING":
                num_training += 1
            elif stage_name == "BLENDING":
                num_blending += 1
            elif stage_name in ("HOLDING", "GRAFTED", "FOSSILIZED"):
                num_holding += 1
        stage_distributions.append((num_training, num_blending, num_holding))

    # Extract base and slot features for each environment
    all_features = []
    for env_idx in range(n_envs):
        signal = batch_signals[env_idx]
        env_state = batch_env_states[env_idx]
        reports = batch_slot_reports[env_idx]
        num_training, num_blending, num_holding = stage_distributions[env_idx]

        # TODO: Determine host_stabilized flag (placeholder for now)
        # This should be set based on some criteria like plateau_epochs threshold
        host_stabilized = False

        # Extract base features (24 dims)
        base_features = _extract_base_features_v3(
            signal=signal,
            env_state=env_state,
            num_training=num_training,
            num_blending=num_blending,
            num_holding=num_holding,
            host_stabilized=host_stabilized,
        )

        # Extract slot features (30 dims per slot)
        slot_features_list = []
        for slot_id in slot_config.slot_ids:
            report = reports.get(slot_id)
            if report is not None:
                # Active slot - extract features
                slot_features = _extract_slot_features_v3(
                    slot_report=report,
                    env_state=env_state,
                    slot_id=slot_id,
                )
            else:
                # Inactive slot - all zeros (30 dims)
                slot_features = torch.zeros(30, dtype=torch.float32)
            slot_features_list.append(slot_features)

        # Concatenate base + all slot features
        env_features = torch.cat([base_features] + slot_features_list, dim=0)
        all_features.append(env_features)

    # Stack all environments into batch
    obs = torch.stack(all_features, dim=0).to(device)

    # Extract blueprint indices (CRITICAL: use np.int64 for nn.Embedding compatibility)
    bp_indices = np.zeros((n_envs, num_slots), dtype=np.int64)
    for env_idx, reports in enumerate(batch_slot_reports):
        for slot_idx, slot_id in enumerate(slot_config.slot_ids):
            report = reports.get(slot_id)
            if report is not None:
                bp_indices[env_idx, slot_idx] = report.blueprint_index
            else:
                bp_indices[env_idx, slot_idx] = -1  # Inactive slot

    blueprint_indices = torch.from_numpy(bp_indices).to(device)
    # tensor is already int64 from numpy dtype

    return obs, blueprint_indices


# =============================================================================
# Task Configuration and Observation Normalization (Phase 2)
# =============================================================================


@dataclass(slots=True)
class TaskConfig:
    """Task-specific configuration for observation normalization."""

    task_type: str  # "classification" or "lm"
    topology: str   # "cnn" or "transformer"
    baseline_loss: float  # Random init loss
    target_loss: float    # Achievable loss
    typical_loss_delta_std: float
    max_epochs: int
    max_steps: int = 10000
    train_to_blend_fraction: float = 0.1  # Fraction of max_epochs to stay in TRAINING before blending
    blending_steps: int = 5  # Steps for alpha ramp during blending

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss

    @staticmethod
    def for_cifar10() -> "TaskConfig":
        return TaskConfig(
            task_type="classification",
            topology="cnn",
            baseline_loss=2.3,  # ln(10)
            target_loss=0.3,
            typical_loss_delta_std=0.05,
            max_epochs=25,
            max_steps=10000,
        )

    @staticmethod
    def for_tinystories() -> "TaskConfig":
        return TaskConfig(
            task_type="lm",
            topology="transformer",
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,
            typical_loss_delta_std=0.15,
            max_epochs=50,
            max_steps=50000,
        )