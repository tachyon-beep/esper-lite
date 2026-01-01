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
import threading
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.slot_config import SlotConfig
# Phase 2 imports: Constants needed for Obs V3 feature extraction
from esper.leyline import (
    DEFAULT_GAMMA,
    NUM_OPS,
    NUM_STAGES,
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_NON_BLUEPRINT_DIM,
    OBS_V3_SLOT_FEATURE_SIZE,
)
# Stage schema for validation and one-hot encoding
# NOTE: Imported at module level since these are fast O(1) lookups used in hot path
from esper.leyline.stage_schema import (
    VALID_STAGE_VALUES as _VALID_STAGE_VALUES,
    NUM_STAGES as _NUM_STAGE_DIMS,
    stage_to_one_hot as _stage_to_one_hot,
)

# HOT PATH: ONLY leyline imports allowed!

if TYPE_CHECKING:
    # Type hints only - not imported at runtime
    pass


__all__ = [
    "safe",
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

    Handles Python floats, numpy scalars, and 0-dim torch tensors.
    Raises TypeError for non-numeric types to avoid masking contract violations.

    Args:
        v: Value to convert (can be None, float, int, numpy scalar, 0-dim tensor)
        default: Default value for None/inf/nan
        max_val: Maximum absolute value (clips to [-max_val, max_val])

    Returns:
        Safe float value

    Raises:
        TypeError: If v is not a numeric type that can be converted to float
    """
    if v is None:
        return default
    try:
        v_float = float(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"safe() expected numeric, got {type(v)!r}") from exc
    if not math.isfinite(v_float):
        return default
    return max(-max_val, min(v_float, max_val))


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
# Total: 1 + 10 + 1 + 1 + 1 + 1 + 8 + 4 + 1 + 1 + 1 = 30 dims per slot (blueprint moved to embedding)
SLOT_FEATURE_SIZE = OBS_V3_SLOT_FEATURE_SIZE

# Obs V3 total for 3 slots: 23 base + 3 slots × 30 features = 113 dims (excludes blueprint embeddings)
# Blueprint embeddings (4 dims × 3 slots = 12) are added inside the network, making total network input 125
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
        Per-slot features: 30 dims × num_slots (stage one-hot, alpha, gradient health,
                                                 contribution, telemetry, etc.)

    For 3 slots: 23 + (30 × 3) = 113 dims

    Note: Blueprint embeddings (4 dims × num_slots) are added inside the network
    via BlueprintEmbedding module, making total network input 113 + 12 = 125 for 3 slots.

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

    Returns tuple of (obs, blueprint_indices):
    - obs: [batch, obs_dim] - base features + slot features (NO blueprint one-hot)
    - blueprint_indices: [batch, num_slots] - int64 indices for nn.Embedding lookup

    Feature breakdown:
    - Base features: 23 dims (epoch, loss, accuracy, raw history, stage counts)
    - Action feedback: 7 dims (last_action_success + last_action_op one-hot) - INCLUDED in base
    - Per-slot features: 30 dims × num_slots (stage, alpha, gradient health, contribution, etc.)

    For 3 slots: 23 + (30 × 3) = 113 dims

    Note: Removed host_stabilized - Tamiyo learns stability from raw telemetry.

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
            elif stage_name in ("HOLDING", "FOSSILIZED"):
                num_holding += 1
        stage_distributions.append((num_training, num_blending, num_holding))

    # Extract base and slot features for each environment
    all_features = []
    for env_idx in range(n_envs):
        signal = batch_signals[env_idx]
        env_state = batch_env_states[env_idx]
        reports = batch_slot_reports[env_idx]
        num_training, num_blending, num_holding = stage_distributions[env_idx]

        # Extract base features (23 dims)
        base_features = _extract_base_features_v3(
            signal=signal,
            env_state=env_state,
            num_training=num_training,
            num_blending=num_blending,
            num_holding=num_holding,
            slot_config=slot_config,
            max_epochs=max_epochs,
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
                    max_epochs=max_epochs,
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
