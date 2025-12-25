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
from typing import TYPE_CHECKING

import torch

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.slot_config import SlotConfig

# HOT PATH: ONLY leyline imports allowed!

# Debug flag for paranoia stage validation (set ESPER_DEBUG_STAGE=1 to enable)
_DEBUG_STAGE_VALIDATION = os.environ.get("ESPER_DEBUG_STAGE", "").lower() in ("1", "true", "yes")

# Import stage schema for validation and one-hot encoding
# NOTE: Imported at module level since these are fast O(1) lookups used in hot path
from esper.leyline.stage_schema import (
    VALID_STAGE_VALUES as _VALID_STAGE_VALUES,
    NUM_STAGES as _NUM_STAGE_DIMS,
    stage_to_one_hot as _stage_to_one_hot,
    STAGE_TO_INDEX as _STAGE_TO_INDEX,
)

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

def safe(v, default: float = 0.0, max_val: float = 100.0) -> float:
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
# Multi-Slot Features (V4 - dynamic dimensions)
# =============================================================================

# Base features (training state without per-slot features)
BASE_FEATURE_SIZE = 23

# Per-slot features (schema v2 - one-hot stage + scaffolding):
# 1 is_active
# + 10 stage one-hot (SeedStage categorical encoding via StageSchema)
# + 15 state (alpha, contribution, velocity, tempo, alpha_target, alpha_mode,
#            alpha_steps_total, alpha_steps_done, time_to_target, alpha_velocity, alpha_algorithm,
#            interaction_sum, boost_received, upstream_alpha, downstream_alpha)
# + 13 blueprint one-hot
# Total: 1 + 10 + 15 + 13 = 39 dims per slot
SLOT_FEATURE_SIZE = 39

# Feature size (with telemetry off): 23 base + 3 slots * 39 features per slot = 140
# With telemetry on: + 3 slots * SeedTelemetry.feature_dim() (26) = 218 total
# NOTE: Default for 3-slot configuration. Use get_feature_size(slot_config) for dynamic slot counts.
MULTISLOT_FEATURE_SIZE = 140

# Observation normalization bounds (kept local to avoid heavier imports on the HOT PATH)
_IMPROVEMENT_CLAMP_PCT_PTS: float = 10.0  # Clamp improvement to ±10 percentage points → [-1, 1]
_DEFAULT_MAX_EPOCHS_DEN: float = 25.0  # Fallback when obs['max_epochs'] not provided
_ALPHA_MODE_MAX: int = max(mode.value for mode in AlphaMode)
_ALPHA_ALGO_MIN: int = min(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_MAX: int = max(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_RANGE: int = max(_ALPHA_ALGO_MAX - _ALPHA_ALGO_MIN, 1)


def get_feature_size(slot_config: SlotConfig) -> int:
    """Get feature size for given slot configuration.

    Args:
        slot_config: Slot configuration defining number of slots.

    Returns:
        Total feature size: BASE_FEATURE_SIZE + num_slots * SLOT_FEATURE_SIZE
    """
    return BASE_FEATURE_SIZE + slot_config.num_slots * SLOT_FEATURE_SIZE


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


def obs_to_multislot_features(
    obs: dict,
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
    batch_signals: list,
    batch_slot_reports: list[dict[str, "SeedStateReport"]],
    use_telemetry: bool,
    max_epochs: int,
    total_params: list[int],
    total_seeds: list[int],
    max_seeds: int,
    slot_config: SlotConfig,
    device: torch.device,
) -> torch.Tensor:
    """Consolidated tensor-driven feature extraction for all environments.
    
    Replaces dict-based loops with vectorized torch operations.
    """
    n_envs = len(batch_signals)
    num_slots = slot_config.num_slots
    state_dim = get_feature_size(slot_config)
    
    # Pre-allocate feature tensor
    features = torch.zeros((n_envs, state_dim), device=device)
    
    # 1. Extract Base Features (0-22)
    # [0-1] Timing (epoch, global_step)
    epochs = torch.tensor([s.metrics.epoch for s in batch_signals], device=device, dtype=torch.float32)
    steps = torch.tensor([s.metrics.global_step for s in batch_signals], device=device, dtype=torch.float32)
    features[:, 0] = epochs
    features[:, 1] = steps
    
    # [2-4] Losses (train, val, delta)
    features[:, 2] = torch.tensor([s.metrics.train_loss for s in batch_signals], device=device).clamp(-10, 10)
    features[:, 3] = torch.tensor([s.metrics.val_loss for s in batch_signals], device=device).clamp(-10, 10)
    features[:, 4] = torch.tensor([s.metrics.loss_delta for s in batch_signals], device=device).clamp(-10, 10)
    
    # [5-7] Accuracies (train, val, delta)
    features[:, 5] = torch.tensor([s.metrics.train_accuracy for s in batch_signals], device=device)
    features[:, 6] = torch.tensor([s.metrics.val_accuracy for s in batch_signals], device=device)
    features[:, 7] = torch.tensor([s.metrics.accuracy_delta for s in batch_signals], device=device).clamp(-10, 10)
    
    # [8-10] Trends (plateau_epochs, best_val_acc, best_val_loss)
    features[:, 8] = torch.tensor([s.metrics.plateau_epochs for s in batch_signals], device=device, dtype=torch.float32)
    features[:, 9] = torch.tensor([s.metrics.best_val_accuracy for s in batch_signals], device=device)
    features[:, 10] = torch.tensor([s.metrics.best_val_loss for s in batch_signals], device=device).clamp(-10, 10)
    
    # [11-15] Loss history (5 values)
    for i, s in enumerate(batch_signals):
        hist = list(s.loss_history[-5:])
        while len(hist) < 5: hist.insert(0, 0.0)
        features[i, 11:16] = torch.tensor(hist, device=device).clamp(-10, 10)
        
    # [16-20] Accuracy history (5 values)
    for i, s in enumerate(batch_signals):
        hist = list(s.accuracy_history[-5:])
        while len(hist) < 5: hist.insert(0, 0.0)
        features[i, 16:21] = torch.tensor(hist, device=device)
        
    # [21] Total params
    features[:, 21] = torch.tensor(total_params, device=device, dtype=torch.float32)
    
    # [22] Seed utilization
    features[:, 22] = torch.tensor(total_seeds, device=device, dtype=torch.float32) / max(max_seeds, 1)
    
    # 2. Extract Slot Features (23 + num_slots * 39)
    # Per-slot layout: [is_active(1), stage_one_hot(10), state(15), blueprint(13)] = 39 dims
    max_epochs_den = float(max(max_epochs, 1))

    for slot_idx, slot_id in enumerate(slot_config.slot_ids):
        offset = BASE_FEATURE_SIZE + slot_idx * SLOT_FEATURE_SIZE

        # Slot features: 1 is_active + 10 stage one-hot + 15 state + 13 blueprint = 39 dims
        for env_idx in range(n_envs):
            report = batch_slot_reports[env_idx].get(slot_id)
            if report:
                contribution = report.metrics.counterfactual_contribution
                if contribution is None:
                    contribution = report.metrics.improvement_since_stage_start

                # Get contribution velocity (raw, not lookahead)
                velocity = report.metrics.contribution_velocity
                velocity_norm = max(-1.0, min(velocity / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))

                # Paranoia check for valid stage value (debug-only)
                stage_val = report.stage.value
                if _DEBUG_STAGE_VALIDATION:
                    assert stage_val in _VALID_STAGE_VALUES, (
                        f"Invalid stage value {stage_val} for slot {slot_id} env {env_idx}"
                    )

                # is_active (offset + 0)
                features[env_idx, offset] = 1.0

                # Stage one-hot (offset + 1 to offset + 10, 10 dims)
                if stage_val in _STAGE_TO_INDEX:
                    stage_idx = _STAGE_TO_INDEX[stage_val]
                    features[env_idx, offset + 1 + stage_idx] = 1.0
                # else: all zeros (already initialized)

                # Other state features (offset + 11 to offset + 25, 15 dims)
                features[env_idx, offset + 11] = report.metrics.current_alpha
                features[env_idx, offset + 12] = max(-1.0, min(contribution / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))
                features[env_idx, offset + 13] = velocity_norm  # Raw velocity
                features[env_idx, offset + 14] = float(report.blend_tempo_epochs) / 12.0
                features[env_idx, offset + 15] = report.alpha_target
                features[env_idx, offset + 16] = float(report.alpha_mode) / _ALPHA_MODE_MAX
                features[env_idx, offset + 17] = min(float(report.alpha_steps_total), max_epochs_den) / max_epochs_den
                features[env_idx, offset + 18] = min(float(report.alpha_steps_done), max_epochs_den) / max_epochs_den
                features[env_idx, offset + 19] = min(float(report.time_to_target), max_epochs_den) / max_epochs_den
                features[env_idx, offset + 20] = max(-1.0, min(report.alpha_velocity, 1.0))
                features[env_idx, offset + 21] = float(report.alpha_algorithm - _ALPHA_ALGO_MIN) / _ALPHA_ALGO_RANGE

                # Scaffolding features (offset + 22 to offset + 25, 4 dims)
                features[env_idx, offset + 22] = min(report.metrics.interaction_sum / 10.0, 1.0)
                features[env_idx, offset + 23] = min(report.metrics.boost_received / 5.0, 1.0)
                features[env_idx, offset + 24] = min(report.metrics.upstream_alpha_sum / 3.0, 1.0)
                features[env_idx, offset + 25] = min(report.metrics.downstream_alpha_sum / 3.0, 1.0)

                # Blueprint one-hot (offset + 26 to offset + 38, 13 dims) - shifted by 4
                bp_idx = _BLUEPRINT_TO_INDEX.get(report.blueprint_id, -1)
                if 0 <= bp_idx < _NUM_BLUEPRINT_TYPES:
                    features[env_idx, offset + 26 + bp_idx] = 1.0
            else:
                # Slot inactive - already zeros from initialization except defaults
                features[env_idx, offset + 21] = float(AlphaAlgorithm.ADD.value - _ALPHA_ALGO_MIN) / _ALPHA_ALGO_RANGE
                features[env_idx, offset + 14] = 5.0 / 12.0  # Default tempo

    # 3. Telemetry Features (Optional)
    if use_telemetry:
        # Append telemetry features from reports
        from esper.leyline import SeedTelemetry
        tele_dim = SeedTelemetry.feature_dim()
        
        tele_features_list = []
        for env_idx in range(n_envs):
            env_tele = []
            for slot_id in slot_config.slot_ids:
                report = batch_slot_reports[env_idx].get(slot_id)
                if report and report.telemetry:
                    env_tele.extend(report.telemetry.to_features())
                else:
                    env_tele.extend([0.0] * tele_dim)
            tele_features_list.append(env_tele)
        
        tele_tensor = torch.tensor(tele_features_list, device=device, dtype=torch.float32)
        features = torch.cat([features, tele_tensor], dim=1)

    return features


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