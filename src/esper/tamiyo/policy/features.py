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
from typing import TYPE_CHECKING

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.slot_config import SlotConfig

# HOT PATH: ONLY leyline imports allowed!

if TYPE_CHECKING:
    # Type hints only - not imported at runtime
    pass


__all__ = [
    "safe",
    "obs_to_multislot_features",
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

# Per-slot features:
# 12 state (is_active, stage, alpha, improvement, tempo, alpha_target, alpha_mode,
#          alpha_steps_total, alpha_steps_done, time_to_target, alpha_velocity, alpha_algorithm)
# + 13 blueprint one-hot
SLOT_FEATURE_SIZE = 25

# Feature size (with telemetry off): 23 base + 3 slots * 25 features per slot = 98
# With telemetry on: + 3 slots * SeedTelemetry.feature_dim() (17) = 149 total
# NOTE: Default for 3-slot configuration. Use get_feature_size(slot_config) for dynamic slot counts.
MULTISLOT_FEATURE_SIZE = 98

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
    - Resource management: seed_utilization (1) [new - for resource awareness]

    Per-slot features (25 dims each):
    - is_active: 1.0 if seed active, 0.0 otherwise
    - stage: seed lifecycle stage (SeedStage enum int, 0-10)
    - alpha: blending alpha (0.0-1.0)
    - improvement: counterfactual contribution delta (normalized to [-1, 1])
    - tempo: blend tempo epochs normalized (0-1)
    - alpha_target: controller target alpha (0.0-1.0)
    - alpha_mode: controller mode normalized to [0, 1] (AlphaMode enum)
    - alpha_steps_total: schedule length normalized to [0, 1]
    - alpha_steps_done: schedule progress normalized to [0, 1]
    - time_to_target: remaining controller steps normalized to [0, 1]
    - alpha_velocity: schedule velocity (clamped to [-1, 1])
    - alpha_algorithm: composition/gating mode normalized to [0, 1] (AlphaAlgorithm enum)
    - blueprint_id: one-hot encoding of blueprint type (13 dims)

    This keeps each slot's local state visible, while still giving Tamiyo
    a single flat observation vector that standard PPO implementations
    can consume without custom architecture changes.

    Feature Layout (for default 3-slot config):
    [0-1]   Timing (epoch, global_step)
    [2-4]   Losses (train, val, delta)
    [5-7]   Accuracies (train, val, delta)
    [8-10]  Trends (plateau_epochs, best_val_acc, best_val_loss)
    [11-15] Loss history (5 values)
    [16-20] Accuracy history (5 values)
    [21]    Total params
    [22]    Seed utilization
    [23-47] Slot 0 (12 state + blueprint[13])
    [48-72] Slot 1 (12 state + blueprint[13])
    [73-97] Slot 2 (12 state + blueprint[13])

    Args:
        obs: Observation dictionary with optional 'slots' key
        total_seeds: Current total seeds across all slots (default 0)
        max_seeds: Maximum allowed seeds (default 1)
        slot_config: Slot configuration (default: 3-slot config)

    Returns:
        List of floats: 23 base + num_slots * 25 slot features

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

    # Per-slot features (25 dims per slot, num_slots determined by slot_config)
    # 12 state features + 13 blueprint one-hot
    max_epochs_den = max(float(obs.get("max_epochs") or _DEFAULT_MAX_EPOCHS_DEN), 1.0)
    slot_features = []
    for slot_id in slot_config.slot_ids:
        slot = obs.get('slots', {}).get(slot_id, {})
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

        # State features (12 dims)
        slot_features.extend([
            float(slot.get('is_active', 0)),
            float(slot.get('stage', 0)),
            alpha,
            safe(slot.get("improvement", 0.0), 0.0, max_val=_IMPROVEMENT_CLAMP_PCT_PTS) / _IMPROVEMENT_CLAMP_PCT_PTS,
            # Tempo normalized to 0-1 range (max is ~12 epochs)
            float(slot.get('blend_tempo_epochs', 5)) / 12.0,
            alpha_target,
            alpha_mode_norm,
            alpha_steps_total_norm,
            alpha_steps_done_norm,
            time_to_target_norm,
            alpha_velocity,
            alpha_algorithm_norm,
        ])
        # Blueprint one-hot (13 dims)
        blueprint_id = slot.get('blueprint_id', None)
        blueprint_idx = _BLUEPRINT_TO_INDEX.get(blueprint_id, -1) if blueprint_id else -1
        blueprint_one_hot = [0.0] * _NUM_BLUEPRINT_TYPES
        if 0 <= blueprint_idx < _NUM_BLUEPRINT_TYPES:
            blueprint_one_hot[blueprint_idx] = 1.0
        slot_features.extend(blueprint_one_hot)

    return base + slot_features


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
