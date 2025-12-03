"""Simic Features - HOT PATH Feature Extraction

CRITICAL: This module is on the HOT PATH for vectorized training.
ONLY import from leyline. NO imports from kasmina, tamiyo, or nissa!

This module extracts features from observations for RL training.
It must be FAST and have minimal dependencies to avoid bottlenecks
in the vectorized PPO training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

# HOT PATH: ONLY leyline imports allowed!
from esper.leyline import TensorSchema, TENSOR_SCHEMA_SIZE

if TYPE_CHECKING:
    # Type hints only - not imported at runtime
    from typing import Any


__all__ = [
    "safe",
    "obs_to_base_features",
    "compute_action_mask",
    "MIN_CULL_AGE",
    "FULL_EVALUATION_AGE",
    "MIN_GERMINATE_EPOCH",
    "MIN_PLATEAU_TO_GERMINATE",
    "TaskConfig",
    "normalize_observation",
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
# Base Features (V1 - 27 dimensions)
# =============================================================================

def obs_to_base_features(obs: dict) -> list[float]:
    """Extract V1-style base features (27 dims) from observation dict.

    Base features capture training state without telemetry:
    - Timing: epoch, global_step (2)
    - Losses: train_loss, val_loss, loss_delta (3)
    - Accuracies: train_accuracy, val_accuracy, accuracy_delta (3)
    - Trends: plateau_epochs, best_val_accuracy, best_val_loss (3)
    - History: loss_history_5 (5), accuracy_history_5 (5)
    - Seed state: has_active_seed, seed_stage, seed_epochs_in_stage,
                  seed_alpha, seed_improvement (5)
    - Slots: available_slots (1)

    Total: 27 features

    Args:
        obs: Observation dictionary from TrainingSnapshot.to_dict()

    Returns:
        List of 27 floats
    """
    return [
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
        float(obs['has_active_seed']),
        float(obs['seed_stage']),
        float(obs['seed_epochs_in_stage']),
        obs['seed_alpha'],
        obs['seed_improvement'],
        float(obs['available_slots']),
    ]


# =============================================================================
# Action Masking
# =============================================================================

# SeedStage.PROBATIONARY = 6 (hardcoded to avoid kasmina import on hot path)
_PROBATIONARY_STAGE = 6

# Minimum seed age before CULL is allowed (structural fix for uninformed culls)
# 10 epochs gives enough signal to evaluate seed quality and ~3% survival rate
# for random exploration (0.5^5), sufficient for learning.
MIN_CULL_AGE = 10

# Epochs needed for confident seed quality assessment (for info potential PBRS)
FULL_EVALUATION_AGE = 10

# Plateau gating: prevent germination until training has plateaued
# This ensures seeds only get credit for improvements AFTER natural training gains
# have exhausted, giving cleaner reward attribution. Matches h-tamiyo behavior.
MIN_GERMINATE_EPOCH = 5        # Let host get easy wins first
MIN_PLATEAU_TO_GERMINATE = 3   # Consecutive epochs with <0.5% improvement


def compute_action_mask(
    has_active_seed: float,
    seed_stage: int,
    num_germinate_actions: int,
    seed_age_epochs: int = 0,
    epoch: int = 0,
    plateau_epochs: int = 0,
) -> list[float]:
    """Compute valid action mask based on current state.

    This enforces the Kasmina state machine rules plus plateau gating:
    - GERMINATE_*: Allowed only if no active seed AND plateau detected
    - FOSSILIZE: Allowed only if seed is in PROBATIONARY stage
    - CULL: Allowed only if active seed AND seed_age >= MIN_CULL_AGE
    - WAIT: Always allowed

    Plateau gating ensures seeds only get credit for improvements AFTER
    natural training gains have exhausted. This matches h-tamiyo behavior
    and fixes reward misattribution from early germination.

    Args:
        has_active_seed: 1.0 if seed is active, 0.0 otherwise
        seed_stage: Current seed stage (SeedStage enum value)
        num_germinate_actions: Number of germinate actions (blueprint count)
        seed_age_epochs: Total epochs since seed germination (default 0)
        epoch: Current training epoch (default 0)
        plateau_epochs: Consecutive epochs with <0.5% improvement (default 0)

    Returns:
        Binary mask list [WAIT, GERMINATE_0..N, FOSSILIZE, CULL]
        where 1.0 = valid, 0.0 = invalid

    Action layout (matches build_action_enum):
        0: WAIT
        1-N: GERMINATE_<BLUEPRINT>
        N+1: FOSSILIZE
        N+2: CULL
    """
    # Total actions: WAIT + germinate actions + FOSSILIZE + CULL
    num_actions = 1 + num_germinate_actions + 2
    mask = [0.0] * num_actions

    # WAIT (index 0): Always valid
    mask[0] = 1.0

    # GERMINATE_* (indices 1 to num_germinate_actions):
    # Only if no active seed AND plateau detected (fixes credit misattribution)
    if has_active_seed < 0.5:  # No active seed
        plateau_met = (
            epoch >= MIN_GERMINATE_EPOCH and
            plateau_epochs >= MIN_PLATEAU_TO_GERMINATE
        )
        if plateau_met:
            for i in range(1, num_germinate_actions + 1):
                mask[i] = 1.0

    # FOSSILIZE (index num_germinate_actions + 1): Only if PROBATIONARY
    if seed_stage == _PROBATIONARY_STAGE:
        mask[num_germinate_actions + 1] = 1.0

    # CULL (index num_germinate_actions + 2): Only if active seed AND old enough
    # MIN_CULL_AGE prevents "germinate-then-cull" churn exploitation
    if has_active_seed >= 0.5 and seed_age_epochs >= MIN_CULL_AGE:
        mask[num_germinate_actions + 2] = 1.0

    return mask


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
    shadowing_fraction: float = 0.1  # Fraction of max_epochs to dwell in SHADOWING (min 1 epoch)
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


def normalize_observation(obs: dict, config: TaskConfig) -> dict:
    """Normalize observations for stable PPO training."""
    achievable_range = config.achievable_range or 1.0
    return {
        "epoch": obs["epoch"] / config.max_epochs,
        "global_step": obs["global_step"] / config.max_steps,
        "train_loss": (obs["train_loss"] - config.target_loss) / achievable_range,
        "val_loss": (obs["val_loss"] - config.target_loss) / achievable_range,
        "loss_delta": obs["loss_delta"] / config.typical_loss_delta_std,
        "plateau_epochs": min(obs["plateau_epochs"] / 10.0, 1.0),
        "seed_alpha": obs["seed_alpha"],
        "has_active_seed": obs["has_active_seed"],
        "seed_stage": obs["seed_stage"] / 7.0,
    }
