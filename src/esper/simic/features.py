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

import torch

# HOT PATH: ONLY leyline imports allowed!
from esper.leyline import (
    TensorSchema,
    TENSOR_SCHEMA_SIZE,
    MIN_CULL_AGE,
    FULL_EVALUATION_AGE,
    MIN_GERMINATE_EPOCH,
    MIN_PLATEAU_TO_GERMINATE,
)

if TYPE_CHECKING:
    # Type hints only - not imported at runtime
    from typing import Any


__all__ = [
    "safe",
    "obs_to_base_features",
    "obs_to_base_features_tensor",
    "compute_action_mask",
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
# Base Features (V3 - 35 dimensions)
# =============================================================================

def obs_to_base_features(obs: dict, max_epochs: int = 200) -> list[float]:
    """Extract V3-style base features (35 dims) with pre-normalization.

    Pre-normalizes features to ~[0, 1] range for early training stability.
    This reduces the burden on RunningMeanStd during the initial warmup phase
    where statistics are poorly estimated.

    Base features capture training state without telemetry:
    - Timing: epoch, global_step (2)
    - Losses: train_loss, val_loss, loss_delta (3)
    - Accuracies: train_accuracy, val_accuracy, accuracy_delta (3)
    - Trends: plateau_epochs, best_val_accuracy, best_val_loss (3)
    - History: loss_history_5 (5), accuracy_history_5 (5)
    - Seed state: has_active_seed, seed_stage, seed_epochs_in_stage,
                  seed_alpha, seed_improvement, seed_counterfactual (6)
    - Slots: available_slots (1)
    - Host state: host_grad_norm, host_learning_phase (2)
    - Blueprint: one-hot encoding (5) [NEW - DRL Expert recommendation]

    Total: 35 features

    Args:
        obs: Observation dictionary from TrainingSnapshot.to_dict()
        max_epochs: Maximum training epochs (for normalization)

    Returns:
        List of 35 floats, pre-normalized to ~[0, 1] range
    """
    # Blueprint one-hot encoding (DRL Expert recommendation)
    # blueprint_id: 0=none, 1=first blueprint, 2=second, etc.
    # One-hot avoids imposing artificial ordinal relationships on categorical data
    blueprint_id = obs.get('seed_blueprint_id', 0)
    num_blueprints = obs.get('num_blueprints', 5)
    blueprint_one_hot = [0.0] * num_blueprints
    if blueprint_id > 0 and blueprint_id <= num_blueprints:
        blueprint_one_hot[blueprint_id - 1] = 1.0  # 1-indexed to 0-indexed

    return [
        # Timing features
        float(obs['epoch']) / max_epochs,                     # [0, 1]
        float(obs['global_step']) / (max_epochs * 100),       # ~[0, 1] assuming ~100 batches/epoch
        # Loss features (safe already clips to 10.0, divide for ~[0, 1])
        safe(obs['train_loss'], 10.0) / 10.0,                 # ~[0, 1]
        safe(obs['val_loss'], 10.0) / 10.0,                   # ~[0, 1]
        safe(obs['loss_delta'], 0.0, max_val=5.0) / 5.0,      # ~[-1, 1]
        # Accuracy features (already [0, 100] -> [0, 1])
        obs['train_accuracy'] / 100.0,                        # [0, 1]
        obs['val_accuracy'] / 100.0,                          # [0, 1]
        safe(obs['accuracy_delta'], 0.0, max_val=50.0) / 50.0,  # ~[-1, 1]
        # Trend features
        float(obs['plateau_epochs']) / 20.0,                  # ~[0, 1] typical max ~20
        obs['best_val_accuracy'] / 100.0,                     # [0, 1]
        safe(obs['best_val_loss'], 10.0) / 10.0,              # ~[0, 1]
        # History features
        *[safe(v, 10.0) / 10.0 for v in obs['loss_history_5']],       # ~[0, 1]
        *[v / 100.0 for v in obs['accuracy_history_5']],              # [0, 1]
        # Seed state features
        float(obs['has_active_seed']),                        # Already 0/1
        float(obs['seed_stage']) / 7.0,                       # Stages 0-7 -> [0, 1]
        float(obs['seed_epochs_in_stage']) / 50.0,            # ~[0, 1] typical max ~50
        obs['seed_alpha'],                                    # Already [0, 1]
        safe(obs['seed_improvement'], 0.0, max_val=10.0) / 10.0,  # [-1, 1] clamped
        float(obs['available_slots']),                        # Usually 0-2, small scale ok
        safe(obs.get('seed_counterfactual', 0.0), 0.0, max_val=10.0) / 10.0,  # [-1, 1] clamped
        # Host state features
        safe(obs.get('host_grad_norm', 0.0), 0.0, max_val=10.0) / 10.0,  # [0, 1] clamped
        obs.get('host_learning_phase', 0.0),                 # Already [0, 1]
        # Blueprint features (NEW - one-hot encoding)
        *blueprint_one_hot,                                   # 5 features, exactly one is 1.0 or all zeros
    ]


def obs_to_base_features_tensor(
    obs: dict,
    device: torch.device,
    max_epochs: int = 200,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract V3-style base features directly as tensor.

    More efficient than obs_to_base_features() + torch.tensor() for
    high-throughput training loops. Avoids Python list allocation.

    Note: This function is NOT designed for torch.compile due to
    dict access and variable-length history. Use for rollout collection
    only; the network forward pass should use the tensor directly.

    Args:
        obs: Observation dictionary
        device: Target device for tensor
        max_epochs: Maximum epochs for normalization
        out: Optional pre-allocated output tensor (35,) for zero-alloc mode

    Returns:
        Tensor of shape (35,) with base features
    """
    if out is None:
        out = torch.empty(35, dtype=torch.float32, device=device)

    # Blueprint one-hot
    blueprint_id = obs.get('seed_blueprint_id', 0)
    num_blueprints = obs.get('num_blueprints', 5)

    # Convert histories to tensors for vectorized assignment
    loss_hist_raw = torch.tensor(obs['loss_history_5'], dtype=torch.float32, device=device)
    acc_hist = torch.tensor(obs['accuracy_history_5'], dtype=torch.float32, device=device)

    # Handle NaN/Inf like safe() does - replace with default value (10.0 for loss)
    loss_hist = torch.where(torch.isfinite(loss_hist_raw), loss_hist_raw, torch.full_like(loss_hist_raw, 10.0))
    # Match safe() clipping: max_val defaults to 100.0, not 10.0
    loss_hist = torch.clamp(loss_hist, min=-100.0, max=100.0) / 10.0

    # Accuracy doesn't use safe() in list version, just divide
    acc_hist = acc_hist / 100.0

    # Fill tensor - scalar assignments
    out[0] = float(obs['epoch']) / max_epochs
    out[1] = float(obs['global_step']) / (max_epochs * 100)
    out[2] = safe(obs['train_loss'], 10.0) / 10.0
    out[3] = safe(obs['val_loss'], 10.0) / 10.0
    out[4] = safe(obs['loss_delta'], 0.0, max_val=5.0) / 5.0
    out[5] = obs['train_accuracy'] / 100.0
    out[6] = obs['val_accuracy'] / 100.0
    out[7] = safe(obs['accuracy_delta'], 0.0, max_val=50.0) / 50.0
    out[8] = float(obs['plateau_epochs']) / 20.0
    out[9] = obs['best_val_accuracy'] / 100.0
    out[10] = safe(obs['best_val_loss'], 10.0) / 10.0

    # History features - vectorized slice assignment
    out[11:16] = loss_hist
    out[16:21] = acc_hist

    # Seed state features
    out[21] = float(obs['has_active_seed'])
    out[22] = float(obs['seed_stage']) / 7.0
    out[23] = float(obs['seed_epochs_in_stage']) / 50.0
    out[24] = obs['seed_alpha']
    out[25] = safe(obs['seed_improvement'], 0.0, max_val=10.0) / 10.0
    out[26] = float(obs['available_slots'])
    out[27] = safe(obs.get('seed_counterfactual', 0.0), 0.0, max_val=10.0) / 10.0
    out[28] = safe(obs.get('host_grad_norm', 0.0), 0.0, max_val=10.0) / 10.0
    out[29] = obs.get('host_learning_phase', 0.0)

    # Blueprint one-hot (5 features)
    out[30:35] = 0.0
    if blueprint_id > 0 and blueprint_id <= num_blueprints:
        out[29 + blueprint_id] = 1.0

    return out


# =============================================================================
# Action Masking
# =============================================================================

# SeedStage.PROBATIONARY = 6 (hardcoded to avoid kasmina import on hot path)
_PROBATIONARY_STAGE = 6


def compute_action_mask(
    has_active_seed: float,
    seed_stage: int,
    num_germinate_actions: int,
    seed_age_epochs: int = 0,
    epoch: int = 0,
    plateau_epochs: int = 0,
    host_stabilized: bool = False,
) -> list[float]:
    """Compute valid action mask based on current state.

    This enforces the Kasmina state machine rules plus stabilization gating:
    - GERMINATE_*: Allowed only if no active seed AND host stabilized AND plateau detected
    - FOSSILIZE: Allowed only if seed is in PROBATIONARY stage
    - CULL: Allowed only if active seed AND seed_age >= MIN_CULL_AGE
    - WAIT: Always allowed

    Stabilization gating ensures seeds only get credit for improvements AFTER
    the explosive growth phase ends and natural training gains have exhausted.
    This matches h-tamiyo behavior and fixes reward misattribution from early
    germination.

    Args:
        has_active_seed: 1.0 if seed is active, 0.0 otherwise
        seed_stage: Current seed stage (SeedStage enum value)
        num_germinate_actions: Number of germinate actions (blueprint count)
        seed_age_epochs: Total epochs since seed germination (default 0)
        epoch: Current training epoch (default 0)
        plateau_epochs: Consecutive epochs with <0.5% improvement (default 0)
        host_stabilized: True if explosive growth phase has ended (default False)

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
    # Only if no active seed AND host stabilized AND plateau detected
    # This ensures seeds only get credit for improvements AFTER explosive growth ends
    if has_active_seed < 0.5:  # No active seed
        plateau_met = (
            epoch >= MIN_GERMINATE_EPOCH and
            plateau_epochs >= MIN_PLATEAU_TO_GERMINATE
        )
        if host_stabilized and plateau_met:
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
