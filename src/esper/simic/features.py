"""Simic Features - HOT PATH Feature Extraction

CRITICAL: This module is on the HOT PATH for vectorized training.
ONLY import from leyline. NO imports from kasmina, tamiyo, or nissa!

This module extracts features from observations for RL training.
It must be FAST and have minimal dependencies to avoid bottlenecks
in the vectorized PPO training loop.
"""

from __future__ import annotations

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
    "telemetry_to_features",
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
# Telemetry Features (V2 - 27 dimensions)
# =============================================================================

def telemetry_to_features(telem: dict) -> list[float]:
    """Extract V2 telemetry features (27 dims) from telemetry snapshot.

    Telemetry features capture deeper model health signals:
    - Gradient health: overall_norm, norm_variance, vanishing_layers,
                      exploding_layers, health_score (5)
    - Per-class accuracy: class 0-9 (10)
    - Class balance: class_variance (1)
    - Loss landscape: sharpness (1)
    - Layer gradients: gradient norms for 7 layers (7)
    - Red flags: severe_class_imbalance, sharp_minimum, gradient_issues (3)

    Total: 27 features

    Args:
        telem: Telemetry dictionary from DiagnosticTracker

    Returns:
        List of 27 floats
    """
    features = []

    # Gradient health (5 features)
    gh = telem.get('gradient_health', {})
    features.extend([
        safe(gh.get('overall_norm', 0), 0, 10),
        safe(gh.get('norm_variance', 0), 0, 10),
        float(gh.get('vanishing_layers', 0)),
        float(gh.get('exploding_layers', 0)),
        safe(gh.get('health_score', 1), 1, 1),
    ])

    # Per-class accuracy (10 features) - sorted by class name
    pca = telem.get('per_class_accuracy', {})
    for i in range(10):
        features.append(safe(pca.get(str(i), 50), 50, 100))

    # Class variance (1 feature)
    features.append(safe(telem.get('class_variance', 0), 0, 1000))

    # Sharpness (1 feature)
    features.append(safe(telem.get('sharpness', 0), 0, 100))

    # Gradient stats per layer - just norms (7 features)
    gs = telem.get('gradient_stats', [])
    layer_norms = [safe(g.get('norm', 0), 0, 10) for g in gs[:7]]
    while len(layer_norms) < 7:
        layer_norms.append(0.0)
    features.extend(layer_norms)

    # Red flags as binary (3 features)
    rf = telem.get('red_flags', [])
    features.append(1.0 if 'severe_class_imbalance' in rf else 0.0)
    features.append(1.0 if 'sharp_minimum' in rf else 0.0)
    features.append(1.0 if 'gradient_issues' in rf else 0.0)

    return features  # 27 features
