# Rich Telemetry Design for Policy Tamiyo

**Date:** 2025-11-26
**Status:** Approved
**Goal:** Capture richer training diagnostics so the policy learns *why* to intervene, not just pattern-matching on loss curves.

## Motivation

The current observation space (27 features) captures symptoms but not causes:
- Loss plateau could mean: vanishing gradients, sharp local minimum, LR too low, or underfitting
- Each cause needs a different intervention
- As we add more tools (blueprints, blending patterns), the policy needs causal understanding

## Design Decisions

1. **YAML configs** validated by Pydantic (human-readable, type-safe)
2. **Profiles** (minimal/standard/diagnostic/research) with override support
3. **Hierarchical features** - base always present, extras appended based on config
4. **Self-documenting packs** - config stored alongside data
5. **Claude-friendly annotations** - narratives, decision briefs, red flags

## TelemetryConfig Schema

```yaml
# telemetry_profiles.yaml

profiles:
  minimal:  # Fast iteration (~27 features)
    history_length: 5
    gradients:
      enabled: false
    per_class:
      enabled: false
    loss_landscape:
      enabled: false

  standard:  # Default for data generation (~45 features)
    history_length: 10
    gradients:
      enabled: true
      layers: ["conv1", "conv2", "fc"]
      track_norm: true
      track_std: true
      detect_vanishing: true
      detect_exploding: true
    per_class:
      enabled: false
    loss_landscape:
      enabled: false

  diagnostic:  # Deep analysis runs (~80 features)
    history_length: 20
    gradients:
      enabled: true
      layers: "all"
      percentiles: [1, 25, 50, 75, 99]
      detect_vanishing: true
      detect_exploding: true
      vanishing_threshold: 1e-7
      exploding_threshold: 1000
    per_class:
      enabled: true
      track_accuracy: true
      track_loss: true
    loss_landscape:
      enabled: true
      perturbation_samples: 5
      estimate_sharpness: true

  research:  # Kitchen sink (~150+ features)
    history_length: 50
    gradients:
      enabled: true
      layers: "all"
      percentiles: [1, 5, 25, 50, 75, 95, 99]
      full_histogram: true
    per_class:
      enabled: true
      track_accuracy: true
      track_loss: true
      track_confusion: true
    loss_landscape:
      enabled: true
      perturbation_samples: 10
    track_weight_norms: true
    track_activation_stats: true
```

## Pydantic Models

```python
# src/esper/telemetry_config.py

from pydantic import BaseModel, Field
from typing import Literal

class GradientConfig(BaseModel):
    enabled: bool = False
    layers: list[str] | Literal["all"] = "all"
    track_norm: bool = True
    track_std: bool = True
    percentiles: list[int] = [1, 50, 99]
    detect_vanishing: bool = True
    detect_exploding: bool = True
    vanishing_threshold: float = 1e-7
    exploding_threshold: float = 1e3

class LossLandscapeConfig(BaseModel):
    enabled: bool = False
    perturbation_samples: int = Field(default=5, ge=1, le=20)
    perturbation_scale: float = Field(default=0.01, gt=0, lt=1)
    estimate_sharpness: bool = True

class PerClassConfig(BaseModel):
    enabled: bool = False
    track_accuracy: bool = True
    track_loss: bool = False
    track_confusion: bool = False

class TelemetryConfig(BaseModel):
    profile_name: str = "standard"
    history_length: int = Field(default=10, ge=5, le=100)

    gradients: GradientConfig = Field(default_factory=GradientConfig)
    loss_landscape: LossLandscapeConfig = Field(default_factory=LossLandscapeConfig)
    per_class: PerClassConfig = Field(default_factory=PerClassConfig)

    track_weight_norms: bool = False
    track_activation_stats: bool = False
```

## Data Pack Structure

```json
{
  "pack_id": "simic_v2_2025-11-26",
  "created_at": "2025-11-26T...",

  "telemetry_config": { ... },

  "feature_schema": {
    "base": ["epoch", "train_loss", "val_loss", ...],
    "gradients": ["grad_norm_conv1", "grad_norm_fc", ...],
    "per_class": ["acc_class_0", "acc_class_1", ...],
    "_index": {
      "epoch": 0,
      "train_loss": 2,
      "grad_norm_conv1": 27
    }
  },

  "quick_stats": {
    "action_distribution": {"WAIT": 1692, "GERMINATE": 70, ...},
    "accuracy_range": [68.7, 82.5],
    "seed_success_rate": 0.39
  },

  "episodes": [{
    "episode_id": "episode_0042",

    "summary": {
      "trajectory": "plateau -> germinate -> blend -> fossilize",
      "key_moments": [
        {"epoch": 12, "event": "plateau_detected"},
        {"epoch": 15, "event": "germinate"},
        {"epoch": 23, "event": "fossilize", "improvement": "+4.2%"}
      ],
      "outcome": "successful_intervention"
    },

    "decisions": [{
      "observation": { ... },
      "action": { ... },

      "decision_context": {
        "narrative": "Loss plateaued for 4 epochs | Gradient health good",
        "red_flags": ["sustained_plateau"],
        "opportunities": ["slot_available", "healthy_gradients"],
        "why_this_action": "4-epoch plateau with healthy gradients...",
        "why_not_alternatives": { ... }
      },

      "hindsight": {
        "was_good_decision": true,
        "accuracy_5_epochs_later": 75.2
      }
    }]
  }]
}
```

## DiagnosticTracker

Key capabilities:
- Registers gradient hooks on configurable layers
- Tracks gradient norms, std, vanishing/exploding percentages
- Computes per-class accuracy when enabled
- Estimates loss landscape sharpness via perturbation
- Generates human/LLM-readable narratives
- Produces decision briefs explaining action choices

## Migration Plan

1. **Phase 1 (now):** Train baseline policy on v1 data (248 episodes, 27 features)
2. **Phase 2:** Implement TelemetryConfig + DiagnosticTracker
3. **Phase 3:** Generate v2 dataset with `diagnostic` profile
4. **Phase 4:** Train v2 policy, compare against baseline
5. **Phase 5:** Measure "did richer data help?" empirically

## Files to Create

- `src/esper/telemetry_config.py` - Pydantic models
- `src/esper/telemetry.py` - DiagnosticTracker implementation
- `src/esper/telemetry_profiles.yaml` - Profile definitions
- Update `src/esper/simic_overnight.py` - Integration

## Performance Impact

| Profile | ~Features | Run Time Impact |
|---------|-----------|-----------------|
| minimal | 27 | Baseline |
| standard | ~45 | +10% |
| diagnostic | ~80 | +30% |
| research | ~150+ | +60% |
