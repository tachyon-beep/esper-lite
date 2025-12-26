# TamiyoBrain: PPO Health Gauges Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
**Status:** ✅ 95% Wired (per-head grad norms missing)

## Overview

The TamiyoBrain widget displays PPO training health through multiple metrics:
- Status Banner (entropy, explained variance, KL, clip fraction, gradient health)
- Gauge Grid (4 gauges with trend indicators)
- Metrics Column (sparklines + values)
- Head Heatmap (per-action-head entropy)

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.637-639 | ✅ Wired |
| `explained_variance` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.642-646 | ✅ Wired |
| `kl_divergence` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.652-654 | ✅ Wired |
| `clip_fraction` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.656-658 | ✅ Wired |
| `policy_loss` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.629-631 | ✅ Wired |
| `value_loss` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.633-635 | ✅ Wired |
| `grad_norm` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.648-650 | ✅ Wired |
| `advantage_mean` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.664 | ✅ Wired |
| `advantage_std` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.665 | ✅ Wired |
| `dead_layers` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.681 | ✅ Wired |
| `exploding_layers` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.682 | ✅ Wired |
| `ratio_min`, `ratio_max` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.669-670 | ✅ Wired |
| `head_slot_entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.693-694 | ✅ Wired |
| `head_blueprint_entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.697-698 | ✅ Wired |
| `head_style_entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.703-704 | ✅ Wired |
| `head_tempo_entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.705-706 | ✅ Wired |
| `head_alpha_*_entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.707-712 | ✅ Wired |
| `head_op_entropy` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.713-714 | ✅ Wired |
| `entropy_coef` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.677-678 | ✅ Wired |
| `layer_gradient_health` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.684-685 | ✅ Wired |
| `entropy_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.639 | ✅ Wired |
| `policy_loss_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.631 | ✅ Wired |
| `value_loss_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.635 | ✅ Wired |
| `grad_norm_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` l.650 | ✅ Wired |
| `head_*_grad_norm` | PPO_UPDATE_COMPLETED | NOT EXTRACTED | ⚠️ Dead |

## Data Flow

```
PPO Training Loop → emit_ppo_update_event()
    ↓
PPOUpdatePayload with all metrics
    ↓
Nissa Event Hub → SanctumAggregator.process_event()
    ↓
_handle_ppo_update() extracts fields → TamiyoState
    ↓
get_snapshot() → SanctumSnapshot.tamiyo
    ↓
TamiyoBrain render methods
```

## Issues Found

### 1. Per-Head Gradient Norms Defined But Not Displayed
- **Fields:** `head_slot_grad_norm` through `head_op_grad_norm` in schema
- **Issue:** Emitted in pipeline but aggregator never extracts them
- **Severity:** Medium - dead code
- **Recommendation:** Either populate and display, or remove from schema

## Recommendations

1. Decide on per-head gradient norms: display alongside entropy bars or remove
2. Add integration test for PPO→Aggregator→Widget pipeline
3. Document field-to-widget correlation in handler docstring
