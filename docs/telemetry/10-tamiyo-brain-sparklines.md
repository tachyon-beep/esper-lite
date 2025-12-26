# TamiyoBrain: Sparklines & History Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
**Status:** ✅ 100% Wired

## Overview

The TamiyoBrain widget displays learning vitals with sparklines and history tracking for 8 key metrics. All history fields are `deque[float]` with `maxlen=10`, enabling compact sparkline rendering via `_render_sparkline()`.

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `policy_loss_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 631 | ✅ Wired |
| `value_loss_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 635 | ✅ Wired |
| `entropy_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 639 | ✅ Wired |
| `grad_norm_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 650 | ✅ Wired |
| `episode_return_history` | BATCH_EPOCH_COMPLETED | `_handle_batch_epoch_completed()` line 899 | ✅ Wired |
| `explained_variance_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 646 | ✅ Wired |
| `kl_divergence_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 654 | ✅ Wired |
| `clip_fraction_history` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 658 | ✅ Wired |

## Data Flow

```
PPO_UPDATE_COMPLETED event → PPOUpdatePayload
    ↓
_handle_ppo_update() extracts:
    - payload.policy_loss → append to policy_loss_history
    - payload.value_loss → append to value_loss_history
    - payload.entropy → append to entropy_history
    - payload.grad_norm → append to grad_norm_history
    - (similarly for kl, clip, explained_variance)
    ↓
get_snapshot() → SanctumSnapshot.tamiyo (with all deques)
    ↓
TamiyoBrain render methods:
    - _render_primary_metrics() for episode_return, entropy
    - _render_metrics_column() for policy_loss, value_loss, grad_norm
    - _render_sparkline() converts deque to unicode blocks (▁▂▃▄▅▆▇█)
```

## Trend Detection

Sparklines include trend indicators via `detect_trend()`:

| Metric | Direction | Good | Bad |
|--------|-----------|------|-----|
| episode_return | higher=better | ↗ | ↘ |
| entropy | stable=good | → | ↘ |
| policy_loss | lower=better | ↘ | ↗ |
| value_loss | lower=better | ↘ | ↗ |
| grad_norm | lower=better | ↘ | ↗ |

## Issues Found

**NONE.** All 8 history metrics are properly wired end-to-end:
1. PPO_UPDATE_COMPLETED events emit all metrics in PPOUpdatePayload
2. Aggregator correctly appends values to corresponding deques
3. Widget renders sparklines with proper normalization
4. History lengths appropriate (10-value windows)

## Verification Details

- All timestamps are UTC (timezone-aware)
- Thread-safe: aggregator uses `threading.Lock`
- Deques bounded at maxlen to prevent memory growth
- Empty history handled with placeholder rendering

## Recommendations

No critical fixes needed. Optional enhancements:
1. Consider maxlen=20-30 for longer trend visibility
2. Episode returns could track individual env returns alongside batch aggregates
