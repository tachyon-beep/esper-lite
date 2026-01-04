# EnvOverview Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/env_overview.py`
**Status:** ✅ Functionally Complete (minor dead code)

## Overview

The EnvOverview widget displays a per-environment table with real-time metrics including accuracy, reward, seed states, and reward components. Each row shows one environment's current status.

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `host_accuracy` | EPOCH_COMPLETED | `_handle_epoch_completed()` (payload.val_accuracy) | ✅ Wired |
| `host_loss` | EPOCH_COMPLETED | `_handle_epoch_completed()` (payload.val_loss) | ✅ Wired |
| `current_epoch` | EPOCH_COMPLETED | `_handle_epoch_completed()` (payload.inner_epoch) | ✅ Wired |
| `best_accuracy` | EPOCH_COMPLETED | `add_accuracy()` method | ✅ Wired |
| `epochs_since_improvement` | EPOCH_COMPLETED | `add_accuracy()` method | ✅ Wired |
| `status` | EPOCH_COMPLETED | `_update_status()` method | ✅ Wired |
| `accuracy_history` | EPOCH_COMPLETED | `add_accuracy()` (appended to deque) | ✅ Wired |
| `reward_history` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` (kind="last_action") | ✅ Wired |
| `current_reward` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `reward_components.base_acc_delta` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `reward_components.bounded_attribution` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `reward_components.seed_contribution` | ANALYTICS_SNAPSHOT | NOT SET | ⚠️ Dead Field |
| `reward_components.compute_rent` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `reward_components.stage_bonus` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `reward_components.ratio_penalty` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `reward_components.alpha_shock` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` | ✅ Wired |
| `seeds[slot_id].*` | EPOCH_COMPLETED, SEED_* events | Multiple handlers | ✅ Wired |
| `counterfactual_matrix` | COUNTERFACTUAL_MATRIX_COMPUTED | `_handle_counterfactual_matrix()` | ✅ Wired |

## Data Flow

```
Training Events (Simic/Kasmina)
    ↓
Leyline Telemetry (Event Types)
    ↓
SanctumAggregator (process_event)
    ↓
Event Handlers (per event type)
    ↓
EnvState (aggregate per-env data)
    ↓
SanctumSnapshot → EnvOverview Widget
```

## Issues Found

### 1. Dead Field: `seed_contribution`
- **Location:** `env_overview.py` line 510
- **Issue:** Widget checks `env.reward_components.seed_contribution` but field is never populated
- **Severity:** Low - fallback to `bounded_attribution` works

### 2. Incomplete Reward Components
- **Missing:** `fossilize_terminal_bonus`, `blending_warning`, `holding_warning`
- **Severity:** Medium - defined in schema but never populated

## Recommendations

1. Remove `seed_contribution` or populate it (currently uses `bounded_attribution` fallback)
2. Either populate missing reward components or remove from schema per no-legacy-code policy
