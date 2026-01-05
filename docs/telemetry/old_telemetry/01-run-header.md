# RunHeader Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/run_header.py`
**Status:** ✅ 95% Wired

## Overview

The RunHeader widget displays training progress and connection status in a two-row header. It shows episode/epoch/batch counters, throughput metrics, best accuracy, memory alarms, and seed stage counts.

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `current_episode` | TRAINING_STARTED, BATCH_EPOCH_COMPLETED | `_handle_training_started()`, `_handle_batch_epoch_completed()` | ✅ Wired |
| `current_epoch` | EPOCH_COMPLETED, BATCH_EPOCH_COMPLETED | `_handle_epoch_completed()`, `_handle_batch_epoch_completed()` | ✅ Wired |
| `max_epochs` | TRAINING_STARTED | `_handle_training_started()` | ✅ Wired |
| `current_batch` | BATCH_EPOCH_COMPLETED | `_handle_batch_epoch_completed()` | ✅ Wired |
| `runtime_seconds` | Computed in `_get_snapshot_unlocked()` | Set at TRAINING_STARTED, updated on snapshot | ✅ Wired |
| `vitals.epochs_per_second` | BATCH_EPOCH_COMPLETED | `_handle_batch_epoch_completed()` | ✅ Wired |
| `vitals.batches_per_hour` | BATCH_EPOCH_COMPLETED | `_handle_batch_epoch_completed()` | ✅ Wired |
| `envs[].best_accuracy` | EPOCH_COMPLETED | `_handle_epoch_completed()` via `env.add_accuracy()` | ✅ Wired |
| `connected` | TRAINING_STARTED | `_handle_training_started()` sets to True | ✅ Wired |
| `staleness_seconds` | Last event timestamp | Updated in `_get_snapshot_unlocked()` | ✅ Wired |
| `training_thread_alive` | NOT POPULATED | None | ⚠️ Unused |
| `total_events_received` | NOT POPULATED | None | ⚠️ Unused |
| `vitals.has_memory_alarm` | System resource monitoring | `_update_system_vitals()` | ✅ Wired |
| `vitals.ram_used_gb`, `vitals.ram_total_gb` | System resource monitoring | `_update_system_vitals()` | ✅ Wired |
| `vitals.gpu_stats` | System resource monitoring | `_update_system_vitals()` | ✅ Wired |
| `task_name` | TRAINING_STARTED | `_handle_training_started()` | ✅ Wired |
| `envs[].status` | EPOCH_COMPLETED | Updated in `env._update_status()` | ✅ Wired |
| `envs[].seeds[].stage` | SEED_GERMINATED, SEED_STAGE_CHANGED, SEED_FOSSILIZED, EPOCH_COMPLETED | Multiple seed event handlers | ✅ Wired |

## Data Flow

```
TRAINING_STARTED → run_id, task_name, max_epochs, connected, start_time
EPOCH_COMPLETED → env.host_accuracy, env.best_accuracy, mean_accuracy_history
BATCH_EPOCH_COMPLETED → current_episode, current_batch, vitals throughput
System monitoring → vitals.ram_*, vitals.gpu_*, vitals.has_memory_alarm
Snapshot generation → staleness_seconds, runtime_seconds
```

## Issues Found

### 1. `total_events_received` - Dead Code
- **Location:** Line 747 in schema.py, line 352 in run_header.py
- **Issue:** Field exists but never incremented by aggregator
- **Severity:** Low - harmless but confusing

### 2. `training_thread_alive` - Incomplete Implementation
- **Location:** Line 749 in schema.py, lines 334-343 in run_header.py
- **Issue:** Field defined but never populated from telemetry
- **Severity:** Low - widget shows "?" fallback, feature unimplemented

## Recommendations

1. Remove `total_events_received` - adds no value, event log shows recency
2. Implement `training_thread_alive` heartbeat or remove the feature
3. Consider caching seed stage counts in aggregator instead of recalculating per render
