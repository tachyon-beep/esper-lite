# Scoreboard Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/scoreboard.py`
**Status:** ✅ 100% Wired

## Overview

The Scoreboard widget displays the top 10 environments by peak accuracy in a leaderboard format. Each row shows rank, episode number, peak accuracy achieved, growth ratio (model size increase from seeds), and seed composition.

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `best_runs` (list) | BATCH_EPOCH_COMPLETED | `_handle_batch_epoch_completed()` | ✅ Wired |
| `cumulative_fossilized` | SEED_FOSSILIZED | `_handle_seed_event()` | ✅ Wired |
| `cumulative_pruned` | SEED_PRUNED | `_handle_seed_event()` | ✅ Wired |
| `envs[].best_accuracy` | EPOCH_COMPLETED | `_handle_epoch_completed()` | ✅ Wired |
| `record.peak_accuracy` | EnvState.best_accuracy | BestRunRecord construction | ✅ Wired |
| `record.episode` | BATCH_EPOCH_COMPLETED | BestRunRecord construction | ✅ Wired |
| `record.growth_ratio` | SEED_FOSSILIZED | BestRunRecord construction | ✅ Wired |
| `record.seeds` | EnvState.best_seeds | BestRunRecord construction | ✅ Wired |
| `record.pinned` | User interaction | `toggle_best_run_pin()` | ✅ Wired |
| `seed.blueprint_id` | SEED_GERMINATED, SEED_FOSSILIZED | `_handle_seed_event()` | ✅ Wired |
| `seed.stage` | SEED_STAGE_CHANGED, SEED_GERMINATED | `_handle_seed_event()` | ✅ Wired |

## Data Flow

```
EPOCH_COMPLETED → Updates EnvState.best_accuracy & best_seeds when new personal best
SEED_* events → Updates seed stages, blueprint IDs, cumulative counts
BATCH_EPOCH_COMPLETED → Creates BestRunRecord, sorts by peak_accuracy, keeps top 10 + pinned
get_snapshot() → Returns list(self._best_runs) to SanctumSnapshot
Scoreboard.update_snapshot() → Renders leaderboard table
```

## Issues Found

### 1. Growth Ratio Timing Assumption
- **Issue:** Calculates growth_ratio at batch end, assumes fossilization occurs before peak accuracy
- **Severity:** Low - design assumption, not a bug

### 2. Empty best_runs Fallback
- **Issue:** Falls back to current `best_accuracy` when `best_runs` empty
- **Severity:** Low - intentional but semantically different from peak tracking

## Recommendations

1. Document growth_ratio timing assumption in code comments
2. Consider making fallback behavior explicit in documentation
3. Blueprint graveyard stats already captured but not displayed - future widget opportunity
