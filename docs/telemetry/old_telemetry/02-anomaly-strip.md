# AnomalyStrip Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/anomaly_strip.py`
**Status:** ✅ 100% Wired

## Overview

The AnomalyStrip widget displays a single-line summary of all anomalies detected across the system. It aggregates 5 key anomaly dimensions:
1. **Stalled environments** (yellow warning)
2. **Degraded environments** (red critical)
3. **Gradient issues** per seed (red critical)
4. **PPO health problems** (yellow warning)
5. **Memory pressure** (red critical)

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `env.status` | EPOCH_COMPLETED → `_update_status()` | `_handle_epoch_completed()` | ✅ Wired |
| `seed.has_exploding` | SEED_GERMINATED, SEED_STAGE_CHANGED, EPOCH_COMPLETED | `_handle_seed_event()`, `_handle_epoch_completed()` | ✅ Wired |
| `seed.has_vanishing` | SEED_GERMINATED, SEED_STAGE_CHANGED, EPOCH_COMPLETED | `_handle_seed_event()`, `_handle_epoch_completed()` | ✅ Wired |
| `tamiyo.entropy_collapsed` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 686 | ✅ Wired |
| `tamiyo.kl_divergence` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` line 652 | ✅ Wired |
| `vitals.has_memory_alarm` | System polling in `_update_system_vitals()` | Computed from GPU/RAM stats | ✅ Wired |

## Data Flow

1. **Environment Status** (Stalled/Degraded):
   - EPOCH_COMPLETED → `_handle_epoch_completed()` → `env._update_status()`
   - Uses hysteresis (3-epoch threshold) to prevent flicker

2. **Gradient Health**:
   - SEED_GERMINATED/SEED_STAGE_CHANGED/EPOCH_COMPLETED → `env.seeds[slot_id].{has_exploding, has_vanishing}`

3. **PPO Health**:
   - PPO_UPDATE_COMPLETED → `tamiyo.entropy_collapsed`, `tamiyo.kl_divergence`
   - Widget threshold: `kl_divergence > 0.05` triggers warning

4. **Memory Pressure**:
   - System vitals polling → `_update_system_vitals()` → `vitals.has_memory_alarm` (>90% threshold)

## Issues Found

**NONE DETECTED.** The AnomalyStrip telemetry wiring is complete and correct.

## Recommendations

1. Consider adding separate counts for explosive vs vanishing gradient issues
2. The kl_divergence > 0.05 threshold should be extracted to a constant
3. Consider tracking which device triggered memory alarm for more precise diagnostics
