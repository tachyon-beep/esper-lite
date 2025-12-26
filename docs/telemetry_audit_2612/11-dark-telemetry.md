# Dark Telemetry Catalog

**Audit Date:** 2025-12-26
**Updated:** 2025-12-26 (wired 6 per-head grad norms, removed dead `steps_per_second`)
**Purpose:** Catalog all telemetry captured by SanctumAggregator but NOT displayed in Sanctum UI widgets
**Rationale:** Preserve record of available data for future features; avoid re-aggregating already-captured metrics

## Overview

The Sanctum TUI displays approximately 60-70% of the telemetry data captured by the aggregator. The remaining "dark telemetry" is valuable for:
- Future debugging features
- Advanced diagnostics panels
- Export/logging capabilities
- Power-user detail views

**This document catalogs what IS captured but NOT displayed.**

---

## TamiyoState Dark Telemetry

| Field | Type | Source Handler | Potential Use |
|-------|------|----------------|---------------|
| `entropy_loss` | float | `_handle_ppo_update()` L661 | PPO diagnostics - separate from entropy coefficient |
| `learning_rate` | float\|None | `_handle_ppo_update()` L674 | LR schedule visualization, decay monitoring |
| `entropy_coef` | float | `_handle_ppo_update()` L678 | Entropy annealing visualization |
| `ratio_std` | float | `_handle_ppo_update()` L671 | Policy stability - high std = erratic updates |
| `advantage_min` | float | `_handle_ppo_update()` L664 | Advantage distribution health |
| `advantage_max` | float | `_handle_ppo_update()` L664 | Advantage distribution health |
| `advantage_raw_mean` | float | `_handle_ppo_update()` L664 | Pre-normalization signal magnitude |
| `advantage_raw_std` | float | `_handle_ppo_update()` L664 | Pre-normalization signal magnitude |
| `dead_layers` | int | `_handle_ppo_update()` L681 | Gradient health - vanishing gradients |
| `exploding_layers` | int | `_handle_ppo_update()` L682 | Gradient health - exploding gradients |
| `nan_grad_count` | int | `_handle_ppo_update()` L683 | NaN detection for training stability |
| `layer_gradient_health` | dict\|None | `_handle_ppo_update()` L684 | Per-layer gradient norms for debugging |
| `entropy_collapsed` | bool | `_handle_ppo_update()` L686 | Policy collapse detection flag |
| `update_time_ms` | float | `_handle_ppo_update()` L689 | PPO update duration for profiling |
| `early_stop_epoch` | int\|None | `_handle_ppo_update()` L690 | KL early stopping triggered epoch |
| `head_style_grad_norm` | float | `_handle_ppo_update()` L701-702 | Per-head gradient health |
| `head_tempo_grad_norm` | float | `_handle_ppo_update()` L703-704 | Per-head gradient health |
| `head_alpha_target_grad_norm` | float | `_handle_ppo_update()` L705-706 | Per-head gradient health |
| `head_alpha_speed_grad_norm` | float | `_handle_ppo_update()` L707-708 | Per-head gradient health |
| `head_alpha_curve_grad_norm` | float | `_handle_ppo_update()` L709-710 | Per-head gradient health |
| `head_op_grad_norm` | float | `_handle_ppo_update()` L711-712 | Per-head gradient health |
| `inner_epoch` | int | `_handle_ppo_update()` L717 | PPO inner loop progress |
| `ppo_batch` | int | `_handle_ppo_update()` L718 | Current batch within PPO update |

### Notes on TamiyoState Dark Telemetry

1. **Per-head grad norms**: All 8 heads now have gradient norms wired through the pipeline (wired 2025-12-26). These are captured but not yet displayed - could power a gradient health heatmap or anomaly detection.

2. **Advantage stats**: `advantage_mean` and `advantage_std` ARE displayed in TamiyoBrain. The raw/min/max variants are dark.

3. **Ratio stats**: Only `ratio_mean` is displayed (in PPO Vitals). The min/max/std are dark.

---

## EnvState Dark Telemetry

| Field | Type | Source Handler | Potential Use |
|-------|------|----------------|---------------|
| `host_loss` | float | `_handle_epoch_completed()` L560 | Training loss curve (only accuracy shown) |
| `best_reward` | float | `add_reward()` method | Per-env best reward tracking |
| `best_reward_epoch` | int | `add_reward()` method | When best reward was achieved |
| `stall_counter` | int | `_update_status()` method | Hysteresis counter (status shown, not raw) |
| `degraded_counter` | int | `_update_status()` method | Hysteresis counter (status shown, not raw) |
| `epochs_since_improvement` | int | `add_accuracy()` method | Stagnation depth |

### Notes on EnvState Dark Telemetry

1. **Loss vs Accuracy**: The TUI focuses on validation accuracy. Host loss is captured but could be valuable for identifying overfitting (loss decreasing while accuracy stagnates).

2. **Hysteresis internals**: The status hysteresis counters are implementation details but could inform debugging ("why isn't this env marked stalled yet?").

---

## SeedState Dark Telemetry

| Field | Type | Source Handler | Potential Use |
|-------|------|----------------|---------------|
| `improvement` | float | `_handle_seed_event()` L809,847 | Accuracy delta at fossilize/prune |
| `prune_reason` | str | `_handle_seed_event()` L846 | Why seed was pruned (shown in event log only) |
| `auto_pruned` | bool | `_handle_seed_event()` L848 | System vs policy prune decision |
| `epochs_total` | int | `_handle_seed_event()` L811,849 | Total seed lifespan |
| `counterfactual` | float | `_handle_seed_event()` L812,850 | Causal attribution score |
| `seed_params` | int | `_handle_seed_event()` L744 | Parameter count per seed |
| `blend_tempo_epochs` | int | `_handle_seed_event()` L749 | Tamiyo's chosen integration speed |

### Notes on SeedState Dark Telemetry

1. **Improvement**: This is the accuracy delta when a seed is fossilized or pruned. Currently only shown in event log metadata, not in the main seed table or scoreboard.

2. **Counterfactual**: Individual seed counterfactual scores are captured but only the aggregate synergy indicator is shown. Individual contributions could be displayed in a detail panel.

3. **Blend tempo**: Tamiyo's chosen integration speed (3=FAST, 5=STANDARD, 8=SLOW) is captured but not displayed anywhere.

---

## RewardComponents Dark Telemetry

| Field | Type | Source Handler | Potential Use |
|-------|------|----------------|---------------|
| `fossilize_terminal_bonus` | float | Not captured from payload | Terminal reward at fossilization |
| `blending_warning` | float | Not captured from payload | Warning signal during blending |
| `holding_warning` | float | Not captured from payload | Warning signal during holding |
| `seed_contribution` | float | Not captured from payload | Legacy attribution (replaced by bounded_attribution) |

### Notes on RewardComponents Dark Telemetry

1. **Not all components are captured**: The aggregator only captures `base_acc_delta`, `bounded_attribution`, `compute_rent`, `stage_bonus`, `ratio_penalty`, and `alpha_shock` from the ANALYTICS_SNAPSHOT payload. The terminal bonuses and warnings are in the schema but not extracted.

2. **RewardHealthPanel**: Uses a subset (stage_bonus, ratio_penalty, alpha_shock) for the health display. The full breakdown is available in the focused env detail panel.

---

## SystemVitals Dark Telemetry

| Field | Type | Source Handler | Potential Use |
|-------|------|----------------|---------------|
| `cpu_percent` | float\|None | `_update_system_vitals()` L1344 | CPU utilization |
| `gpu_temperature` | float | `_update_system_vitals()` L1388 | Thermal throttling detection |
| `gpu_utilization` | float | `_update_system_vitals()` L1376 | GPU compute utilization (always 0.0 currently) |

### Notes on SystemVitals Dark Telemetry

1. **CPU percent**: Collected but never displayed. Could be added to RunHeader or a system panel.

2. **GPU utilization**: Currently always 0.0 because NVML integration is not implemented. The field exists for future use.

3. **`steps_per_second`**: REMOVED from schema (2025-12-26) - was never populated and is redundant with epochs_per_second/batches_per_hour.

---

## Snapshot-Level Dark Telemetry

| Field | Type | Source | Status |
|-------|------|--------|--------|
| `poll_count` | int | `app.py:317` | Debug field, used in logging only |
| `max_batches` | int | Static (100) | Batch progress bar |

### Notes on Snapshot-Level Dark Telemetry

1. **`total_events_received`**: ✅ WIRED - populated by `backend.py:89`, displayed in `run_header.py:346-352`

2. **`training_thread_alive`**: ✅ WIRED - populated by `app.py:321`, displayed in `run_header.py:334-342`

3. **`poll_count`**: Debug field, intentionally not displayed in widgets.

---

## BestRunRecord Dark Telemetry

The BestRunRecord captures full env snapshots at peak accuracy. Most fields ARE used in the detail modal, but some are captured without display:

| Field | Type | Captured | Displayed |
|-------|------|----------|-----------|
| `host_loss` | float | Yes | No |
| `host_params` | int | Yes | No |
| `fossilized_count` | int | Yes | No (in detail modal) |
| `pruned_count` | int | Yes | No (in detail modal) |
| `blueprint_spawns` | dict | Yes | No |
| `blueprint_fossilized` | dict | Yes | No |
| `blueprint_prunes` | dict | Yes | No |

### Notes on BestRunRecord Dark Telemetry

1. **Blueprint graveyard**: Per-blueprint lifecycle stats are captured but not displayed in the scoreboard or detail modal. Could power a "blueprint success rate" analysis.

---

## Cumulative Counters (Aggregator State)

These are aggregator-internal counters that appear in the snapshot but may not be fully displayed:

| Field | Location | Displayed |
|-------|----------|-----------|
| `cumulative_fossilized` | SanctumSnapshot | Yes (RunHeader seed counts) |
| `cumulative_pruned` | SanctumSnapshot | Yes (RunHeader seed counts) |
| `cumulative_blueprint_spawns` | SanctumSnapshot | No |
| `cumulative_blueprint_fossilized` | SanctumSnapshot | No |
| `cumulative_blueprint_prunes` | SanctumSnapshot | No |

### Notes on Cumulative Counters

1. **Blueprint lifecycle cumulative stats**: Available for a "training run summary" panel showing which blueprints are most successful.

---

## Summary: Dark Telemetry by Category

| Category | Total Fields | Displayed | Dark |
|----------|--------------|-----------|------|
| TamiyoState (PPO/Policy) | ~45 | ~28 | ~17 |
| EnvState (Per-Env) | ~35 | ~30 | ~5 |
| SeedState (Per-Seed) | ~15 | ~10 | ~5 |
| RewardComponents | ~12 | ~6 | ~6 |
| SystemVitals | ~11 | ~8 | ~3 |
| BestRunRecord | ~20 | ~15 | ~5 |
| Snapshot-Level | ~30 | ~28 | ~2 |

**Approximate totals:** ~43 dark telemetry fields captured but not displayed.

---

## Recommendations for Future Use

### High Value (Low Effort to Display)

1. **`learning_rate` + `entropy_coef`**: Add to PPO Vitals or a dedicated "Schedule" sparkline
2. **`host_loss`**: Add to EnvOverview as optional column
3. **`cpu_percent`**: Add to RunHeader system alarm section
4. **`improvement`**: Show in Scoreboard or seed table tooltips
5. **Per-head grad norms (8 total)**: Now fully wired - add gradient heatmap to TamiyoBrain

### Medium Value (Detail Panel Candidates)

1. **Advantage distribution** (min/max/raw): PPO diagnostics panel
2. **Gradient health** (dead_layers, exploding_layers, nan_grad_count): Add to AnomalyStrip
3. **Blueprint graveyard stats**: Training summary modal
4. **`blend_tempo_epochs`**: Seed detail tooltip

### Low Priority (Schema Cleanup Candidates)

None remaining after 2025-12-26 cleanup.

---

## Changelog

### 2025-12-26: Wiring and Cleanup

**Wired (6 fields):**
- `head_style_grad_norm`
- `head_tempo_grad_norm`
- `head_alpha_target_grad_norm`
- `head_alpha_speed_grad_norm`
- `head_alpha_curve_grad_norm`
- `head_op_grad_norm`

These were computed by PPO but not emitted. Now fully wired through:
- `emitters.py` → `PPOUpdatePayload` → `aggregator.py` → `schema.py`

**Removed (1 field):**
- `steps_per_second` from `SystemVitals` - never populated, redundant with existing throughput metrics

**Corrected (2 fields):**
- `total_events_received` - was incorrectly marked as dead; actually wired via `backend.py` → `run_header.py`
- `training_thread_alive` - was incorrectly marked as dead; actually wired via `app.py` → `run_header.py`
