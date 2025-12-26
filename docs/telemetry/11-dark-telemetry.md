# Dark Telemetry Catalog

**Audit Date:** 2025-12-26
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
| `head_slot_grad_norm` | float | `_handle_ppo_update()` L696 | Per-head gradient health |
| `head_blueprint_grad_norm` | float | `_handle_ppo_update()` L700 | Per-head gradient health |
| `head_style_grad_norm` | float | schema only | Per-head gradient health (not captured) |
| `head_tempo_grad_norm` | float | schema only | Per-head gradient health (not captured) |
| `head_alpha_target_grad_norm` | float | schema only | Per-head gradient health (not captured) |
| `head_alpha_speed_grad_norm` | float | schema only | Per-head gradient health (not captured) |
| `head_alpha_curve_grad_norm` | float | schema only | Per-head gradient health (not captured) |
| `head_op_grad_norm` | float | schema only | Per-head gradient health (not captured) |
| `inner_epoch` | int | `_handle_ppo_update()` L717 | PPO inner loop progress |
| `ppo_batch` | int | `_handle_ppo_update()` L718 | Current batch within PPO update |

### Notes on TamiyoState Dark Telemetry

1. **Per-head grad norms**: Only `head_slot_grad_norm` and `head_blueprint_grad_norm` are captured from payload. The other 6 head grad norms exist in schema but are never populated by the aggregator (dead fields - see P2 issues in 00-index.md).

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
| `steps_per_second` | float | Not populated | Training step throughput |
| `gpu_utilization` | float | `_update_system_vitals()` L1376 | GPU compute utilization (always 0.0 currently) |

### Notes on SystemVitals Dark Telemetry

1. **CPU percent**: Collected but never displayed. Could be added to RunHeader or a system panel.

2. **GPU utilization**: Currently always 0.0 because NVML integration is not implemented. The field exists for future use.

3. **Steps per second**: Field exists but is never populated by the aggregator.

---

## Snapshot-Level Dark Telemetry

| Field | Type | Source | Potential Use |
|-------|------|--------|---------------|
| `total_events_received` | int | Not populated | Debug: total event count |
| `training_thread_alive` | bool\|None | Not populated | Thread health indicator |
| `poll_count` | int | Not populated | Debug: UI poll cycles |
| `max_batches` | int | Static (100) | Batch progress bar |

### Notes on Snapshot-Level Dark Telemetry

1. **Dead fields**: `total_events_received` and `training_thread_alive` are in schema but never populated. These are marked as P2 cleanup items in 00-index.md.

2. **Poll count**: Debug field, never populated.

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
| SystemVitals | ~12 | ~8 | ~4 |
| BestRunRecord | ~20 | ~15 | ~5 |
| Snapshot-Level | ~30 | ~27 | ~3 |

**Approximate totals:** ~45 dark telemetry fields captured but not displayed.

---

## Recommendations for Future Use

### High Value (Low Effort to Display)

1. **`learning_rate` + `entropy_coef`**: Add to PPO Vitals or a dedicated "Schedule" sparkline
2. **`host_loss`**: Add to EnvOverview as optional column
3. **`cpu_percent`**: Add to RunHeader system alarm section
4. **`improvement`**: Show in Scoreboard or seed table tooltips

### Medium Value (Detail Panel Candidates)

1. **Advantage distribution** (min/max/raw): PPO diagnostics panel
2. **Gradient health** (dead_layers, exploding_layers, nan_grad_count): Add to AnomalyStrip
3. **Blueprint graveyard stats**: Training summary modal
4. **`blend_tempo_epochs`**: Seed detail tooltip

### Low Priority (Schema Cleanup Candidates)

1. **Per-head grad norms** (6 unpopulated fields): Either populate from PPO or remove
2. **`steps_per_second`**: Either populate or remove
3. **`training_thread_alive`**, `total_events_received`: Either wire up or remove

---

## Schema Fields Never Populated

These fields exist in schema.py but are never populated by the aggregator:

| Field | Location | Status |
|-------|----------|--------|
| `head_style_grad_norm` | TamiyoState | Dead (payload doesn't emit) |
| `head_tempo_grad_norm` | TamiyoState | Dead (payload doesn't emit) |
| `head_alpha_target_grad_norm` | TamiyoState | Dead (payload doesn't emit) |
| `head_alpha_speed_grad_norm` | TamiyoState | Dead (payload doesn't emit) |
| `head_alpha_curve_grad_norm` | TamiyoState | Dead (payload doesn't emit) |
| `head_op_grad_norm` | TamiyoState | Dead (payload doesn't emit) |
| `steps_per_second` | SystemVitals | Dead (never computed) |
| `total_events_received` | SanctumSnapshot | Dead (never incremented) |
| `training_thread_alive` | SanctumSnapshot | Dead (never set) |
| `poll_count` | SanctumSnapshot | Dead (never incremented) |

**Recommendation:** Per no-legacy-code policy, remove dead fields during next schema cleanup.
