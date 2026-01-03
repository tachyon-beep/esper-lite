# Telemetry Audit Records

This folder contains **89 telemetry records** documenting the complete data flow for every metric consumed by the Sanctum TUI system.

## Audit Scope

Each record traces a metric's journey:
1. **Emitter** — Where the value is computed (usually `simic/`)
2. **Transport** — How it reaches the aggregator (`karn/sanctum/aggregator.py`)
3. **Schema** — Where it's stored (`karn/sanctum/schema.py`)
4. **Consumer** — Which widgets display it (`karn/sanctum/widgets/`)

## Record Count by Category

| Category | ID Range | Count | Description |
|----------|----------|-------|-------------|
| Training | 001-099 | 7 | Core training loop metrics (episode, epoch, batch, runtime) |
| Policy | 100-199 | 20 | Policy network metrics (entropy, KL, clip fraction, advantages) |
| Value | 200-299 | 6 | Value function metrics (explained variance, value stats) |
| Gradient | 300-399 | 10 | Gradient health (norms, NaN/Inf counts, per-head tracking) |
| Reward | 400-499 | 3 | Reward signal health (PBRS, anti-gaming, hypervolume) |
| Seed | 500-599 | 16 | Seed lifecycle (counts, rates, trends, gradient flags) |
| Environment | 600-699 | 6 | Per-environment state (observation stats, env status) |
| Infrastructure | 700-799 | 20 | System resources (GPU, CPU, memory, connection status) |
| Decision | 800-899 | 1 | Tamiyo decision snapshots |
| **Total** | | **89** | |

## Wiring Status Summary

### Fully Wired (83 metrics)

The vast majority of metrics have complete data pipelines from emitter through aggregator to widget display. These are marked with all checkboxes checked in their Wiring Verification section.

### Wiring Gaps (4 metrics)

These metrics have schema definitions and widget consumers, but the emitter code is stubbed or missing:

| ID | Metric | Schema Location | Issue |
|----|--------|-----------------|-------|
| TELE-600 | `obs_nan_count` | `EnvState.obs_nan_count` | Emitter not implemented |
| TELE-601 | `obs_inf_count` | `EnvState.obs_inf_count` | Emitter not implemented |
| TELE-602 | `outlier_pct` | `EnvState.outlier_pct` | Emitter not implemented |
| TELE-603 | `normalization_drift` | `EnvState.normalization_drift` | Emitter not implemented |

**Root Cause:** The observation statistics were planned but never wired from the training loop. The schema and widget code exist, but no code in `simic/` emits these values.

### Partially Wired (1 metric)

| ID | Metric | Issue |
|----|--------|-------|
| TELE-610 | `episode_stats` | Only `total_episodes` is populated; `avg_return`, `best_return`, `worst_return` remain at defaults |

### Broken Wiring (1 metric)

| ID | Metric | Issue |
|----|--------|-------|
| TELE-301 | `inf_grad_count` | Field exists but contains bugs preventing correct increment |

**Bugs identified:**
1. Counter logic doesn't properly detect infinite gradients
2. Reset timing causes missed counts

## Key Patterns Discovered

### Well-Designed Pipelines

- **PPO metrics** (entropy, KL, clip fraction, losses) — Clean emission from `PPOAgent`, proper aggregation, multiple widget consumers
- **Infrastructure metrics** (GPU, CPU, memory) — Direct PyTorch/psutil queries in aggregator, no emission needed
- **Seed lifecycle** — `SeedLifecycleStats` dataclass aggregates all seed state changes cleanly

### Common Wiring Pattern

```
[simic/agent/ppo.py]
  → TelemetryEmitter.emit_ppo_update(metrics)
  → [leyline/telemetry.py] Event payload
  → [karn/sanctum/aggregator.py] handle_ppo_update()
  → [karn/sanctum/schema.py] TamiyoState fields
  → [karn/sanctum/widgets/*] Display
```

### Infrastructure Pattern (No Emission)

```
[torch.cuda / psutil]
  → [karn/sanctum/aggregator.py] _update_system_stats()
  → [karn/sanctum/schema.py] SystemVitals fields
  → [karn/sanctum/widgets/*] Display
```

## Cross-References

- **Template:** `../templates/TELEMETRY_RECORD_TEMPLATE.md`
- **Example:** `../templates/EXAMPLE_TELE-001_entropy.md`
- **Schema source:** `src/esper/karn/sanctum/schema.py`
- **Aggregator source:** `src/esper/karn/sanctum/aggregator.py`
- **Widget consumers:** `src/esper/karn/sanctum/widgets/`

## Audit Methodology

Records were created by:
1. Reading each widget to identify consumed fields
2. Tracing each field back through schema → aggregator → emitter
3. Documenting the complete pipeline with file paths and line numbers
4. Verifying wiring status (emitter exists, transport works, consumer reads)
5. Marking gaps honestly when wiring was incomplete

## Next Steps

To fix the identified gaps:

1. **TELE-600 through TELE-603:** Implement observation stats emission in `simic/training/vectorized.py` during rollout collection
2. **TELE-610:** Wire remaining `EpisodeStats` fields from episode completion events
3. **TELE-301:** Fix inf_grad_count increment logic in gradient collection

---

*Audit completed: 2026-01-03*
