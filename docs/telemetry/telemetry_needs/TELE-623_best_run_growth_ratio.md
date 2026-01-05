# Telemetry Record: [TELE-623] Best Run Growth Ratio

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-623` |
| **Name** | Best Run Growth Ratio |
| **Category** | `scoreboard` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How much larger is the mutated model compared to the original host due to fossilized seeds?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [x] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | ratio (multiplier) |
| **Range** | `[1.0, ~2.0+]` — 1.0 means no growth, 1.2 means 20% larger |
| **Precision** | 2 decimal places for display |
| **Default** | `1.0` (no fossilized seeds) |

### Semantic Meaning

> Growth ratio measures the relative size increase of the model due to fossilized seeds:
>
> **growth_ratio = (host_params + fossilized_params) / host_params**
>
> - **1.0:** No fossilized seeds, model is original size
> - **1.05:** 5% parameter increase from fossilized seeds
> - **1.1+:** Significant growth, seeds adding notable capacity
> - **1.5+:** Major expansion, may indicate over-mutation
>
> This metric helps evaluate the efficiency of fossilized seeds. High accuracy with low growth ratio indicates efficient mutations; high growth ratio with low accuracy indicates wasted capacity.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **No Growth** | `<= 1.0` | No fossilized seeds (displayed dim) |
| **Efficient** | `1.0 < ratio < 1.1` | Modest growth (displayed cyan) |
| **Significant** | `>= 1.1` | Notable growth (displayed bold cyan) |

**Threshold Source:** `src/esper/karn/sanctum/widgets/scoreboard.py` — `_format_growth_ratio()` lines 401-408

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed in aggregator from host_params and seed_params at BestRunRecord creation |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_episode_ended()` |
| **Line(s)** | 1230-1237 |

```python
# Calculate growth ratio
base_params = env.host_params
seed_params_total = sum(
    int(seed.seed_params or 0) for seed in env.best_seeds.values()
)
growth_ratio = (
    (base_params + seed_params_total) / base_params
    if base_params > 0
    else 1.0
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Seed parameters tracked in SeedState.seed_params | `karn/sanctum/schema.py` |
| **2. Collection** | EnvState.best_seeds captures seeds at peak accuracy | `karn/sanctum/schema.py` (line 603-618) |
| **3. Aggregation** | EPISODE_ENDED computes ratio from host_params + seed_params | `karn/sanctum/aggregator.py` (lines 1230-1237) |
| **4. Delivery** | Written to `snapshot.best_runs[i].growth_ratio` | `karn/sanctum/schema.py` |

```
[EnvState.host_params, EnvState.best_seeds[].seed_params]
  --EPISODE_ENDED-->
  [growth_ratio = (host + sum(seed_params)) / host]
  --BestRunRecord-->
  [snapshot.best_runs[i].growth_ratio]
  --Scoreboard-->
  ["1.05x" display]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `growth_ratio` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].growth_ratio` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1245 |
| **Default Value** | `1.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard | `widgets/scoreboard.py` (lines 401-408) | Displayed in "Grw" column as formatted ratio with color coding |
| EnvState (live) | `schema.py` (lines 570-578) | Also available as `env.growth_ratio` property for live view |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Aggregator computes growth_ratio at lines 1230-1237
- [x] **Transport works** — Value computed from EnvState fields at EPISODE_ENDED
- [x] **Schema field exists** — `BestRunRecord.growth_ratio: float = 1.0` at line 1245
- [x] **Default is correct** — `1.0` represents no growth (original host size)
- [x] **Consumer reads it** — Scoreboard displays via `_format_growth_ratio()`
- [x] **Display is correct** — Rendered as ratio with "x" suffix (e.g., "1.05x")
- [x] **Thresholds applied** — Three-tier color: dim/cyan/bold cyan

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_growth_ratio_calculation` | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_format_growth_ratio` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Scoreboard "Grw" column showing growth ratios
4. Verify runs with no fossilized seeds show "1.00x" in dim
5. Verify runs with fossilized seeds show colored ratios based on growth amount

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EnvState.host_params | field | Base model parameter count |
| SeedState.seed_params | field | Per-seed parameter count |
| EnvState.best_seeds | snapshot | Seeds at time of peak accuracy |
| EPISODE_ENDED event | event | Triggers BestRunRecord creation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Efficiency analysis | research | Accuracy per parameter added |
| Scoreboard display | display | Growth column visualization |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Growth ratio is computed from `best_seeds` (the seeds at peak accuracy time), not current seeds. This ensures the ratio reflects the model state when peak performance was achieved.
>
> **Parameter Counting:** The ratio includes all seeds in `best_seeds` regardless of stage. This means BLENDING seeds (partially integrated) are counted at their full seed_params, which slightly overstates growth during blending.
>
> **Live vs Historical:** The EnvState class also has a `growth_ratio` property (line 570-578) that computes live growth from `host_params + fossilized_params`. The BestRunRecord.growth_ratio is a snapshot at peak accuracy time.
>
> **Efficiency Metric:** Dividing peak_accuracy by growth_ratio gives an "accuracy per size" efficiency metric. High accuracy with low growth indicates efficient mutations.
>
> **Wiring Status:** Fully wired and operational. Growth ratio is computed at EPISODE_ENDED from best_seeds snapshot.
