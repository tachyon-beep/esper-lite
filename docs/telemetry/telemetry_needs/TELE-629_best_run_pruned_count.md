# Telemetry Record: [TELE-629] Best Run Pruned Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-629` |
| **Name** | Best Run Pruned Count |
| **Category** | `episode` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds had been pruned by the time this environment reached its peak accuracy?"

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
| **Type** | `int` |
| **Units** | count (seeds) |
| **Range** | `[0, total_slots]` |
| **Precision** | Integer |
| **Default** | `0` |

### Semantic Meaning

> Pruned count in a BestRunRecord represents the **point-in-time snapshot** of how many seeds had been pruned (removed due to poor performance) by the time the environment achieved its peak accuracy during that episode.
>
> This differs fundamentally from TELE-504 (`SeedLifecycleStats.prune_count`):
> - **TELE-504:** Cumulative count for the entire training run, continuously incrementing
> - **TELE-629:** Snapshot count at peak accuracy for a specific environment/episode
>
> Use cases:
> - **Efficiency analysis:** Did best runs experience more or fewer pruned experiments?
> - **Policy health:** High prune counts relative to fossilized may indicate struggling policy
> - **Historical comparison:** Compare failure rates across top-performing runs

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Context-dependent** | N/A | Value interpreted relative to fossilized_count and total slots |

**Threshold Source:** Display always uses red styling (pruning represents failed experiments)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EnvState.pruned_count accumulated during episode |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_seed_pruned()` |
| **Line(s)** | 1144 |

```python
# In _handle_seed_pruned():
env.pruned_count += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `SEED_PRUNED` event increments `EnvState.pruned_count` | `karn/sanctum/aggregator.py` (line 1144) |
| **2. Accumulation** | Count accumulates in EnvState throughout episode | `karn/sanctum/schema.py` (line 460) |
| **3. Aggregation** | EPISODE_ENDED handler copies to BestRunRecord | `karn/sanctum/aggregator.py` (line 1261) |
| **4. Delivery** | Written to `snapshot.best_runs[i].pruned_count` | `karn/sanctum/schema.py` (line 1268) |
| **5. Reset** | EnvState.pruned_count reset to 0 for next episode | `karn/sanctum/aggregator.py` (line 1290) |

```
[SEED_PRUNED events during episode]
  --increment-->
  [EnvState.pruned_count]
  --EPISODE_ENDED-->
  [BestRunRecord(pruned_count=env.pruned_count)]
  --snapshot.best_runs-->
  [HistoricalEnvDetail widget]
  --reset-->
  [EnvState.pruned_count = 0]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `pruned_count` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].pruned_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1268 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HistoricalEnvDetail header | `widgets/historical_env_detail.py` (line 237) | `Pruned: {record.pruned_count}` with red styling |
| HistoricalEnvDetail metrics | `widgets/historical_env_detail.py` (line 282) | `Pruned: {record.pruned_count}` with red styling in Seed Counts row |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `_handle_seed_pruned()` increments `env.pruned_count` at line 1144
- [x] **Transport works** - EPISODE_ENDED handler copies value to BestRunRecord at line 1261
- [x] **Schema field exists** - `BestRunRecord.pruned_count: int = 0` at line 1268
- [x] **Default is correct** - 0 appropriate before any seeds are pruned in an episode
- [x] **Consumer reads it** - HistoricalEnvDetail displays in header and metrics section
- [x] **Display is correct** - Rendered with red styling (negative outcome)
- [x] **Reset works** - Value reset to 0 at episode boundary (line 1290)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (BestRunRecord) | `tests/telemetry/test_tele_best_run_counts.py` | `TestTELE629BestRunPrunedCount` (7 tests) | `[x]` |
| Schema/Default | `tests/telemetry/test_tele_best_run_counts.py` | `test_pruned_count_default_is_zero` | `[x]` |
| Consumer Access | `tests/telemetry/test_tele_best_run_counts.py` | `test_pruned_count_consumer_access_header` | `[x]` |
| Consumer Access | `tests/telemetry/test_tele_best_run_counts.py` | `test_pruned_count_consumer_access_metrics` | `[x]` |
| Distinction | `tests/telemetry/test_tele_best_run_counts.py` | `test_pruned_count_distinct_from_tele_504` | `[x]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically or `PYTHONPATH=src uv run python -m esper.karn.sanctum`)
3. Wait for seeds to be pruned during training
4. Click on a row in the Scoreboard to open HistoricalEnvDetail modal
5. Verify "Pruned: N" appears in header with red styling
6. Verify "Pruned: N" appears in Seed Counts row with red styling
7. Compare values across different best run records

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_PRUNED` event | event | Triggers increment of EnvState.pruned_count |
| `EnvState.pruned_count` | state | Accumulates prunes within episode |
| `EPISODE_ENDED` event | event | Triggers BestRunRecord creation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HistoricalEnvDetail display | widget | Shows pruned count in header and metrics |
| Historical analysis | research | Correlate pruning with peak accuracy |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Critical Distinction from TELE-504:**
> - **TELE-504 (`SeedLifecycleStats.prune_count`):** Cumulative count across the entire training run. Never resets. Tracks total prunes for run-level statistics.
> - **TELE-629 (`BestRunRecord.pruned_count`):** Point-in-time snapshot for a specific episode at peak accuracy. Resets each episode. Tracks prunes that occurred during that particular best run.
>
> **Per-Episode Semantics:** The count is accumulated throughout the episode via SEED_PRUNED events, then captured in BestRunRecord at EPISODE_ENDED. The EnvState.pruned_count is then reset to 0 for the next episode (line 1290 in aggregator.py).
>
> **Display Context:** HistoricalEnvDetail shows this alongside `fossilized_count` to give a complete picture of seed lifecycle outcomes at the time of peak accuracy. Red styling indicates pruning represents failed experiments.
>
> **Interpretation:** A high pruned_count relative to fossilized_count may indicate:
> - The policy struggled to develop viable seed modules during this episode
> - Aggressive pruning thresholds removed experiments too early
> - The task was particularly challenging for this environment
>
> **Wiring Status:** Fully wired and operational. Pruned count flows from SEED_PRUNED events through EnvState accumulation to BestRunRecord snapshot at EPISODE_ENDED.
