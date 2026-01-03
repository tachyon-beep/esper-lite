# Telemetry Record: [TELE-628] Best Run Fossilized Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-628` |
| **Name** | Best Run Fossilized Count |
| **Category** | `episode` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds had fossilized by the time this environment reached its peak accuracy?"

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

> Fossilized count in a BestRunRecord represents the **point-in-time snapshot** of how many seeds had successfully fossilized (permanently integrated with the host) by the time the environment achieved its peak accuracy during that episode.
>
> This differs fundamentally from TELE-503 (`SeedLifecycleStats.fossilize_count`):
> - **TELE-503:** Cumulative count for the entire training run, continuously incrementing
> - **TELE-628:** Snapshot count at peak accuracy for a specific environment/episode
>
> Use cases:
> - **Correlation analysis:** Did higher fossilized counts correlate with better peak accuracy?
> - **Timing insights:** Were successful runs characterized by early or late fossilization?
> - **Historical comparison:** Compare seed progression across top-performing runs

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Context-dependent** | N/A | Value interpreted relative to pruned_count and total slots |

**Threshold Source:** Display always uses green styling (fossilization is a positive outcome)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EnvState.fossilized_count accumulated during episode |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_seed_fossilized()` |
| **Line(s)** | 1084 |

```python
# In _handle_seed_fossilized():
env.fossilized_count += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `SEED_FOSSILIZED` event increments `EnvState.fossilized_count` | `karn/sanctum/aggregator.py` (line 1084) |
| **2. Accumulation** | Count accumulates in EnvState throughout episode | `karn/sanctum/schema.py` (line 459) |
| **3. Aggregation** | EPISODE_ENDED handler copies to BestRunRecord | `karn/sanctum/aggregator.py` (line 1260) |
| **4. Delivery** | Written to `snapshot.best_runs[i].fossilized_count` | `karn/sanctum/schema.py` (line 1267) |
| **5. Reset** | EnvState.fossilized_count reset to 0 for next episode | `karn/sanctum/aggregator.py` (line 1289) |

```
[SEED_FOSSILIZED events during episode]
  --increment-->
  [EnvState.fossilized_count]
  --EPISODE_ENDED-->
  [BestRunRecord(fossilized_count=env.fossilized_count)]
  --snapshot.best_runs-->
  [HistoricalEnvDetail widget]
  --reset-->
  [EnvState.fossilized_count = 0]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `fossilized_count` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].fossilized_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1267 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HistoricalEnvDetail header | `widgets/historical_env_detail.py` (line 235) | `Fossilized: {record.fossilized_count}` with green styling |
| HistoricalEnvDetail metrics | `widgets/historical_env_detail.py` (line 280) | `Fossilized: {record.fossilized_count}` with green styling in Seed Counts row |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `_handle_seed_fossilized()` increments `env.fossilized_count` at line 1084
- [x] **Transport works** - EPISODE_ENDED handler copies value to BestRunRecord at line 1260
- [x] **Schema field exists** - `BestRunRecord.fossilized_count: int = 0` at line 1267
- [x] **Default is correct** - 0 appropriate before any seeds fossilize in an episode
- [x] **Consumer reads it** - HistoricalEnvDetail displays in header and metrics section
- [x] **Display is correct** - Rendered with green styling (positive outcome)
- [x] **Reset works** - Value reset to 0 at episode boundary (line 1289)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (BestRunRecord) | `tests/telemetry/test_tele_best_run_counts.py` | `TestTELE628BestRunFossilizedCount` (7 tests) | `[x]` |
| Schema/Default | `tests/telemetry/test_tele_best_run_counts.py` | `test_fossilized_count_default_is_zero` | `[x]` |
| Consumer Access | `tests/telemetry/test_tele_best_run_counts.py` | `test_fossilized_count_consumer_access_header` | `[x]` |
| Consumer Access | `tests/telemetry/test_tele_best_run_counts.py` | `test_fossilized_count_consumer_access_metrics` | `[x]` |
| Distinction | `tests/telemetry/test_tele_best_run_counts.py` | `test_fossilized_count_distinct_from_tele_503` | `[x]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically or `PYTHONPATH=src uv run python -m esper.karn.sanctum`)
3. Wait for seeds to fossilize during training
4. Click on a row in the Scoreboard to open HistoricalEnvDetail modal
5. Verify "Fossilized: N" appears in header with green styling
6. Verify "Fossilized: N" appears in Seed Counts row with green styling
7. Compare values across different best run records

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_FOSSILIZED` event | event | Triggers increment of EnvState.fossilized_count |
| `EnvState.fossilized_count` | state | Accumulates fossilizations within episode |
| `EPISODE_ENDED` event | event | Triggers BestRunRecord creation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HistoricalEnvDetail display | widget | Shows fossilized count in header and metrics |
| Historical analysis | research | Correlate fossilization with peak accuracy |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Critical Distinction from TELE-503:**
> - **TELE-503 (`SeedLifecycleStats.fossilize_count`):** Cumulative count across the entire training run. Never resets. Tracks total fossilizations for run-level statistics.
> - **TELE-628 (`BestRunRecord.fossilized_count`):** Point-in-time snapshot for a specific episode at peak accuracy. Resets each episode. Tracks fossilizations contributing to that particular best run.
>
> **Per-Episode Semantics:** The count is accumulated throughout the episode via SEED_FOSSILIZED events, then captured in BestRunRecord at EPISODE_ENDED. The EnvState.fossilized_count is then reset to 0 for the next episode (line 1289 in aggregator.py).
>
> **Display Context:** HistoricalEnvDetail shows this alongside `pruned_count` to give a complete picture of seed lifecycle outcomes at the time of peak accuracy. Green styling indicates fossilization is a positive outcome.
>
> **Wiring Status:** Fully wired and operational. Fossilized count flows from SEED_FOSSILIZED events through EnvState accumulation to BestRunRecord snapshot at EPISODE_ENDED.
