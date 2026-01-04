# Telemetry Record: [TELE-624] Best Run Episode

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-624` |
| **Name** | Best Run Episode |
| **Category** | `scoreboard` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "Which episode (batch) produced this best run record?"

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
| **Units** | episode number (0-indexed internally, 1-indexed for display) |
| **Range** | `[0, max_episodes)` |
| **Precision** | Integer |
| **Default** | `0` (first episode) |

### Semantic Meaning

> Episode number identifies which training episode (batch) produced this best run record. This allows operators to:
>
> - **Track progress:** See when best performances occurred
> - **Identify trends:** Earlier episodes with high accuracy may indicate fast convergence
> - **Debug issues:** Late episode peaks may indicate slow learning or lucky initialization
>
> Note: Episode is 0-indexed internally but displayed as 1-indexed (episode 0 shows as "Ep 1").

### Health Thresholds

N/A — Episode number is an identifier, not a health metric.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from episode_start + env_id at EPISODE_ENDED |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_episode_ended()` |
| **Line(s)** | 1241 |

```python
record = BestRunRecord(
    env_id=env.env_id,
    episode=episode_start + env.env_id,  # <-- Global episode number
    peak_accuracy=env.best_accuracy,
    # ...
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | EPISODE_ENDED event provides episode_start from payload | `simic/telemetry.py` |
| **2. Aggregation** | Aggregator computes global episode: episode_start + env_id | `karn/sanctum/aggregator.py` |
| **3. Delivery** | Written to `snapshot.best_runs[i].episode` | `karn/sanctum/schema.py` |

```
[EPISODE_ENDED payload.episode_start]
  --plus env.env_id-->
  [BestRunRecord.episode = episode_start + env_id]
  --snapshot.best_runs-->
  [Scoreboard "Ep" column: episode + 1]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `episode` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].episode` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1239 |
| **Default Value** | N/A (always set from event) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard | `widgets/scoreboard.py` (line 243) | Displayed in "Ep" column as `str(record.episode + 1)` (1-indexed) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EPISODE_ENDED event provides episode_start
- [x] **Transport works** — Aggregator computes episode = episode_start + env_id
- [x] **Schema field exists** — `BestRunRecord.episode: int` at line 1239
- [x] **Consumer reads it** — Scoreboard displays as 1-indexed episode number
- [x] **Display is correct** — Plain numeric display, no color coding

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_episode_ended_episode_numbering` | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_episode_display_one_indexed` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Scoreboard "Ep" column showing episode numbers
4. Verify episode numbers are 1-indexed (first episode shows "1", not "0")
5. Verify episodes increment across batches

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPISODE_ENDED event | event | Provides episode_start |
| EnvState.env_id | field | Used to compute global episode number |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Scoreboard display | display | Episode identification |
| Historical analysis | research | Correlating performance with training progress |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Episode number is computed as `episode_start + env_id` to provide a globally unique episode identifier. With N environments running in parallel, each environment gets a different episode number per batch.
>
> **1-Indexed Display:** The internal 0-indexed episode is displayed as 1-indexed for user friendliness. This matches user expectations ("Episode 1" instead of "Episode 0").
>
> **Multi-Environment Context:** In vectorized training with N environments, a single batch may produce N best run records with consecutive episode numbers (episode_start, episode_start+1, ..., episode_start+N-1).
>
> **Wiring Status:** Fully wired and operational. Episode is set at BestRunRecord creation from EPISODE_ENDED payload.
