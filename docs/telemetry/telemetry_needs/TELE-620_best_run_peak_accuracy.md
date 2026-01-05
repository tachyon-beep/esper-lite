# Telemetry Record: [TELE-620] Best Run Peak Accuracy

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-620` |
| **Name** | Best Run Peak Accuracy |
| **Category** | `scoreboard` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What is the highest accuracy this environment achieved during its episode?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [x] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | percentage (0.0 to 100.0) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before first accuracy measurement) |

### Semantic Meaning

> Peak accuracy represents the highest validation accuracy achieved by the environment during a single episode. This captures the "high water mark" for the run, which may differ from the final accuracy if the model regressed after reaching its peak.
>
> - **High peak (>80%):** Strong performance achieved at some point
> - **Low peak (<50%):** Episode never achieved good accuracy
> - **Peak == Final:** Model held its gains or is still improving
> - **Peak > Final:** Model regressed from peak (trajectory delta negative)
>
> Peak accuracy is the primary metric for the Best Runs leaderboard ranking.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Excellent** | `>= 85.0` | Outstanding performance |
| **Healthy** | `70.0 <= value < 85.0` | Good performance |
| **Warning** | `50.0 <= value < 70.0` | Mediocre performance |
| **Critical** | `< 50.0` | Poor performance |

**Threshold Source:** Display conventions in `src/esper/karn/sanctum/widgets/scoreboard.py` - peak displayed in bold green

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EnvState.add_accuracy() updates best_accuracy when new high is achieved |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState.add_accuracy()` |
| **Line(s)** | 588-618 |

```python
def add_accuracy(self, accuracy: float, epoch: int, episode: int = 0) -> None:
    """Add accuracy and update best/status tracking."""
    # ...
    if accuracy > self.best_accuracy:
        self.best_accuracy = accuracy
        self.best_accuracy_epoch = epoch
        self.best_accuracy_episode = episode
        # ... snapshot best_seeds and volatile state
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Accuracy update from EPOCH_COMPLETED event | `simic/telemetry.py` |
| **2. Collection** | EnvState.add_accuracy() tracks best_accuracy | `karn/sanctum/schema.py` |
| **3. Aggregation** | EPISODE_ENDED handler creates BestRunRecord with peak_accuracy | `karn/sanctum/aggregator.py` (line 1239-1267) |
| **4. Delivery** | Written to `snapshot.best_runs[i].peak_accuracy` | `karn/sanctum/schema.py` |

```
[EPOCH_COMPLETED event]
  --payload.accuracy-->
  [SanctumAggregator.handle_epoch_completed()]
  --EnvState.add_accuracy()-->
  [EnvState.best_accuracy updated]
  --EPISODE_ENDED-->
  [BestRunRecord(peak_accuracy=env.best_accuracy)]
  --snapshot.best_runs-->
  [Scoreboard widget]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `peak_accuracy` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].peak_accuracy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1241 |
| **Default Value** | `0.0` (from EnvState.best_accuracy default) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard | `widgets/scoreboard.py` (lines 241-249) | Displayed in "Peak" column as `[bold green]{peak_accuracy:.1f}[/bold green]` |
| Scoreboard Stats | `widgets/scoreboard.py` (lines 185-186) | Used to compute global_best and mean_best in stats header |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EnvState.add_accuracy() tracks best_accuracy at line 595-598
- [x] **Transport works** — EPOCH_COMPLETED event triggers add_accuracy(), EPISODE_ENDED creates BestRunRecord
- [x] **Schema field exists** — `BestRunRecord.peak_accuracy: float` at line 1241
- [x] **Default is correct** — `0.0` is appropriate before first accuracy measurement
- [x] **Consumer reads it** — Scoreboard displays peak_accuracy in table and stats header
- [x] **Display is correct** — Rendered as bold green percentage with 1 decimal place
- [x] **Thresholds applied** — Display always uses green for peak (best is always good to show)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_env_state_add_accuracy_updates_best` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_episode_ended_creates_best_run_record` | `[ ]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_backend.py` | Telemetry flow integration | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | Peak accuracy display | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically or `PYTHONPATH=src uv run python -m esper.karn.sanctum`)
3. Observe Scoreboard widget — "BEST RUNS" panel shows Peak column
4. Verify highest Peak values appear at top of leaderboard
5. After several episodes, query telemetry: `SELECT peak_accuracy FROM best_runs ORDER BY peak_accuracy DESC LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED event | event | Provides accuracy values for tracking |
| EnvState.best_accuracy | state | Tracks running best within episode |
| EPISODE_ENDED event | event | Triggers BestRunRecord creation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| TELE-626 scoreboard_global_best | derived | max(peak_accuracy) across all records |
| TELE-627 scoreboard_mean_best | derived | mean(peak_accuracy) across all records |
| TELE-622 trajectory_delta | derived | final_accuracy - peak_accuracy |
| Leaderboard ranking | display | Records sorted by peak_accuracy descending |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Peak accuracy is captured at the moment the new best is achieved (not at episode end). This ensures the snapshot of contributing seeds, reward components, and other volatile state accurately reflects the system state at the peak.
>
> **Leaderboard Behavior:** The Best Runs leaderboard maintains up to 10 records. Records are sorted by peak_accuracy descending, so the highest-performing runs appear at the top.
>
> **Interactive Features:** Clicking a row in the Scoreboard opens a historical detail modal showing the full EnvState snapshot at the time peak_accuracy was achieved.
>
> **Wiring Status:** Fully wired and operational. Peak accuracy flows from EPOCH_COMPLETED through EnvState.add_accuracy() to BestRunRecord creation at EPISODE_ENDED.
