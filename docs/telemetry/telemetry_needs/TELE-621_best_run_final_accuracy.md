# Telemetry Record: [TELE-621] Best Run Final Accuracy

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-621` |
| **Name** | Best Run Final Accuracy |
| **Category** | `scoreboard` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What accuracy did this environment finish with at the end of its episode?"

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

> Final accuracy represents the validation accuracy at the end of the episode. This captures where the model ended up, which may differ from peak_accuracy if the model regressed after reaching its peak.
>
> Comparing final to peak reveals trajectory:
> - **final > peak:** Still climbing (rare, possible with smoothing)
> - **final == peak:** Held gains or peaked at the end
> - **final < peak:** Model regressed from its peak
>
> Large gaps between peak and final indicate training instability or overfitting.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `final >= peak - 1.0` | Held steady (within 1% of peak) |
| **Warning** | `peak - 5.0 <= final < peak - 1.0` | Minor regression |
| **Critical** | `final < peak - 5.0` | Severe regression (>5% drop) |

**Threshold Source:** `src/esper/karn/sanctum/widgets/scoreboard.py` — `_format_trajectory()` lines 341-372

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EnvState.host_accuracy at time of EPISODE_ENDED event |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_episode_ended()` |
| **Line(s)** | 1243 |

```python
record = BestRunRecord(
    env_id=env.env_id,
    episode=episode_start + env.env_id,
    peak_accuracy=env.best_accuracy,
    final_accuracy=env.host_accuracy,  # <-- Current accuracy at episode end
    # ...
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | EPOCH_COMPLETED events update EnvState.host_accuracy | `simic/telemetry.py` |
| **2. Collection** | EnvState.add_accuracy() sets host_accuracy | `karn/sanctum/schema.py` (line 592) |
| **3. Aggregation** | EPISODE_ENDED reads current host_accuracy for final_accuracy | `karn/sanctum/aggregator.py` (line 1243) |
| **4. Delivery** | Written to `snapshot.best_runs[i].final_accuracy` | `karn/sanctum/schema.py` |

```
[EPOCH_COMPLETED events during episode]
  --payload.accuracy-->
  [EnvState.host_accuracy updated continuously]
  --EPISODE_ENDED-->
  [BestRunRecord(final_accuracy=env.host_accuracy)]
  --snapshot.best_runs-->
  [Scoreboard widget trajectory column]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `final_accuracy` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].final_accuracy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1242 |
| **Default Value** | `0.0` (not explicitly set, comes from EnvState.host_accuracy) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard | `widgets/scoreboard.py` (lines 341-372) | Displayed in "Traj" column via `_format_trajectory()` as arrow + final value |
| Worst Trajectory Panel | `widgets/scoreboard.py` (lines 293-306) | Used to filter/sort runs with regression (final - peak < -0.5) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EnvState.host_accuracy updated by add_accuracy() at line 592
- [x] **Transport works** — EPISODE_ENDED handler reads env.host_accuracy for final_accuracy
- [x] **Schema field exists** — `BestRunRecord.final_accuracy: float` at line 1242
- [x] **Default is correct** — `0.0` is appropriate before first accuracy measurement
- [x] **Consumer reads it** — Scoreboard uses final_accuracy in trajectory formatting
- [x] **Display is correct** — Shown as arrow indicator with final value (e.g., "↘85.2")
- [x] **Thresholds applied** — Color coding based on delta from peak (green/dim/yellow/red)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_env_state_host_accuracy_tracking` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_episode_ended_captures_final_accuracy` | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | Trajectory formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Scoreboard "Traj" column showing trajectory arrows
4. Green ↗ = still climbing, dim ─→ = held, yellow/red ↘ = regressed
5. Verify Worst Trajectory panel shows runs with largest regression

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED event | event | Provides accuracy values continuously |
| EnvState.host_accuracy | state | Current accuracy updated each epoch |
| EPISODE_ENDED event | event | Triggers BestRunRecord creation with final value |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| TELE-622 trajectory_delta | derived | final_accuracy - peak_accuracy |
| Worst Trajectory Panel | display | Filters/sorts by regression amount |
| Trajectory display | display | Arrow direction and color based on delta |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Final accuracy is captured at the moment EPISODE_ENDED fires, representing the last known accuracy before the environment resets. This is the natural "where did we end up" measurement.
>
> **Trajectory Calculation:** The trajectory delta (TELE-622) is `final_accuracy - peak_accuracy`. Negative values indicate regression from peak.
>
> **Worst Trajectory Panel:** The bottom panel of the Scoreboard shows runs filtered by `(final - peak) < -0.5` (at least 0.5% regression), sorted by worst regression first. This helps identify runs that lost the most ground.
>
> **Wiring Status:** Fully wired and operational. Final accuracy flows from EnvState.host_accuracy to BestRunRecord.final_accuracy at EPISODE_ENDED.
