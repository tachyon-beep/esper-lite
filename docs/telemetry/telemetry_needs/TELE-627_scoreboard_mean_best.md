# Telemetry Record: [TELE-627] Scoreboard Mean Best

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-627` |
| **Name** | Scoreboard Mean Best |
| **Category** | `scoreboard` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What is the average peak accuracy across all recorded best runs?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` (derived) |
| **Units** | percentage (0.0 to 100.0) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (no runs recorded yet) |

### Semantic Meaning

> Mean best is a **derived metric** computed as `mean(peak_accuracy)` across all BestRunRecords. It provides a more robust view of overall training quality than global_best alone.
>
> - **Mean best:** Average peak accuracy across all recorded runs
> - **Consistency indicator:** High mean = most runs performed well, not just one outlier
> - **Complements global_best:** Global best shows ceiling, mean shows typical performance
>
> A high global_best with low mean_best indicates inconsistent training (one lucky run). High global_best with high mean_best indicates consistent, reproducible performance.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Excellent** | `>= 80.0` | Consistently strong performance |
| **Good** | `65.0 <= value < 80.0` | Generally good with some variation |
| **Moderate** | `50.0 <= value < 65.0` | Average performance, room for improvement |
| **Poor** | `< 50.0` | Training struggling on average |

**Threshold Source:** No explicit thresholds in code — displayed in plain style

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed at display time from best_runs list |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/scoreboard.py` |
| **Function/Method** | `Scoreboard._refresh_stats()` |
| **Line(s)** | 186 |

```python
if best_runs:
    global_best = max(r.peak_accuracy for r in best_runs)
    mean_best = sum(r.peak_accuracy for r in best_runs) / len(best_runs)  # <--
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Storage** | BestRunRecords stored in snapshot.best_runs | `karn/sanctum/schema.py` |
| **2. Computation** | mean(peak_accuracy) computed at display refresh | `karn/sanctum/widgets/scoreboard.py` |
| **3. Display** | Shown in stats header as "Mean: {value}%" | `karn/sanctum/widgets/scoreboard.py` |

```
[snapshot.best_runs[].peak_accuracy]
  --mean() aggregation-->
  [mean_best = sum(peak_accuracy) / len(best_runs)]
  --stats header-->
  ["Mean: 72.5%"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | N/A (derived, not stored) |
| **Field** | N/A — computed from `snapshot.best_runs[].peak_accuracy` |
| **Path** | Computed in widget from `snapshot.best_runs` |
| **Schema File** | N/A (derived) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard Stats | `widgets/scoreboard.py` (lines 194-199) | Displayed in header as "[dim]Mean:[/dim] {mean_best:.1f}%" |

---

## 5. Wiring Verification

### Checklist

- [x] **Source data exists** — `snapshot.best_runs` list with peak_accuracy fields
- [x] **Computation works** — `sum(peak_accuracy) / len(best_runs)` in _refresh_stats()
- [x] **Display is correct** — Plain percentage in stats header
- [x] **Fallback handling** — Falls back to env best_accuracy if no best_runs exist yet
- [x] **Division safety** — Only computed when `len(best_runs) > 0`

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_stats_header_mean_best` | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_stats_header_mean_best_fallback` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Scoreboard stats header — "Mean: X.X%" should show average peak
4. Compare to "Best: X.X%" — mean should be <= global best
5. Verify mean updates as more runs are recorded

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TELE-620 peak_accuracy | field | Individual run peak accuracies |
| snapshot.best_runs | list | All best run records for aggregation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Stats header display | display | Secondary "Mean" metric |
| Consistency analysis | research | Comparing mean to global best reveals variance |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Mean best is computed at display time rather than stored. This ensures it's always consistent with the current best_runs list without needing synchronization.
>
> **Relationship to Global Best:** The ratio `mean_best / global_best` is an implicit consistency metric:
> - Ratio near 1.0 = very consistent, all runs performing similarly
> - Ratio near 0.5 = inconsistent, global best is an outlier
>
> **Display Format:** Mean best is displayed in plain style (not bold) to visually distinguish it from the more important global_best metric.
>
> **Fallback Behavior:** Like global_best, when `best_runs` is empty, the widget falls back to computing mean from `EnvState.best_accuracy` across all environments (lines 188-191).
>
> **Note on Population:** The mean is computed over all records in best_runs, which is capped at 10 unpinned records plus any pinned records. This means the mean reflects recent performance, not necessarily the entire training history.
>
> **Wiring Status:** Fully wired and operational. This is a derived metric computed from stored peak_accuracy fields.
