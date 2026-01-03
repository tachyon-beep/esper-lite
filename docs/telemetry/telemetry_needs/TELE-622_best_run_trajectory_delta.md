# Telemetry Record: [TELE-622] Best Run Trajectory Delta

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-622` |
| **Name** | Best Run Trajectory Delta |
| **Category** | `scoreboard` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "How much did this run regress (or improve) from its peak accuracy to its final accuracy?"

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
| **Type** | `float` (derived) |
| **Units** | percentage points |
| **Range** | Typically `[-100.0, +10.0]` — large negative = severe regression |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (when peak == final) |

### Semantic Meaning

> Trajectory delta is a **derived metric** computed as `final_accuracy - peak_accuracy`. It measures how much the model's accuracy changed from its peak to the end of the episode.
>
> - **Positive delta (>+0.5%):** Still climbing at episode end (green ↗)
> - **Near zero (-1.0% to +0.5%):** Held steady at peak (dim ─→)
> - **Small negative (-2.0% to -1.0%):** Minor regression (dim ↘)
> - **Moderate negative (-5.0% to -2.0%):** Moderate regression (yellow ↘)
> - **Large negative (<-5.0%):** Severe regression (red ↘)
>
> This metric is critical for identifying runs with training instability or late-stage overfitting.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Climbing** | `delta > +0.5%` | Still improving at episode end |
| **Held** | `-1.0% <= delta <= +0.5%` | Maintained peak performance |
| **Minor Drop** | `-2.0% <= delta < -1.0%` | Small regression |
| **Moderate Drop** | `-5.0% <= delta < -2.0%` | Moderate regression |
| **Severe Drop** | `delta < -5.0%` | Severe regression, indicates instability |

**Threshold Source:** `src/esper/karn/sanctum/widgets/scoreboard.py` — `_format_trajectory()` lines 354-372

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed at display time from peak_accuracy and final_accuracy |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/scoreboard.py` |
| **Function/Method** | `Scoreboard._format_trajectory()` |
| **Line(s)** | 341-372 |

```python
def _format_trajectory(self, record: "BestRunRecord") -> str:
    peak = record.peak_accuracy
    final = record.final_accuracy
    delta = final - peak  # <-- Derived computation

    if delta > 0.5:
        return f"[green]↗{final:.1f}[/green]"
    elif delta >= -1.0:
        return f"[dim]─→{final:.1f}[/dim]"
    # ... etc
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Storage** | peak_accuracy and final_accuracy stored in BestRunRecord | `karn/sanctum/schema.py` |
| **2. Computation** | delta computed at display time: `final - peak` | `karn/sanctum/widgets/scoreboard.py` |
| **3. Display** | Arrow indicator + color based on delta thresholds | `karn/sanctum/widgets/scoreboard.py` |

```
[BestRunRecord.peak_accuracy, BestRunRecord.final_accuracy]
  --delta = final - peak-->
  [_format_trajectory()]
  --threshold comparison-->
  [colored arrow + final value: "↘85.2"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` (derived, not stored) |
| **Field** | N/A — computed from `peak_accuracy` and `final_accuracy` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].final_accuracy - snapshot.best_runs[i].peak_accuracy` |
| **Schema File** | N/A (derived) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard | `widgets/scoreboard.py` (lines 341-372) | Arrow direction and color in "Traj" column |
| Worst Trajectory Panel | `widgets/scoreboard.py` (lines 293-306) | Filters records with delta < -0.5%, sorts by delta ascending |

---

## 5. Wiring Verification

### Checklist

- [x] **Source fields exist** — `BestRunRecord.peak_accuracy` and `BestRunRecord.final_accuracy`
- [x] **Computation works** — `_format_trajectory()` computes delta correctly
- [x] **Display is correct** — Arrow indicators (↗, ─→, ↘) with appropriate colors
- [x] **Thresholds applied** — Five-tier threshold system: green/dim/dim/yellow/red
- [x] **Worst Trajectory filter** — Correctly filters delta < -0.5% and sorts ascending

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (widget) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_format_trajectory_thresholds` | `[ ]` |
| Unit (widget) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_worst_trajectory_filtering` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 20`
2. Launch Sanctum TUI
3. Observe Scoreboard "Traj" column — verify arrow directions match peak/final relationship
4. Check color coding:
   - Green ↗ for final > peak + 0.5%
   - Dim ─→ for held (within 1%)
   - Yellow/red ↘ for moderate/severe regression
5. Verify Worst Trajectory panel only shows runs with at least 0.5% regression

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TELE-620 peak_accuracy | field | Peak accuracy for delta calculation |
| TELE-621 final_accuracy | field | Final accuracy for delta calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Trajectory display | display | Arrow direction and color |
| Worst Trajectory Panel | display | Filtering and sorting of regressed runs |
| Training stability analysis | research | Identifies unstable runs |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Trajectory delta is computed at display time rather than stored. This ensures it's always consistent with peak and final values without needing synchronization.
>
> **Threshold Rationale:**
> - **+0.5%:** Small positive buffer to distinguish "still climbing" from noise
> - **-1.0%:** Within normal variation, considered "held steady"
> - **-2.0%:** Noticeable but not alarming regression
> - **-5.0%:** Significant regression worthy of attention
> - **<-5.0%:** Severe regression indicating training failure or instability
>
> **Worst Trajectory Panel:** The 0.5% threshold for filtering (`delta < -0.5`) excludes runs that only had minor fluctuation. The panel shows the 5 worst regressions, helping operators quickly identify problematic runs.
>
> **Wiring Status:** Fully wired and operational. This is a derived metric computed from stored fields.
