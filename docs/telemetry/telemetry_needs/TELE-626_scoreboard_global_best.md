# Telemetry Record: [TELE-626] Scoreboard Global Best

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-626` |
| **Name** | Scoreboard Global Best |
| **Category** | `scoreboard` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What is the highest peak accuracy achieved across all runs in this training session?"

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

> Global best is a **derived metric** computed as `max(peak_accuracy)` across all BestRunRecords. It represents the absolute best performance achieved in the training session.
>
> - **Global best:** The single highest peak accuracy across all environments and episodes
> - **Session summary:** Quick answer to "what's our best result so far?"
> - **Progress tracking:** When global best increases, training is making progress
>
> This is the headline metric for the scoreboard — the number operators care about most.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Excellent** | `>= 85.0` | Outstanding session performance |
| **Good** | `70.0 <= value < 85.0` | Solid session performance |
| **Moderate** | `50.0 <= value < 70.0` | Acceptable but room for improvement |
| **Poor** | `< 50.0` | Session struggling, may need intervention |

**Threshold Source:** Display conventions in `src/esper/karn/sanctum/widgets/scoreboard.py` — displayed in bold green

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed at display time from best_runs list |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/scoreboard.py` |
| **Function/Method** | `Scoreboard._refresh_stats()` |
| **Line(s)** | 185-186 |

```python
if best_runs:
    global_best = max(r.peak_accuracy for r in best_runs)
    mean_best = sum(r.peak_accuracy for r in best_runs) / len(best_runs)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Storage** | BestRunRecords stored in snapshot.best_runs | `karn/sanctum/schema.py` |
| **2. Computation** | max(peak_accuracy) computed at display refresh | `karn/sanctum/widgets/scoreboard.py` |
| **3. Display** | Shown in stats header as "Best: {value}%" | `karn/sanctum/widgets/scoreboard.py` |

```
[snapshot.best_runs[].peak_accuracy]
  --max() aggregation-->
  [global_best = max(r.peak_accuracy for r in best_runs)]
  --stats header-->
  ["Best: 87.3%"]
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
| Scoreboard Stats | `widgets/scoreboard.py` (lines 194-199) | Displayed in header as "[dim]Best:[/dim] [bold green]{global_best:.1f}%[/bold green]" |

---

## 5. Wiring Verification

### Checklist

- [x] **Source data exists** — `snapshot.best_runs` list with peak_accuracy fields
- [x] **Computation works** — `max(r.peak_accuracy for r in best_runs)` in _refresh_stats()
- [x] **Display is correct** — Bold green percentage in stats header
- [x] **Fallback handling** — Falls back to env best_accuracy if no best_runs exist yet

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_stats_header_global_best` | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_stats_header_empty_runs` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Scoreboard stats header — "Best: X.X%" should show global maximum
4. Verify global best updates when a new record breaks the previous high
5. Verify display is bold green

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
| Stats header display | display | Headline "Best" metric |
| Session summary | analysis | Overall training performance |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Global best is computed at display time rather than stored. This ensures it's always consistent with the current best_runs list without needing synchronization.
>
> **Fallback Behavior:** When `best_runs` is empty (early in training), the widget falls back to computing global_best from `EnvState.best_accuracy` across all environments (lines 188-191). This ensures the stats header shows meaningful data even before EPISODE_ENDED events create BestRunRecords.
>
> **Display Format:** The global best is prominently displayed in bold green in the stats header, making it immediately visible as the session's headline metric.
>
> **Wiring Status:** Fully wired and operational. This is a derived metric computed from stored peak_accuracy fields.
