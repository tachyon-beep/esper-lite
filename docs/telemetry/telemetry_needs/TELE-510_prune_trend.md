# Telemetry Record: [TELE-510] Prune Trend

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-510` |
| **Name** | Prune Trend |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the prune rate increasing, stable, or decreasing over recent training history?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `str` |
| **Units** | categorical |
| **Range** | `"rising"`, `"stable"`, `"falling"` |
| **Precision** | N/A (categorical) |
| **Default** | `"stable"` |

### Semantic Meaning

> Prune trend indicates the direction of change in the prune rate over recent training history.
>
> Computed by comparing the mean of the most recent 5 prune rate samples to the mean of the 5 samples before that. A 20% change threshold determines the trend:
>
> - **"rising"**: Recent mean exceeds older mean by >20%
> - **"falling"**: Recent mean is <20% below older mean
> - **"stable"**: Change is within +/-20% threshold
>
> The prune rate itself is computed as: `cumulative_prune_count / current_episode`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `"stable"` or `"falling"` | Prune rate is controlled or decreasing |
| **Warning** | `"rising"` | Prune rate increasing - may indicate training instability |
| **Critical** | N/A | No critical threshold defined |

**Note:** Rising prune trend is displayed with a red style and downward-right arrow in the TUI, falling with green and upward-right arrow, stable with dim style and horizontal arrow.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed in aggregator from `_prune_rate_history` deque |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `detect_rate_trend()` helper function |
| **Line(s)** | 96-116 (function), 533 (invocation) |

```python
def detect_rate_trend(history: deque[float]) -> str:
    """Detect trend in rate history (rising/stable/falling).

    Compares recent 5 samples to older 5 samples.
    """
    if len(history) < 5:
        return "stable"

    recent = list(history)[-5:]
    older = list(history)[-10:-5] if len(history) >= 10 else list(history)[:5]

    if not older:
        return "stable"

    recent_mean = sum(recent) / len(recent)
    older_mean = sum(older) / len(older)

    # 20% change threshold
    threshold = 0.2 * max(abs(older_mean), 0.01)
    diff = recent_mean - older_mean
    # ... returns "rising", "falling", or "stable"
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `SEED_PRUNED` event from `Slot.prune()` | `kasmina/slot.py:1581-1594` |
| **2. Collection** | `_cumulative_pruned` counter incremented | `aggregator.py:1145` |
| **3. Aggregation** | Prune rate computed and appended to `_prune_rate_history` | `aggregator.py:1210-1213` |
| **4. Delivery** | `detect_rate_trend(_prune_rate_history)` -> `SeedLifecycleStats.prune_trend` | `aggregator.py:533` |

```
[Slot.prune()] --SEED_PRUNED--> [Aggregator._cumulative_pruned++]
     --> [batch_completed: rate_history.append()] --> [snapshot: detect_rate_trend()]
         --> [SeedLifecycleStats.prune_trend]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `prune_trend` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.prune_trend` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 196 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py:116` | Displayed with trend arrow (arrow, color based on trend) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `Slot.prune()` emits `SEED_PRUNED` event
- [x] **Transport works** - Aggregator increments `_cumulative_pruned` on event, appends rate to history on batch_completed
- [x] **Schema field exists** - `SeedLifecycleStats.prune_trend: str = "stable"`
- [x] **Default is correct** - `"stable"` appropriate before sufficient history accumulates
- [x] **Consumer reads it** - SlotsPanel accesses `snapshot.seed_lifecycle.prune_trend`
- [x] **Display is correct** - Trend arrow and color applied via `trend_arrow()` helper
- [x] **Thresholds applied** - Rising = red arrow, Falling = green arrow, Stable = dim arrow

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | N/A | `[ ]` |
| Unit (aggregator) | N/A | N/A | `[ ]` |
| Integration (end-to-end) | N/A | N/A | `[ ]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

**Note:** No automated tests found for `prune_trend`, `detect_rate_trend`, or `SeedLifecycleStats`.

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 50`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Prune" row with trend indicator
4. Verify trend arrow updates as pruning events accumulate
5. Force high pruning (e.g., set low improvement threshold) to verify "rising" detection

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_PRUNED` events | event | Increments `_cumulative_pruned` counter |
| `BATCH_COMPLETED` events | event | Triggers rate history update |
| `_prune_rate_history` deque | state | Requires 5+ samples for non-stable trend detection |
| `current_episode` | counter | Used to compute prune rate (prunes / episode) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SlotsPanel display | display | Shows trend arrow with color coding |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial telemetry record creation |

---

## 8. Notes

> **Design Decision:** The 20% threshold for trend detection provides a balance between sensitivity and noise rejection. The minimum denominator of 0.01 prevents division issues when older mean is near zero.
>
> **Deque Size:** `_prune_rate_history` has `maxlen=20`, meaning trend detection considers at most the last 20 rate samples. With 5 samples needed for comparison, trend detection becomes meaningful after approximately 5 batches.
>
> **Rate Calculation:** Prune rate is cumulative (total prunes / total episodes), not per-batch. This means the rate naturally stabilizes over time as the denominator grows, potentially making the trend indicator less sensitive in late training.
>
> **Missing Tests:** There are no unit tests for `detect_rate_trend()` or the rate history accumulation logic. This is a gap in test coverage.
>
> **Widget Display:** The SlotsPanel shows the trend as part of a per-episode rate display line:
> ```
> Prune{arrow}{rate}/ep
> ```
> Where arrow is one of: "up-right" (rising/green), "right" (stable/dim), "down-right" (falling/red).
