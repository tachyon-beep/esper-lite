# Telemetry Record: [TELE-509] Germination Trend

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-509` |
| **Name** | Germination Trend |
| **Category** | `seed` |
| **Priority** | `P2-nice-to-have` |

## 2. Purpose

### What question does this answer?

> "Is the rate of seed germination increasing, stable, or decreasing over recent training history?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
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
| **Units** | categorical indicator |
| **Range** | `"rising"`, `"stable"`, `"falling"` |
| **Precision** | N/A (categorical) |
| **Default** | `"stable"` |

### Semantic Meaning

> Indicates the trend direction of the germination rate (germinations per episode) over recent training history.
>
> The trend is computed by comparing the mean of the 5 most recent germination rate samples to the mean of the previous 5 samples. A 20% change threshold determines whether the trend is "rising", "falling", or "stable".
>
> **Interpretation:**
> - **rising**: Germination rate is increasing - more seeds being activated per episode
> - **stable**: Germination rate is relatively constant
> - **falling**: Germination rate is decreasing - fewer seeds being activated per episode
>
> A falling trend near training end is expected as slots become fossilized. A falling trend early may indicate slot exhaustion or policy collapse.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `"stable"` or `"rising"` early, `"falling"` late | Normal training progression |
| **Warning** | `"falling"` in early/mid training | May indicate premature slot saturation |
| **Critical** | N/A | No critical threshold for trend alone |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from germination rate history in aggregator |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `detect_rate_trend()` + `SanctumAggregator._build_snapshot()` |
| **Line(s)** | 96-121 (detect_rate_trend), 532 (trend assignment) |

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

    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    return "stable"
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `SEED_GERMINATED` event from Slot | `kasmina/slot.py:1365-1381` |
| **2. Collection** | Event handled, increments `_cumulative_germinated` | `karn/sanctum/aggregator.py:1016` |
| **3. Aggregation** | Rate history updated on `BATCH_COMPLETED` | `karn/sanctum/aggregator.py:1207-1214` |
| **4. Delivery** | Trend computed via `detect_rate_trend()` | `karn/sanctum/aggregator.py:532` |

```
[Slot.germinate()] --SEED_GERMINATED--> [Aggregator._handle_event()]
    --> _cumulative_germinated += 1
    --> (on BATCH_COMPLETED) _germination_rate_history.append(rate)
    --> [_build_snapshot()] --> detect_rate_trend(_germination_rate_history)
    --> [SeedLifecycleStats.germination_trend]
```

### Upstream Data Dependencies

The trend is derived from `_germination_rate_history`, which is populated during `BATCH_COMPLETED` event handling:

```python
# aggregator.py:1207-1214
if self._current_episode > 0:
    germ_rate = self._cumulative_germinated / self._current_episode
    prune_rate = self._cumulative_pruned / self._current_episode
    foss_rate = self._cumulative_fossilized / self._current_episode
    self._germination_rate_history.append(germ_rate)
    self._prune_rate_history.append(prune_rate)
    self._fossilize_rate_history.append(foss_rate)
```

The history is a `deque[float]` with `maxlen=20`, storing the 20 most recent germination rate values.

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `germination_trend: str = "stable"` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.germination_trend` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 195 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py:115` | Displayed with trend arrow (rising=green, stable=dim, falling=red) |

```python
# slots_panel.py:106-119
def trend_arrow(trend: str) -> tuple[str, str]:
    """Return (arrow, style) for trend."""
    if trend == "rising":
        return "↗", "green"
    elif trend == "falling":
        return "↘", "red"
    else:
        return "→", "dim"

g_arrow, g_style = trend_arrow(lifecycle.germination_trend)
# ...
result.append(f"Germ{g_arrow}", style=g_style)
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `SEED_GERMINATED` events increment `_cumulative_germinated`
- [x] **Transport works** — Rate history populated on `BATCH_COMPLETED`
- [x] **Schema field exists** — `SeedLifecycleStats.germination_trend: str = "stable"`
- [x] **Default is correct** — `"stable"` appropriate before history accumulates
- [x] **Consumer reads it** — SlotsPanel accesses `lifecycle.germination_trend`
- [x] **Display is correct** — Trend arrow renders with appropriate style
- [ ] **Thresholds applied** — No color thresholds beyond trend arrow styling

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (detect_rate_trend) | — | — | `[ ]` Not found |
| Unit (aggregator) | — | — | `[ ]` Not found |
| Integration (end-to-end) | — | — | `[ ]` Not found |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 50`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel lifecycle section
4. Verify "Germ" shows trend arrow (right arrow when stable)
5. During active germination period, verify arrow may show rising/falling as rate changes
6. After many episodes with fossilized slots, verify trend shows falling or stable

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-505` germination_count | telemetry | Base count used to compute rate |
| `BATCH_COMPLETED` event | event | Triggers rate history update |
| `_germination_rate_history` | internal | deque(maxlen=20) storing rate samples |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SlotsPanel display | display | Visual trend indicator |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation during tele_audit_2 |

---

## 8. Notes

> **Design Decision:** The 20% threshold for trend detection balances sensitivity with noise reduction. The 5-sample window comparison provides enough smoothing to avoid spurious trend changes while still responding to meaningful rate shifts.
>
> **Warmup Behavior:** Returns `"stable"` until at least 5 rate samples have been collected, preventing false trend signals during early training.
>
> **Test Gap:** No unit tests exist for `detect_rate_trend()` or the germination trend wiring. Consider adding:
> - Unit test for `detect_rate_trend()` with various history patterns
> - Integration test verifying trend updates after germination events
>
> **Related Metrics:** `prune_trend` (TELE-510) and `fossilize_trend` (TELE-511) use the same `detect_rate_trend()` function with their respective rate histories.
