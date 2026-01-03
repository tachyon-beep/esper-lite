# Telemetry Record: [TELE-511] Fossilize Trend

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-511` |
| **Name** | Fossilize Trend |
| **Category** | `seed` |
| **Priority** | `P2-nice-to-have` |

## 2. Purpose

### What question does this answer?

> "Is the fossilization rate increasing, stable, or decreasing over recent training history?"

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
| **Units** | categorical (trend direction) |
| **Range** | `{"rising", "stable", "falling"}` |
| **Precision** | N/A (categorical) |
| **Default** | `"stable"` |

### Semantic Meaning

> Indicates the direction of change in fossilization rate over recent training history.
> Computed by comparing the mean fossilization rate of the most recent 5 samples
> against the mean of the preceding 5 samples (or all available if less than 10).
>
> - **"rising"**: Recent fossilization rate is >20% higher than older rate
> - **"stable"**: Rate change is within +/-20% threshold
> - **"falling"**: Recent fossilization rate is >20% lower than older rate
>
> In the botanical lifecycle metaphor:
> - Rising trend suggests improving module integration success
> - Falling trend may indicate training stability issues or policy regression
> - Stable trend indicates consistent fossilization behavior

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `"rising"` or `"stable"` | Fossilization rate is improving or consistent |
| **Warning** | `"falling"` | Fossilization rate declining - may indicate issues |
| **Critical** | N/A | No critical threshold defined |

**Threshold Source:** `SlotsPanel.trend_arrow()` maps trends to visual indicators:
- `"rising"` -> green arrow (↗)
- `"stable"` -> dim arrow (→)
- `"falling"` -> red arrow (↘)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from `fossilize_rate` history in aggregator |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `detect_rate_trend()` |
| **Line(s)** | 96-121 |

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
| **1. Emission** | `SEED_FOSSILIZED` event increments `_cumulative_fossilized` | `aggregator.py:1085` |
| **2. Collection** | `_fossilize_rate_history.append(foss_rate)` on batch completion | `aggregator.py:1214` |
| **3. Aggregation** | `detect_rate_trend(self._fossilize_rate_history)` | `aggregator.py:534` |
| **4. Delivery** | Written to `SeedLifecycleStats.fossilize_trend` | `aggregator.py:534` |

```
[SEED_FOSSILIZED] --> [_cumulative_fossilized++] --> [BATCH_EPOCH_COMPLETED]
    --> [foss_rate = _cumulative_fossilized / episodes] --> [_fossilize_rate_history.append()]
    --> [detect_rate_trend()] --> [SeedLifecycleStats.fossilize_trend]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `fossilize_trend` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.fossilize_trend` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 197 |

```python
@dataclass
class SeedLifecycleStats:
    """Seed lifecycle aggregate metrics for TamiyoBrain display."""
    # ...
    # Trend indicators (computed from rate history)
    germination_trend: str = "stable"  # "rising", "stable", "falling"
    prune_trend: str = "stable"
    fossilize_trend: str = "stable"  # <-- Line 197
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displays trend arrow with color coding |

```python
# Lines 106-126 in slots_panel.py
def trend_arrow(trend: str) -> tuple[str, str]:
    """Return (arrow, style) for trend."""
    if trend == "rising":
        return "↗", "green"
    elif trend == "falling":
        return "↘", "red"
    else:
        return "→", "dim"

# ...
f_arrow, f_style = trend_arrow(lifecycle.fossilize_trend)
# ...
result.append(f"Foss{f_arrow}", style=f_style)
result.append(f"{lifecycle.fossilize_rate:.2f}/ep", style="dim")
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `SEED_FOSSILIZED` events increment `_cumulative_fossilized`, rates computed on batch completion
- [x] **Transport works** — Rate history collected and trend computed via `detect_rate_trend()`
- [x] **Schema field exists** — `SeedLifecycleStats.fossilize_trend: str = "stable"`
- [x] **Default is correct** — "stable" appropriate before enough data for trend detection
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.seed_lifecycle.fossilize_trend`
- [x] **Display is correct** — Trend arrow with color coding rendered
- [x] **Thresholds applied** — Visual indicators (green/dim/red arrows) based on trend value

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | No `detect_rate_trend` unit test found | `[ ]` |
| Unit (aggregator) | N/A | No SeedLifecycleStats trend test found | `[ ]` |
| Integration (end-to-end) | N/A | No end-to-end trend test found | `[ ]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Foss" row with trend arrow
4. Wait for sufficient fossilization events (need 5+ rate samples)
5. Verify trend arrow updates as fossilization rate changes
6. Verify colors: green for rising, dim for stable, red for falling

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-503` fossilize_count | telemetry | Cumulative fossilization count used to compute rate |
| `TELE-505` fossilize_rate | telemetry | Rate values stored in history for trend computation |
| `BATCH_EPOCH_COMPLETED` event | event | Triggers rate history update |
| `_fossilize_rate_history` | internal | Deque(maxlen=20) storing rate samples |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SlotsPanel trend display | display | Visual indicator of fossilization trend direction |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** Trend detection uses a 20% threshold to avoid noise from small fluctuations. The rate history has a maxlen of 20 to provide sufficient window for trend detection while remaining responsive to recent changes.
>
> **Algorithm Details:**
> - Requires minimum 5 samples before trend can change from "stable"
> - Compares mean of most recent 5 samples to mean of preceding 5 samples
> - Uses 20% relative threshold (with 0.01 floor for near-zero rates)
>
> **Widget Display:** The SlotsPanel displays fossilize_trend as an arrow indicator:
> - ↗ (green) for "rising" - positive trend
> - → (dim) for "stable" - neutral trend
> - ↘ (red) for "falling" - negative trend
>
> **Related Metrics:** This trend is computed alongside:
> - `germination_trend` (uses same algorithm on germination_rate_history)
> - `prune_trend` (uses same algorithm on prune_rate_history)
>
> **Test Gap:** No unit tests exist for `detect_rate_trend()` function or trend computation in aggregator. Consider adding:
> - Unit tests for `detect_rate_trend()` with various history patterns
> - Integration tests verifying trend updates after fossilization events
