# Telemetry Record: [TELE-227] Return Variance

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-227` |
| **Name** | Return Variance |
| **Category** | `value` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How consistent is the policy? Is there high variability in episode outcomes that might indicate instability?"

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
| **Type** | `float` |
| **Units** | squared episode return units |
| **Range** | `[0, +inf)` (non-negative) |
| **Precision** | 1 decimal place for display (shown as sqrt for std) |
| **Default** | `0.0` (before return data available) |

### Semantic Meaning

> Return Variance measures the spread of episode returns around their mean:
>
> **Variance = E[(G - E[G])^2]**
>
> Where G = episode return.
>
> - **Low variance:** Consistent policy, similar performance across episodes
> - **High variance (> 100):** Inconsistent policy, performance varies widely
>
> The widget displays sqrt(variance) as "Ret sigma" for easier interpretation, since standard deviation is in the same units as returns.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value <= 100` | Displayed in cyan; policy is reasonably consistent |
| **Warning** | `value > 100` | Displayed in yellow; high variability in outcomes |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` (line 111) - yellow if variance > 100

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | NOT YET IMPLEMENTED |
| **File** | N/A |
| **Function/Method** | N/A |
| **Line(s)** | N/A |

```python
# PLANNED IMPLEMENTATION:
# Compute from episode return buffer (last N completed episodes)
#
# episode_returns = np.array([...])  # Collected from episode completions
# return_variance = episode_returns.var()
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 756) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.return_variance` | `karn/sanctum/schema.py` |

```
[Episode completion - PLANNED]
  --return_variance-->
  [emit_episode_stats_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.return_variance]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `return_variance` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.return_variance` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 756 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 110-112) | Displayed as "Ret sigma" (sqrt of variance) in yellow if variance > 100, otherwise cyan |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes return_variance
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.return_variance: float = 0.0` at line 756
- [x] **Default is correct** — `0.0` is appropriate before return data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.return_variance`
- [x] **Display is correct** — Panel displays sqrt(variance) as sigma with appropriate color
- [x] **Thresholds applied** — Consumer uses variance > 100 threshold for yellow warning

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | NOT IMPLEMENTED | `[ ]` |
| Unit (aggregator) | N/A | NOT IMPLEMENTED | `[ ]` |
| Integration (end-to-end) | N/A | NOT IMPLEMENTED | `[ ]` |
| Widget (ValueDiagnosticsPanel) | `tests/karn/sanctum/widgets/tamiyo_brain/test_value_diagnostics_panel.py` | Rendering test | `[ ]` |

### Manual Verification Steps

1. **BLOCKED:** Emitter not implemented - cannot verify end-to-end
2. Once implemented: Start training and observe ValueDiagnosticsPanel
3. "Ret sigma" should decrease as policy converges to consistent behavior
4. Yellow warning appears when variance > 100 (sigma > 10)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Episode completions | event | Requires completed episode returns |
| Return buffer | data | Buffer of recent episode returns (e.g., last 100) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ValueDiagnosticsPanel display | display | Shows return variability |
| Policy consistency assessment | analysis | High variance = inconsistent policy |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Display Transformation:** The widget displays sqrt(variance) as "Ret sigma" because:
> 1. Standard deviation is in the same units as returns (more intuitive)
> 2. Easier to compare to mean return
> 3. sqrt(100) = 10, so threshold of variance > 100 corresponds to sigma > 10
>
> **Expected Trajectory:** Return variance should generally decrease over training as the policy converges. However, some variance is expected due to environment stochasticity.
>
> **Relationship to Percentiles:** High variance often correlates with large P90-P10 spread (TELE-224, TELE-226). Both indicate inconsistent policy behavior.
>
> **Implementation Note:** Should be computed from the same episode return buffer as the percentile metrics for consistency.
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
