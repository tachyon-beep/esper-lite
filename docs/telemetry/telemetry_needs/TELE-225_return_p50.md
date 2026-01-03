# Telemetry Record: [TELE-225] Return P50

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-225` |
| **Name** | Return P50 |
| **Category** | `value` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What is the median episode performance? Is the typical episode successful, independent of outliers?"

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
| **Units** | episode return (sum of rewards) |
| **Range** | unbounded (depends on reward scale) |
| **Precision** | 0 decimal places for display (integer-like) |
| **Default** | `0.0` (before return data available) |

### Semantic Meaning

> Return P50 is the 50th percentile (median) of episode returns, representing typical performance:
>
> **P50 = value where 50% of episodes have lower returns**
>
> The median is more robust than mean because:
> - Not skewed by outlier episodes (very good or very bad)
> - Better represents "typical" episode performance
> - Stable even with non-normal return distributions
>
> Per DRL expert: Median return is often a better primary metric than mean return for RL training because episode returns frequently have heavy tails.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Displayed in green; typical episode is positive |
| **Concern** | `value < 0` | Displayed in red; typical episode has negative return |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 93, 98-99) - color based on sign

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
# episode_returns = [...]  # Collected from episode completions
# return_p50 = np.percentile(episode_returns, 50)  # = median
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 751) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.return_p50` | `karn/sanctum/schema.py` |

```
[Episode completion - PLANNED]
  --return_p50-->
  [emit_episode_stats_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.return_p50]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `return_p50` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.return_p50` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 751 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 98-99) | Displayed as "p50:X" in red if negative, green if positive |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes return_p50
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.return_p50: float = 0.0` at line 751
- [x] **Default is correct** — `0.0` is appropriate before return data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.return_p50`
- [x] **Display is correct** — Panel renders with sign prefix and appropriate color
- [x] **Thresholds applied** — Consumer uses sign for color

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
3. P50 should track learning progress as typical episode improves
4. Compare to mean return to detect skewness

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
| ValueDiagnosticsPanel display | display | Shows typical performance |
| Bimodal detection | analysis | Combined with P10, P90 to detect distribution shape |
| Mean vs median comparison | analysis | Large difference indicates skewness |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Percentile Triad:** P10, P50, P90 should be viewed together (see TELE-224, TELE-226). They reveal distribution shape that mean return would obscure.
>
> **Median vs Mean:** When P50 differs significantly from mean return:
> - P50 > mean: Left-skewed distribution (some catastrophic failures)
> - P50 < mean: Right-skewed distribution (some exceptional successes)
>
> **Robustness:** Median is preferred over mean for RL training metrics because:
> 1. Episode returns often have heavy tails
> 2. Single outlier episodes don't skew the metric
> 3. Better reflects "typical" agent behavior
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
