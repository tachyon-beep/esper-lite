# Telemetry Record: [TELE-224] Return P10

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-224` |
| **Name** | Return P10 |
| **Category** | `value` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What is the worst-case performance? Are there episodes with severely negative returns indicating catastrophic failures?"

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

> Return P10 is the 10th percentile of episode returns, representing the worst 10% of episodes:
>
> **P10 = value where 10% of episodes have lower returns**
>
> - **P10 near P50:** Consistent policy, few catastrophic failures
> - **P10 << P50:** Bimodal distribution, some episodes fail completely
> - **Large (P90 - P10) spread:** Inconsistent policy, warrants investigation
>
> Per DRL expert: Return percentiles catch bimodal policies that mean return would miss. A policy might look healthy on average but have a long tail of failures.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Displayed in green; worst episodes still positive |
| **Concern** | `value < 0` | Displayed in red; some episodes have negative returns |
| **Bimodal Warning** | `P90 - P10 > 50` | Yellow warning icon shown; policy is inconsistent |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 92-106) - color based on sign, spread warning at > 50

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
# return_p10 = np.percentile(episode_returns, 10)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 750) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.return_p10` | `karn/sanctum/schema.py` |

```
[Episode completion - PLANNED]
  --return_p10-->
  [emit_episode_stats_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.return_p10]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `return_p10` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.return_p10` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 750 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 91-107) | Displayed as "p10:X" in red if negative, green if positive, with spread warning if P90-P10 > 50 |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes return_p10
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.return_p10: float = 0.0` at line 750
- [x] **Default is correct** — `0.0` is appropriate before return data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.return_p10`
- [x] **Display is correct** — Panel renders with sign prefix and appropriate color
- [x] **Thresholds applied** — Consumer uses sign for color, spread > 50 for warning

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
3. Compare P10 to P50 and P90 to assess distribution shape
4. Large P90-P10 spread triggers yellow warning icon

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
| ValueDiagnosticsPanel display | display | Shows worst-case performance |
| Bimodal detection | analysis | Combined with P50, P90 to detect distribution shape |
| TELE-227 return_variance | related | P10/P90 spread correlates with variance |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Percentile Triad:** P10, P50, P90 should be viewed together (see TELE-225, TELE-226). They reveal distribution shape that mean return would obscure.
>
> **Bimodal Detection:** When P10 is very negative but P90 is very positive, the policy is bimodal - sometimes succeeding spectacularly, sometimes failing completely. This often indicates the policy found two local optima and is not converging.
>
> **Implementation Note:** Should be computed from a rolling buffer of episode returns (e.g., last 100 episodes) to provide stable percentile estimates.
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
