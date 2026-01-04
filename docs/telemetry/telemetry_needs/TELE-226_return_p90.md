# Telemetry Record: [TELE-226] Return P90

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-226` |
| **Name** | Return P90 |
| **Category** | `value` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What is the best-case performance? What returns is the policy capable of achieving in its best episodes?"

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

> Return P90 is the 90th percentile of episode returns, representing the best 10% of episodes:
>
> **P90 = value where 90% of episodes have lower returns**
>
> - **P90 >> P50:** Policy occasionally achieves excellent results but inconsistent
> - **P90 near P50:** Consistent policy, performance doesn't vary much
> - **Large (P90 - P10) spread:** Inconsistent policy, warrants investigation
>
> P90 shows the policy's "capability ceiling" - what it can achieve when conditions align.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Displayed in green; best episodes are positive |
| **Concern** | `value < 0` | Displayed in red; even best episodes have negative returns |
| **Bimodal Warning** | `P90 - P10 > 50` | Yellow warning icon shown; policy is inconsistent |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 94, 100-106) - color based on sign, spread warning at > 50

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
# return_p90 = np.percentile(episode_returns, 90)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 752) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.return_p90` | `karn/sanctum/schema.py` |

```
[Episode completion - PLANNED]
  --return_p90-->
  [emit_episode_stats_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.return_p90]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `return_p90` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.return_p90` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 752 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 100-106) | Displayed as "p90:X" in red if negative, green if positive, with spread warning if P90-P10 > 50 |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes return_p90
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.return_p90: float = 0.0` at line 752
- [x] **Default is correct** — `0.0` is appropriate before return data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.return_p90`
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
3. P90 shows policy's best performance capability
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
| ValueDiagnosticsPanel display | display | Shows best-case performance |
| Bimodal detection | analysis | Combined with P10, P50 to detect distribution shape |
| Spread warning | display | P90 - P10 > 50 triggers warning |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Percentile Triad:** P10, P50, P90 should be viewed together (see TELE-224, TELE-225). They reveal distribution shape that mean return would obscure.
>
> **Capability Assessment:** P90 shows what the policy is *capable of* achieving, not what it typically achieves. If P90 is high but P50 is low, the policy knows how to succeed but doesn't do it consistently.
>
> **Spread Warning:** The ValueDiagnosticsPanel shows a yellow warning icon when (P90 - P10) > 50. This indicates:
> - Highly variable performance
> - Possible bimodal distribution
> - Policy might be unstable or have multiple local optima
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
