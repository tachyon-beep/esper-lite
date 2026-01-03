# Telemetry Record: [TELE-221] TD Error Mean

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-221` |
| **Name** | TD Error Mean |
| **Category** | `value` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "Is the value function systematically over- or under-estimating returns, indicating bias in the critic?"

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
| **Units** | reward units (same scale as returns) |
| **Range** | unbounded (typically -50 to +50) |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before first TD error computation) |

### Semantic Meaning

> TD Error Mean is the average temporal difference error across the batch:
>
> **TD Error = r + gamma * V(s') - V(s)**
>
> Where:
> - r = immediate reward
> - gamma = discount factor
> - V(s') = value estimate of next state
> - V(s) = value estimate of current state
>
> - **TD Mean near 0:** Value function is well-calibrated, predictions match targets
> - **TD Mean > 0:** Value function under-estimates returns (pessimistic)
> - **TD Mean < 0:** Value function over-estimates returns (optimistic)
> - **|TD Mean| > 15:** Severe bias, may indicate target network staleness or reward scale issues
>
> High mean TD error indicates systematic bias in value predictions, which corrupts advantage estimates.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `abs(value) < 5` | Value function well-calibrated |
| **Warning** | `5 <= abs(value) < 15` | Moderate bias in value estimates |
| **Critical** | `abs(value) >= 15` | Severe bias, advantages unreliable |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` - `_get_td_error_style()` method (lines 163-175)

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
# Compute mean TD error from rollout buffer data
# TD error = reward + gamma * next_value - current_value
#
# td_errors = rewards + gamma * next_values * (1 - dones) - values
# td_error_mean = td_errors.mean().item()
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 741) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.td_error_mean` | `karn/sanctum/schema.py` |

```
[PPOAgent.update() - PLANNED]
  --td_error_mean-->
  [emit_ppo_update_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.td_error_mean]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `td_error_mean` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.td_error_mean` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 741 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 82-84) | Displayed as "TD Mean" with sign prefix (+/-) and color-coded style (green/yellow/red based on absolute value) |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes td_error_mean
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.td_error_mean: float = 0.0` at line 741
- [x] **Default is correct** — `0.0` is appropriate before TD error data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.td_error_mean`
- [x] **Display is correct** — Panel renders with sign prefix and `_get_td_error_style()` for coloring
- [x] **Thresholds applied** — Consumer uses abs(mean) < 5/15 thresholds for color coding

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
3. TD Mean should hover near 0 for a healthy value network
4. Verify color transitions: green (|mean| < 5) -> yellow (5-15) -> red (>= 15)
5. Watch for sustained positive (pessimistic) or negative (optimistic) bias

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Value predictions V(s) | computation | Current state value estimates |
| Next state values V(s') | computation | Next state value estimates |
| Rewards | data | Immediate rewards from environment |
| Discount factor gamma | config | Used in TD target calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ValueDiagnosticsPanel display | display | Shows bias direction and magnitude |
| Value calibration diagnosis | research | Identifies systematic over/under-estimation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Interpretation:** TD error mean indicates bias direction:
> - Positive mean: Value function is pessimistic (under-estimates returns)
> - Negative mean: Value function is optimistic (over-estimates returns)
> - Near zero: Well-calibrated
>
> **Relationship to td_error_std (TELE-222):** Mean and std together tell the full story. High mean with low std = consistent bias. Low mean with high std = noisy but unbiased. Both high = learning instability.
>
> **Implementation Note:** Should be computed alongside TELE-222 (td_error_std) from the same TD error tensor for efficiency.
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
