# Telemetry Record: [TELE-222] TD Error Std

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-222` |
| **Name** | TD Error Std |
| **Category** | `value` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "How noisy are the value function's gradient targets? Is the variance in TD errors excessive, leading to unstable learning?"

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
| **Range** | `[0, +inf)` (non-negative) |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before first TD error computation) |

### Semantic Meaning

> TD Error Std is the standard deviation of temporal difference errors across the batch:
>
> **TD Error Std = std(r + gamma * V(s') - V(s))**
>
> - **Low std (< 5):** Consistent value predictions, stable gradient targets
> - **Moderate std (5-15):** Normal variance in early training
> - **High std (> 15):** Noisy gradient targets, value function struggling
>
> High TD error std indicates the value function is receiving inconsistent gradient signals, which can slow learning or cause instability. Normal in early training but should decrease as value network converges.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | (no explicit threshold in widget) | Context-dependent; shown for operator interpretation |
| **Context** | Displayed alongside TD Mean | Interpretation depends on mean value |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 86-87) - displayed as cyan (informational, no color-coded thresholds)

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
# Compute std of TD errors from rollout buffer data
# TD error = reward + gamma * next_value - current_value
#
# td_errors = rewards + gamma * next_values * (1 - dones) - values
# td_error_std = td_errors.std().item()
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 742) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.td_error_std` | `karn/sanctum/schema.py` |

```
[PPOAgent.update() - PLANNED]
  --td_error_std-->
  [emit_ppo_update_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.td_error_std]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `td_error_std` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.td_error_std` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 742 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 86-87) | Displayed as "TD Std" in cyan (informational) alongside TD Mean |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes td_error_std
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.td_error_std: float = 0.0` at line 742
- [x] **Default is correct** — `0.0` is appropriate before TD error data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.td_error_std`
- [x] **Display is correct** — Panel renders in cyan as informational metric
- [x] **Thresholds applied** — No color thresholds; displayed for context with TD Mean

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
3. TD Std should start high and decrease as value network converges
4. Compare with TD Mean: low mean + high std = noisy but unbiased

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
| ValueDiagnosticsPanel display | display | Provides variance context for TD Mean |
| Value convergence diagnosis | research | High std late in training = value network not converging |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Interpretation Pattern:** TD error mean and std together tell the full story:
> - Low mean, low std: Well-calibrated, stable value function
> - Low mean, high std: Unbiased but noisy (normal early training)
> - High mean, low std: Consistent bias (systematic error)
> - High mean, high std: Both biased and noisy (problematic)
>
> **Expected Trajectory:** TD std should decrease over training as the value network converges. If std stays high or increases, the value function is struggling to fit the data.
>
> **Implementation Note:** Should be computed alongside TELE-221 (td_error_mean) from the same TD error tensor for efficiency.
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
