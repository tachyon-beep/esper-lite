# Telemetry Record: [TELE-223] Bellman Error

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-223` |
| **Name** | Bellman Error |
| **Category** | `value` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the value function's prediction error spiking, which often precedes NaN losses or value collapse?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

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
| **Units** | squared reward units |
| **Range** | `[0, +inf)` (non-negative) |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before first Bellman error computation) |

### Semantic Meaning

> Bellman Error is the squared temporal difference error, measuring how far the value function is from satisfying the Bellman equation:
>
> **Bellman Error = mean(|V(s) - (r + gamma * V(s'))|^2)**
>
> This is the value function's training objective. Spikes in Bellman error:
> - Often PRECEDE NaN losses (early warning signal)
> - Indicate value function instability
> - Can signal reward scale issues or divergence
>
> Unlike TD error (signed), Bellman error is always positive and squared, amplifying large errors.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value < 20` | Value function stable, predictions near targets |
| **Warning** | `20 <= value < 50` | Elevated error, monitor for trend |
| **Critical** | `value >= 50` | Severe error, NaN likely imminent |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` - `_get_bellman_style()` method (lines 177-187)

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
# Compute squared Bellman error from value predictions and TD targets
#
# td_targets = rewards + gamma * next_values * (1 - dones)
# bellman_error = ((values - td_targets) ** 2).mean().item()
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 746) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.bellman_error` | `karn/sanctum/schema.py` |

```
[PPOAgent.update() - PLANNED]
  --bellman_error-->
  [emit_ppo_update_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.bellman_error]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `bellman_error` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.bellman_error` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 746 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 76-78) | Displayed as "Bellman" with color-coded style (green < 20, yellow 20-50, red >= 50) |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes bellman_error
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.bellman_error: float = 0.0` at line 746
- [x] **Default is correct** — `0.0` is appropriate before Bellman error data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.bellman_error`
- [x] **Display is correct** — Panel renders with `_get_bellman_style()` for color coding
- [x] **Thresholds applied** — Consumer uses 20/50 thresholds for color coding

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
3. Bellman should start moderate and decrease as value network learns
4. Watch for sudden spikes - often precede NaN losses by 10-20 batches
5. Verify color transitions: green (< 20) -> yellow (20-50) -> red (>= 50)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Value predictions V(s) | computation | Current state value estimates |
| TD targets | computation | r + gamma * V(s') or GAE targets |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ValueDiagnosticsPanel display | display | Early warning for value collapse |
| NaN prediction system | system | Spikes predict NaN losses |
| Training stability alerts | system | Could trigger checkpoint save before crash |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Critical Diagnostic:** Bellman error spikes often PRECEDE NaN losses. A spike from 20 to 100+ typically indicates value function is about to diverge. This makes it one of the most important early warning signals.
>
> **Relationship to Value Loss:** Bellman error is closely related to value_loss (which is also MSE between predictions and targets). The difference is that value_loss uses GAE-bootstrapped targets while Bellman error uses true TD(0) targets. Bellman error is more sensitive to instability.
>
> **Action on Critical:** When Bellman error exceeds 50:
> 1. Save checkpoint immediately
> 2. Consider reducing learning rate
> 3. Check for reward scale issues
> 4. Verify value network isn't receiving corrupted data
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
