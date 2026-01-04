# Telemetry Record: [TELE-220] V-Return Correlation

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-220` |
| **Name** | V-Return Correlation |
| **Category** | `value` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the value function actually learning to predict returns, or are advantage estimates effectively random noise?"

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
| **Units** | Pearson correlation coefficient |
| **Range** | `[-1.0, 1.0]` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` (before sufficient data for correlation) |

### Semantic Meaning

> V-Return Correlation measures how well the value function V(s) predicts actual episode returns:
>
> **Pearson Correlation = cov(V(s), G) / (std(V(s)) * std(G))**
>
> Where G = actual return from state s.
>
> - **Corr >= 0.8:** Excellent - value network is well-calibrated, advantages are meaningful
> - **Corr >= 0.5:** Good - value network is learning, advantages are useful
> - **Corr >= 0.3:** Warning - value network is weak, advantages are noisy
> - **Corr < 0.3:** Critical - value network provides no predictive power, PPO is essentially REINFORCE
>
> Per DRL expert review: **This is THE primary diagnostic for RL training failures.** Low V-Return correlation means advantage estimates are garbage, regardless of how healthy policy metrics look. When correlation is low, policy gradient updates are based on noise.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Excellent** | `value >= 0.8` | Value network well-calibrated; advantage estimates reliable |
| **Good** | `0.5 <= value < 0.8` | Value network learning; advantages useful but noisy |
| **Warning** | `0.3 <= value < 0.5` | Value network weak; advantages mostly noise |
| **Critical** | `value < 0.3` | Value network not learning; PPO degrades to REINFORCE |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` - `_get_correlation_style()` method (lines 146-161)

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
# Compute Pearson correlation between value predictions and actual returns
# Should be computed from value_predictions and actual_returns deques
# in ValueFunctionMetrics after sufficient samples accumulated (>= 20)
#
# correlation = compute_correlation(
#     vf_metrics.value_predictions,
#     vf_metrics.actual_returns
# )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 735) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.v_return_correlation` | `karn/sanctum/schema.py` |

```
[PPOAgent.update() - PLANNED]
  --v_return_correlation-->
  [emit_ppo_update_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.v_return_correlation]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `v_return_correlation` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.v_return_correlation` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 735 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 72-74) | Displayed as "V-Corr" with color-coded style and trend icon (>=0.8 green bold with up-arrow, <0.3 red bold with down-arrow) |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes v_return_correlation
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.v_return_correlation: float = 0.0` at line 735
- [x] **Default is correct** — `0.0` is appropriate before correlation data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.v_return_correlation`
- [x] **Display is correct** — Panel renders with `_get_correlation_style()` providing appropriate styling
- [x] **Thresholds applied** — Consumer uses 0.8/0.5/0.3 thresholds for color coding

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
3. V-Corr should start near 0 and rise as value network learns
4. Verify color transitions: red (<0.3) -> yellow (0.3-0.5) -> green (0.5-0.8) -> green bold (>=0.8)
5. After training, query telemetry: `SELECT v_return_correlation FROM ppo_updates ORDER BY batch DESC LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Value predictions | computation | Requires V(s) from value network forward pass |
| Actual returns | computation | Requires complete episode returns (GAE/TD target) |
| Sufficient sample size | data | Need >= 20 samples for meaningful correlation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ValueDiagnosticsPanel display | display | Primary diagnostic for value network health |
| Value health alerts | system | Could trigger auto-intervention if correlation stays critical |
| Training failure diagnosis | research | Root cause indicator when policy looks healthy but training fails |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Critical Priority:** Per DRL expert review, v_return_correlation is THE most important diagnostic for RL training failures. Low correlation means the value network is not learning, and advantage estimates are noise. This metric should be prioritized for implementation.
>
> **Implementation Guidance:** The schema already includes `value_predictions` and `actual_returns` deques in ValueFunctionMetrics (lines 760-761). The aggregator should populate these from PPO update data, then compute correlation using `compute_correlation()` from schema.py (lines 47-84).
>
> **Display:** The ValueDiagnosticsPanel is already implemented and will render this metric correctly once the emitter is wired. The `_get_correlation_style()` method returns appropriate colors and trend icons.
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
