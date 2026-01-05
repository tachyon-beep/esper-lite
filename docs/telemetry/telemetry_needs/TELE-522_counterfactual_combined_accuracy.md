# Telemetry Record: [TELE-522] Counterfactual Combined Accuracy

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-522` |
| **Name** | Counterfactual Combined Accuracy |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the model's accuracy with ALL seeds enabled at their current alpha values?"

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
| **Units** | percentage (0.0 to 100.0) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before first computation) |

### Semantic Meaning

> Combined accuracy represents the model's validation accuracy with all active seeds enabled at full alpha.
> This is the "treatment" measurement showing the ensemble's collective performance.
>
> **Key relationships:**
> - `total_improvement = combined_accuracy - baseline_accuracy`
> - `synergy = combined - baseline - sum(individual_contributions)`
>
> If combined accuracy is significantly higher than the sum of individual contributions + baseline,
> this indicates positive synergy (seeds are working together).
> If combined is LOWER than expected, this indicates interference (seeds are hurting each other).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `combined > baseline` | Seeds are improving model performance |
| **Warning** | `combined == baseline` | Seeds have no net effect |
| **Critical** | `combined < baseline` | Seeds are hurting model (interference) |

**Threshold Source:** Implicit in synergy calculation

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Counterfactual engine (not yet implemented) |
| **File** | TBD |
| **Function/Method** | TBD |
| **Line(s)** | TBD |

```python
# PLANNED: Counterfactual engine computes combined
# by evaluating host model with all seeds at alpha=1
combined_acc = evaluate_model(host, validation_batch, seed_alphas=all_ones)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TBD - Counterfactual engine event | TBD |
| **2. Collection** | TBD - CounterfactualConfig with all-true mask | TBD |
| **3. Aggregation** | TBD - Aggregator handler | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Derived via `combined_accuracy` property | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --CounterfactualConfig(seed_mask=(True,...), accuracy=X)-->
  [SanctumAggregator]
  --env_state.counterfactual_matrix.configs-->
  [CounterfactualSnapshot.combined_accuracy (property)]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `CounterfactualSnapshot` |
| **Field** | `combined_accuracy` (property) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].counterfactual_matrix.combined_accuracy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 334-341 |
| **Default Value** | `0.0` (when no matching config found) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 95) | Used for bar scaling (max value) |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 159) | "All seeds" bar in Combined section |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (lines 163-165) | Actual improvement calculation |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Not yet implemented (counterfactual engine TBD)
- [ ] **Transport works** — No event emission path exists
- [x] **Schema field exists** — `CounterfactualSnapshot.combined_accuracy` property at lines 334-341
- [x] **Default is correct** — Returns `0.0` when no all-true config found
- [x] **Consumer reads it** — CounterfactualPanel accesses `self._matrix.combined_accuracy`
- [x] **Display is correct** — Rendered as "All seeds" bar in Combined section
- [ ] **Thresholds applied** — No explicit health thresholds in widget

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | TBD | TBD | `[ ]` |
| Unit (aggregator) | TBD | TBD | `[ ]` |
| Unit (schema property) | `tests/karn/sanctum/test_schema.py` | `test_counterfactual_combined_accuracy` | `[ ]` |
| Widget (CounterfactualPanel) | `tests/karn/sanctum/widgets/test_counterfactual_panel.py` | Combined rendering | `[ ]` |

### Manual Verification Steps

1. Start training with multiple seeds
2. Launch Sanctum TUI
3. After first episode completes, observe CounterfactualPanel
4. Verify "All seeds" bar shows combined accuracy
5. Verify improvement value is `combined - baseline`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Validation batch | data | Requires validation data for accuracy evaluation |
| All active seeds | state | Seeds must be germinated and integrated |
| All-seeds-enabled config | computation | Requires CounterfactualConfig with all-true mask |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Total improvement | computation | `improvement = combined - baseline` |
| Total synergy | computation | `synergy = combined - baseline - sum(individuals)` |
| Waterfall visualization | display | Combined is the rightmost/final bar |
| Interference detection | display | `combined < expected` triggers interference warning |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** This property exists in the schema but relies on configs list which has NO emitter implemented.
>
> **Implementation Note:** The `combined_accuracy` is a computed property that searches configs for the all-true seed_mask. This represents the actual ensemble performance with all seeds contributing.
>
> **Interference Detection:** When `combined_accuracy < baseline + sum(individual_contributions)`, seeds are interfering with each other. The CounterfactualPanel displays this prominently with a red "INTERFERENCE DETECTED" banner (see lines 175-179).
