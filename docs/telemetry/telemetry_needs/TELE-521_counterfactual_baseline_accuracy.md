# Telemetry Record: [TELE-521] Counterfactual Baseline Accuracy

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-521` |
| **Name** | Counterfactual Baseline Accuracy |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the host model's accuracy with ALL seeds disabled? (the counterfactual baseline)"

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

> Baseline accuracy represents the host model's validation accuracy with all seeds disabled (alpha=0).
> This is the "control" measurement against which individual and combined seed contributions are measured.
>
> **Formula context:**
> - `individual_contribution[seed] = accuracy_with_seed_only - baseline`
> - `total_improvement = combined_accuracy - baseline`
> - `synergy = combined - baseline - sum(individual_contributions)`
>
> A higher baseline indicates the host model is already performing well without seeds.
> A low baseline with high combined accuracy indicates seeds are providing significant value.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value > 0.0` | Baseline computed, attribution analysis available |
| **Warning** | `value == 0.0` | No baseline available (pre-computation) |
| **Critical** | N/A | Not applicable for this field |

**Threshold Source:** Implicit — `value == 0.0` indicates no data

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
# PLANNED: Counterfactual engine computes baseline
# by evaluating host model with all seeds at alpha=0
baseline_acc = evaluate_model(host, validation_batch, seed_alphas=all_zeros)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TBD - Counterfactual engine event | TBD |
| **2. Collection** | TBD - CounterfactualConfig with all-false mask | TBD |
| **3. Aggregation** | TBD - Aggregator handler | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Derived via `baseline_accuracy` property | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --CounterfactualConfig(seed_mask=(False,...), accuracy=X)-->
  [SanctumAggregator]
  --env_state.counterfactual_matrix.configs-->
  [CounterfactualSnapshot.baseline_accuracy (property)]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `CounterfactualSnapshot` |
| **Field** | `baseline_accuracy` (property) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].counterfactual_matrix.baseline_accuracy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 327-332 |
| **Default Value** | `0.0` (when no matching config found) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 94) | Used to render baseline bar in waterfall visualization |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 109) | Starting point for individual contribution bars |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (lines 163-165) | Reference for synergy calculation display |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Not yet implemented (counterfactual engine TBD)
- [ ] **Transport works** — No event emission path exists
- [x] **Schema field exists** — `CounterfactualSnapshot.baseline_accuracy` property at lines 327-332
- [x] **Default is correct** — Returns `0.0` when no all-false config found
- [x] **Consumer reads it** — CounterfactualPanel accesses `self._matrix.baseline_accuracy`
- [x] **Display is correct** — Rendered as first bar in waterfall ("Baseline (Host only)")
- [ ] **Thresholds applied** — No health thresholds in widget (informational display)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | TBD | TBD | `[ ]` |
| Unit (aggregator) | TBD | TBD | `[ ]` |
| Unit (schema property) | `tests/karn/sanctum/test_schema.py` | `test_counterfactual_baseline_accuracy` | `[ ]` |
| Widget (CounterfactualPanel) | `tests/karn/sanctum/widgets/test_counterfactual_panel.py` | Baseline rendering | `[ ]` |

### Manual Verification Steps

1. Start training with multiple seeds
2. Launch Sanctum TUI
3. After first episode completes, observe CounterfactualPanel
4. Verify baseline bar shows host accuracy with all seeds disabled
5. Verify individual bars show improvement over baseline

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Validation batch | data | Requires validation data for accuracy evaluation |
| Host model checkpoint | state | Baseline evaluated on current host weights |
| All-seeds-disabled config | computation | Requires CounterfactualConfig with all-false mask |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Individual contributions | computation | `individual_contribution = seed_acc - baseline` |
| Pair contributions | computation | `pair_contribution = pair_acc - baseline` |
| Total synergy | computation | `synergy = combined - baseline - sum(individuals)` |
| Waterfall visualization | display | Baseline is the leftmost reference point |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** This property exists in the schema but relies on configs list which has NO emitter implemented.
>
> **Implementation Note:** The `baseline_accuracy` is a computed property that searches configs for the all-false seed_mask. This design avoids data duplication but requires at least one CounterfactualConfig with mask=(False, False, ...) to return a meaningful value.
>
> **Evaluation Cost:** Computing baseline requires one forward pass on validation data with all seeds disabled. This is included in the 2^n evaluations for full factorial.
