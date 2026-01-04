# Telemetry Record: [TELE-524] Counterfactual Individual Contributions

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-524` |
| **Name** | Counterfactual Individual Contributions |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How much does each individual seed contribute to accuracy improvement when acting alone?"

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
| **Type** | `dict[str, float]` |
| **Units** | percentage points (improvement over baseline) |
| **Range** | Typically -5.0 to +10.0 per seed |
| **Precision** | 1 decimal place for display |
| **Default** | `{}` (empty dict before computation) |

### Semantic Meaning

> Individual contributions measure each seed's solo accuracy improvement over the baseline.
>
> **Formula per seed:**
> ```
> contribution[slot_id] = accuracy_with_only_this_seed - baseline_accuracy
> ```
>
> **Interpretation:**
> - **Positive value:** Seed improves model when acting alone
> - **Zero:** Seed has no effect (possibly dormant or not yet learned)
> - **Negative:** Seed hurts model when acting alone (rare, indicates training issue)
>
> This is the "marginal contribution" of each seed, useful for:
> - Identifying high-value seeds (candidates for fossilization)
> - Identifying low-value seeds (candidates for pruning)
> - Computing synergy (see TELE-523)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `contribution > 1.0` | Seed provides meaningful improvement |
| **Neutral** | `0.0 <= contribution <= 1.0` | Seed provides marginal improvement |
| **Warning** | `contribution < 0.0` | Seed hurts model (should investigate or prune) |

**Threshold Source:** Implicit in bar visualization (positive delta = green, negative = red)

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
# PLANNED: Individual contributions computed from factorial results
for slot_id in active_slots:
    solo_acc = evaluate_model(host, val_batch, only_seed=slot_id)
    contributions[slot_id] = solo_acc - baseline
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TBD - Counterfactual engine event | TBD |
| **2. Collection** | TBD - CounterfactualConfigs with single-seed masks | TBD |
| **3. Aggregation** | TBD - Aggregator handler | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Derived via `individual_contributions()` method | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --CounterfactualConfigs (solo seed masks)-->
  [SanctumAggregator]
  --env_state.counterfactual_matrix-->
  [CounterfactualSnapshot.individual_contributions() (method)]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `CounterfactualSnapshot` |
| **Field** | `individual_contributions()` (method) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].counterfactual_matrix.individual_contributions()` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 350-360 |
| **Default Value** | `{}` (when no solo configs available) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 96) | Computed for individual bars |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (lines 113-118) | Rendered as "Individual" section bars |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 163) | Sum used for expected improvement calculation |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Not yet implemented (counterfactual engine TBD)
- [ ] **Transport works** — No event emission path exists
- [x] **Schema field exists** — `CounterfactualSnapshot.individual_contributions()` method at lines 350-360
- [x] **Default is correct** — Returns `{}` when no single-seed configs exist
- [x] **Consumer reads it** — CounterfactualPanel calls `self._matrix.individual_contributions()`
- [x] **Display is correct** — Each seed shown as bar with delta annotation
- [x] **Thresholds applied** — Delta coloring (green if positive, red if negative)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | TBD | TBD | `[ ]` |
| Unit (aggregator) | TBD | TBD | `[ ]` |
| Unit (schema method) | `tests/karn/sanctum/test_schema.py` | `test_counterfactual_individual_contributions` | `[ ]` |
| Widget (CounterfactualPanel) | `tests/karn/sanctum/widgets/test_counterfactual_panel.py` | Individual bars rendering | `[ ]` |

### Manual Verification Steps

1. Start training with multiple seeds
2. Launch Sanctum TUI
3. After first episode completes, observe CounterfactualPanel "Individual" section
4. Verify each active slot has a bar showing its solo contribution
5. Verify bars are ordered by slot_id
6. Verify delta annotations show improvement over baseline

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| baseline_accuracy (TELE-521) | computation | Contributions are relative to baseline |
| Single-seed CounterfactualConfigs | data | Each seed needs solo evaluation |
| slot_ids | metadata | List of active slot IDs |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Total synergy (TELE-523) | computation | `synergy = combined - baseline - sum(individuals)` |
| Pair synergy | computation | `pair_synergy = pair_contrib - ind1 - ind2` |
| Individual bars | display | Waterfall visualization |
| Expected improvement | display | `expected = sum(individual_contributions)` |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** This method exists in the schema but relies on configs list which has NO emitter implemented.
>
> **Implementation Detail:** The method iterates through `slot_ids` and constructs a single-seed mask for each, then looks up the corresponding config's accuracy. If no matching config is found for a slot, that slot is omitted from the result dict.
>
> **Mask Construction:**
> ```python
> # For slot at index i in n-slot system:
> mask = tuple(j == i for j in range(n))
> # e.g., for slot 1 in 3-slot system: (False, True, False)
> ```
>
> **Use Cases:**
> - Identifying which seeds to prioritize for fossilization (highest contribution)
> - Identifying which seeds to consider pruning (zero or negative contribution)
> - Debugging ensemble behavior (why is combined less than expected?)
