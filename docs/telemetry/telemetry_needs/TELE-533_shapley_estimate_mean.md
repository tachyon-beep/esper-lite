# Telemetry Record: [TELE-533] Shapley Estimate Mean

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-533` |
| **Name** | Shapley Estimate Mean |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the expected marginal contribution of a specific slot to the ensemble's performance?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | accuracy contribution (percentage points) |
| **Range** | Typically `[-10.0, +10.0]` for accuracy deltas |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> The mean Shapley value for a single slot, representing the expected marginal contribution of that slot to ensemble accuracy across all possible coalition orderings.
>
> **Computation:**
> ```
> mean = (1/n!) * sum over all permutations of [
>     accuracy(coalition ∪ {slot}) - accuracy(coalition)
> ]
> ```
> In practice, estimated via Monte Carlo sampling over ~100-1000 permutations.
>
> **Interpretation:**
> - **mean = +2.5:** This slot adds ~2.5 percentage points to accuracy on average
> - **mean = 0.0:** This slot has no net effect (neither helps nor hurts)
> - **mean = -1.2:** This slot reduces accuracy by ~1.2 percentage points

### Health Thresholds

| Level | Condition | Meaning | UI Style |
|-------|-----------|---------|----------|
| **Positive** | `mean > 0.01` | Slot contributes positively | green |
| **Neutral** | `abs(mean) <= 0.01` | Negligible contribution | dim |
| **Negative** | `mean < -0.01` | Slot hurts performance | red |

**Threshold Source:** `src/esper/karn/sanctum/widgets/shapley_panel.py` - lines 99-104

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Shapley computation via permutation sampling |
| **File** | `NOT YET IMPLEMENTED` |
| **Function/Method** | TBD - computes average marginal contribution |
| **Line(s)** | N/A |

```python
# PLANNED IMPLEMENTATION (not yet wired)
# Compute mean via Monte Carlo permutation sampling:
marginal_contributions = []
for _ in range(n_samples):
    perm = random_permutation(active_slots)
    coalition = []
    for slot in perm:
        before = evaluate_accuracy(coalition)
        coalition.append(slot)
        after = evaluate_accuracy(coalition)
        marginal_contributions[slot].append(after - before)

mean = np.mean(marginal_contributions[slot])
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in ShapleyEstimate within ShapleySnapshot | TBD |
| **2. Collection** | ANALYTICS_SNAPSHOT event | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Extracted from event payload | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `shapley_snapshot.values[slot_id].mean` | `karn/sanctum/schema.py` |

```
[Shapley Computation]
  --for each slot, compute mean-->
  [ShapleyEstimate(mean=X, std=Y, n_samples=N)]
  --pack into ShapleySnapshot-->
  [SanctumAggregator]
  ---->
  [EnvState.shapley_snapshot.values[slot_id].mean]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ShapleyEstimate` |
| **Field** | `mean` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].shapley_snapshot.values[slot_id].mean` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 265 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ShapleyPanel | `widgets/shapley_panel.py` (lines 78-109) | Displays mean value with color coding |
| ShapleySnapshot.get_mean() | `schema.py` (lines 282-286) | Accessor method for mean lookup |
| ShapleySnapshot.get_significance() | `schema.py` (lines 288-293) | Uses mean vs std for significance |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** - NO emitter implemented yet
- [ ] **Transport works** - Aggregator handler TBD
- [x] **Schema field exists** - `ShapleyEstimate.mean: float = 0.0` at line 265
- [x] **Default is correct** - `0.0` indicates no contribution measured
- [x] **Consumer reads it** - ShapleyPanel accesses `estimate.mean` at line 88
- [x] **Display is correct** - Shows mean with +/- sign, color coded
- [x] **Thresholds applied** - 0.01 threshold for positive/negative coloring

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | Emitter not implemented | `[ ]` |
| Unit (ShapleyEstimate) | `tests/karn/sanctum/test_schema.py` | `test_shapley_estimate` | `[ ]` |
| Widget (ShapleyPanel) | `tests/karn/sanctum/widgets/test_shapley_panel.py` | Mean display | `[ ]` |

### Manual Verification Steps

1. Create ShapleyEstimate with known mean value
2. Verify ShapleyPanel displays value correctly formatted
3. Verify color: green for >0.01, red for <-0.01, dim for near-zero
4. Verify significance star when `abs(mean) > 1.96 * std`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Permutation sampling | computation | Monte Carlo estimation of Shapley values |
| Counterfactual evaluation | system | Must evaluate accuracy for coalitions |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ShapleyPanel value display | widget | Primary display of contribution |
| get_significance() | method | Uses mean/std ratio |
| ranked_slots() | method | Sorts by mean |
| Pruning decisions | future | Could auto-prune negative contributors |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** Schema field and consumers exist, but no emitter populates the value. Part of the overall Shapley computation feature (TELE-530).
>
> **Display Format:** ShapleyPanel formats mean as:
> - `+2.345` for positive (with explicit + sign)
> - `-1.234` for negative
> - For values >= 1.0, uses 1 decimal place (`+2.3`)
> - For values < 1.0, uses 3 decimal places (`+0.023`)
>
> **Statistical Interpretation:** The mean alone is not sufficient for decision-making. Always consider with std (TELE-534) to assess significance. A mean of +0.5 with std of 2.0 is statistically indistinguishable from zero.
