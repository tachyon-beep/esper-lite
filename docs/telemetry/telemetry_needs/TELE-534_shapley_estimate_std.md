# Telemetry Record: [TELE-534] Shapley Estimate Std

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-534` |
| **Name** | Shapley Estimate Std |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How certain are we about this slot's Shapley value? Is the measured contribution statistically reliable?"

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
| **Units** | accuracy points (same as mean) |
| **Range** | `[0.0, +inf)` (non-negative) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Standard deviation of the Shapley value estimate from Monte Carlo permutation sampling. Represents uncertainty in the mean estimate due to:
> 1. Finite sample size (not all permutations evaluated)
> 2. Stochastic evaluation noise (validation accuracy variance)
>
> **Interpretation:**
> - **Low std (< 0.1):** High confidence in mean estimate
> - **High std (> mean):** Low confidence; mean may be misleading
> - **std = 0:** Either only 1 sample, or perfectly consistent marginals (rare)
>
> **Statistical significance:**
> ```python
> # 95% confidence interval
> is_significant = abs(mean) > 1.96 * std
> ```
> If significant, we're 95% confident the true contribution is non-zero.

### Health Thresholds

| Level | Condition | Meaning | UI Style |
|-------|-----------|---------|----------|
| **Significant** | `abs(mean) > 1.96 * std` | 95% CI excludes zero | bold + star |
| **Not Significant** | `abs(mean) <= 1.96 * std` | Cannot distinguish from zero | normal + circle |

**Threshold Source:** `src/esper/karn/sanctum/widgets/shapley_panel.py` - line 92, `schema.py` - line 293

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Shapley computation via permutation sampling |
| **File** | `NOT YET IMPLEMENTED` |
| **Function/Method** | TBD - computes std of marginal contributions |
| **Line(s)** | N/A |

```python
# PLANNED IMPLEMENTATION (not yet wired)
# Compute std from Monte Carlo samples:
marginal_contributions = []  # Collected during permutation sampling
std = np.std(marginal_contributions[slot], ddof=1)  # Sample std
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in ShapleyEstimate within ShapleySnapshot | TBD |
| **2. Collection** | ANALYTICS_SNAPSHOT event | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Extracted from event payload | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `shapley_snapshot.values[slot_id].std` | `karn/sanctum/schema.py` |

```
[Shapley Computation]
  --for each slot, compute std-->
  [ShapleyEstimate(mean=X, std=Y, n_samples=N)]
  --pack into ShapleySnapshot-->
  [SanctumAggregator]
  ---->
  [EnvState.shapley_snapshot.values[slot_id].std]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ShapleyEstimate` |
| **Field** | `std` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].shapley_snapshot.values[slot_id].std` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 266 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ShapleyPanel | `widgets/shapley_panel.py` (lines 89, 92-96) | Displays std with +/- prefix, uses for significance |
| ShapleySnapshot.get_significance() | `schema.py` (lines 288-293) | Computes significance using z=1.96 |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** - NO emitter implemented yet
- [ ] **Transport works** - Aggregator handler TBD
- [x] **Schema field exists** - `ShapleyEstimate.std: float = 0.0` at line 266
- [x] **Default is correct** - `0.0` indicates no uncertainty (or no data)
- [x] **Consumer reads it** - ShapleyPanel accesses `estimate.std` at line 89
- [x] **Display is correct** - Shows as "+/-X.XXX" format
- [x] **Thresholds applied** - 1.96*std used for significance determination

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | Emitter not implemented | `[ ]` |
| Unit (ShapleyEstimate) | `tests/karn/sanctum/test_schema.py` | `test_shapley_estimate` | `[ ]` |
| Unit (get_significance) | `tests/karn/sanctum/test_schema.py` | `test_shapley_significance` | `[ ]` |
| Widget (ShapleyPanel) | `tests/karn/sanctum/widgets/test_shapley_panel.py` | Std and significance display | `[ ]` |

### Manual Verification Steps

1. Create ShapleyEstimate with known mean and std values
2. Verify ShapleyPanel displays std as "+/-X.XXX"
3. Test significance: mean=1.0, std=0.4 should be significant (1.0 > 1.96*0.4=0.784)
4. Test non-significance: mean=0.5, std=0.5 should NOT be significant (0.5 < 1.96*0.5=0.98)
5. Verify significance indicator: star for significant, circle for not

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Permutation sampling | computation | Std computed from sample variance |
| n_samples | field | More samples = lower std (typically) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ShapleyPanel uncertainty display | widget | Shows +/- bound |
| get_significance() | method | Uses std for 95% CI calculation |
| Automated pruning | future | Only prune if negative AND significant |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** Schema field and consumers exist, but no emitter populates the value. Part of the overall Shapley computation feature (TELE-530).
>
> **Display Format:** ShapleyPanel formats std as:
> - `+/-0.234` for std < 1.0
> - `+/-2.3` for std >= 1.0 (1 decimal place)
>
> **Statistical Significance Logic:**
> ```python
> # From schema.py ShapleySnapshot.get_significance()
> def get_significance(self, slot_id: str, z: float = 1.96) -> bool:
>     """Check if slot's contribution is statistically significant (95% CI)."""
>     if slot_id not in self.values:
>         return False
>     est = self.values[slot_id]
>     return abs(est.mean) > z * est.std if est.std > 0 else est.mean != 0
> ```
>
> **Edge Cases:**
> - `std = 0` with `mean != 0`: Treated as significant (deterministic result)
> - `std = 0` with `mean = 0`: Treated as not significant (no contribution detected)
>
> **Sample Size Relationship:** More permutation samples typically yield lower std (central limit theorem). The `n_samples` field (ShapleyEstimate) tracks this for debugging.
