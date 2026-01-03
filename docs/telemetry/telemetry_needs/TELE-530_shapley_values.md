# Telemetry Record: [TELE-530] Shapley Values

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-530` |
| **Name** | Shapley Values |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How much does each seed slot contribute to the ensemble's performance, accounting for all possible coalition orderings?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [x] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `dict[str, ShapleyEstimate]` |
| **Units** | accuracy contribution (percentage points) |
| **Range** | Typically `[-10.0, +10.0]` for accuracy deltas |
| **Precision** | 3 decimal places for display |
| **Default** | `{}` (empty dict before first computation) |

### Semantic Meaning

> Shapley values are the gold standard for fair attribution in cooperative games. For each slot, the Shapley value represents the average marginal contribution across all possible orderings of coalition formation.
>
> **ShapleyEstimate structure:**
> - `mean`: Expected marginal contribution (percentage points of accuracy)
> - `std`: Standard deviation (uncertainty from permutation sampling)
> - `n_samples`: Number of permutation samples used
>
> **Interpretation:**
> - **Positive value (e.g., +2.5):** Slot contributes +2.5 accuracy points on average
> - **Near-zero value:** Slot has negligible impact (candidate for pruning)
> - **Negative value:** Slot hurts performance (should be pruned)
>
> Shapley values sum to the total improvement over baseline (all seeds disabled).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Positive** | `mean > 0.01` | Slot contributes positively (green in UI) |
| **Neutral** | `abs(mean) <= 0.01` | Negligible contribution (dim in UI) |
| **Negative** | `mean < -0.01` | Slot hurts performance (red in UI) |
| **Significant** | `abs(mean) > 1.96 * std` | 95% confidence contribution is real (bold in UI) |

**Threshold Source:** `src/esper/karn/sanctum/widgets/shapley_panel.py` - lines 99-104, 92

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Shapley computation at episode boundaries |
| **File** | `NOT YET IMPLEMENTED` |
| **Function/Method** | TBD - likely `compute_shapley_values()` in counterfactual engine |
| **Line(s)** | N/A |

```python
# PLANNED IMPLEMENTATION (not yet wired)
# At episode end, compute Shapley values via permutation sampling:
# 1. For each permutation of active slots
# 2. Add slots one-by-one, measure accuracy delta
# 3. Average marginal contributions = Shapley value
# 4. Emit via ANALYTICS_SNAPSHOT event kind="shapley_computed"
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | ANALYTICS_SNAPSHOT event with kind="shapley_computed" | TBD |
| **2. Collection** | Event bus subscription | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Aggregator builds ShapleySnapshot from event payload | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `env_state.shapley_snapshot.values` | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --shapley_computed-->
  [emit_analytics_snapshot()]
  --AnalyticsPayload-->
  [SanctumAggregator.handle_analytics()]
  --build ShapleySnapshot-->
  [EnvState.shapley_snapshot.values]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ShapleySnapshot` |
| **Field** | `values` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].shapley_snapshot.values` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 278 |
| **Default Value** | `{}` (empty dict via `field(default_factory=dict)`) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ShapleyPanel | `widgets/shapley_panel.py` (lines 70-112) | Renders per-slot Shapley values with color coding and significance stars |
| BestRunRecord | `schema.py` (line 1257) | Snapshots Shapley attribution at peak accuracy for historical detail modal |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** - NO emitter implemented yet
- [ ] **Transport works** - Aggregator handler TBD
- [x] **Schema field exists** - `ShapleySnapshot.values: dict[str, ShapleyEstimate]` at line 278
- [x] **Default is correct** - `{}` is appropriate default before first computation
- [x] **Consumer reads it** - ShapleyPanel directly accesses `snapshot.values`
- [x] **Display is correct** - ShapleyPanel renders with color coding per threshold
- [x] **Thresholds applied** - Uses 0.01 and 1.96*std thresholds from shapley_panel.py

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | Emitter not implemented | `[ ]` |
| Unit (aggregator) | N/A | Aggregator handler not implemented | `[ ]` |
| Integration (end-to-end) | N/A | Full pipeline not wired | `[ ]` |
| Widget (ShapleyPanel) | `tests/karn/sanctum/widgets/test_shapley_panel.py` | Panel rendering | `[ ]` |

### Manual Verification Steps

1. Start training with multiple active seeds
2. Wait for episode boundary (Shapley computation triggers)
3. Observe ShapleyPanel - should show per-slot values with significance indicators
4. Verify color coding: green (positive), red (negative), bold (significant)
5. After training, query telemetry for Shapley snapshots

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Episode boundaries | event | Shapley computed at episode end only |
| Counterfactual evaluation | system | Requires ability to evaluate accuracy with different seed configurations |
| Multiple active seeds | condition | Need 2+ seeds for meaningful Shapley computation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ShapleyPanel display | widget | Primary consumer for real-time visualization |
| BestRunRecord.shapley_snapshot | schema | Captured at peak accuracy for historical view |
| Pruning decisions | future | Could use negative Shapley for auto-prune |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** Schema and consumer (ShapleyPanel) exist, but NO EMITTER is implemented. This is a planned feature that requires:
> 1. Counterfactual engine to support permutation sampling
> 2. Episode boundary hook to trigger computation
> 3. Event emission via ANALYTICS_SNAPSHOT
> 4. Aggregator handler to populate EnvState.shapley_snapshot
>
> **Computational Cost:** Shapley computation is O(n!) in theory, but practical implementations use Monte Carlo sampling with ~100-1000 permutations. This should be computed asynchronously to avoid blocking training.
>
> **Relationship to CounterfactualSnapshot:** CounterfactualSnapshot provides raw 2^n accuracy matrix. ShapleySnapshot provides summarized per-slot attributions derived from similar methodology. They are complementary views.
