# Telemetry Record: [TELE-525] Counterfactual Pair Contributions

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-525` |
| **Name** | Counterfactual Pair Contributions |
| **Category** | `attribution` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How much does each PAIR of seeds contribute together, and do they synergize or interfere?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [x] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `dict[tuple[str, str], float]` |
| **Units** | percentage points (improvement over baseline) |
| **Range** | Typically -5.0 to +15.0 per pair |
| **Precision** | 1 decimal place for display |
| **Default** | `{}` (empty dict before computation) |

### Semantic Meaning

> Pair contributions measure the combined accuracy improvement of two seeds acting together.
>
> **Formula per pair:**
> ```
> pair_contribution[(slot_a, slot_b)] = accuracy_with_both_seeds - baseline_accuracy
> ```
>
> **Pair synergy:**
> ```
> pair_synergy = pair_contribution - individual_a - individual_b
> ```
>
> **Interpretation:**
> - **pair_synergy > 0:** These seeds work better together (complementary)
> - **pair_synergy = 0:** These seeds are independent (no interaction)
> - **pair_synergy < 0:** These seeds interfere (one or both hurt the other)
>
> This is crucial for understanding ensemble dynamics:
> - Which seed pairs should be fossilized together?
> - Which pairs are fighting and should not coexist?

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `pair_synergy > 0.5` | Strong synergy (green highlight) |
| **Neutral** | `-0.5 <= pair_synergy <= 0.5` | Independent seeds |
| **Warning** | `pair_synergy < -0.5` | Pair interference |

**Threshold Source:** `src/esper/karn/sanctum/widgets/counterfactual_panel.py` — line 132 (`pair_synergy > 0.5` = green)

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
# PLANNED: Pair contributions computed from factorial results
for i, slot_a in enumerate(active_slots):
    for slot_b in active_slots[i+1:]:
        pair_acc = evaluate_model(host, val_batch, seeds=[slot_a, slot_b])
        pairs[(slot_a, slot_b)] = pair_acc - baseline
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TBD - Counterfactual engine event | TBD |
| **2. Collection** | TBD - CounterfactualConfigs with two-seed masks | TBD |
| **3. Aggregation** | TBD - Aggregator handler | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Derived via `pair_contributions()` method | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --CounterfactualConfigs (pair masks)-->
  [SanctumAggregator]
  --env_state.counterfactual_matrix-->
  [CounterfactualSnapshot.pair_contributions() (method)]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `CounterfactualSnapshot` |
| **Field** | `pair_contributions()` (method) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].counterfactual_matrix.pair_contributions()` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 362-374 |
| **Default Value** | `{}` (when no pair configs available) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 97) | Computed for pair bars |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (lines 120-153) | Rendered in "Pairs" or "Top Combinations" section |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Not yet implemented (counterfactual engine TBD)
- [ ] **Transport works** — No event emission path exists
- [x] **Schema field exists** — `CounterfactualSnapshot.pair_contributions()` method at lines 362-374
- [x] **Default is correct** — Returns `{}` when no pair configs exist
- [x] **Consumer reads it** — CounterfactualPanel calls `self._matrix.pair_contributions()`
- [x] **Display is correct** — Pairs shown as bars with synergy highlighting
- [x] **Thresholds applied** — `pair_synergy > 0.5` = green highlight

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | TBD | TBD | `[ ]` |
| Unit (aggregator) | TBD | TBD | `[ ]` |
| Unit (schema method) | `tests/karn/sanctum/test_schema.py` | `test_counterfactual_pair_contributions` | `[ ]` |
| Widget (CounterfactualPanel) | `tests/karn/sanctum/widgets/test_counterfactual_panel.py` | Pair bars rendering | `[ ]` |

### Manual Verification Steps

1. Start training with 2-3 seeds
2. Launch Sanctum TUI
3. After first episode completes, observe CounterfactualPanel "Pairs" section
4. Verify each pair has a bar showing combined contribution
5. Verify high-synergy pairs are highlighted green
6. With 4+ seeds, verify "Top Combinations" shows top 5 by synergy

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| baseline_accuracy (TELE-521) | computation | Pair contributions are relative to baseline |
| Pair CounterfactualConfigs | data | Each pair needs combined evaluation |
| individual_contributions (TELE-524) | computation | Needed for pair synergy calculation |
| slot_ids | metadata | List of active slot IDs |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Pair synergy | display | Highlights synergistic pairs |
| Top Combinations | display | Sorted list for 4+ seeds |
| Seed partnership recommendations | system | Future: suggest which seeds to fossilize together |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** This method exists in the schema but relies on configs list which has NO emitter implemented.
>
> **Scalability:** With n seeds, there are n(n-1)/2 pairs. For 2 seeds: 1 pair. For 3 seeds: 3 pairs. For 4 seeds: 6 pairs. For 8 seeds: 28 pairs.
>
> **Display Adaptation:** The widget adapts display based on seed count:
> - **2-3 seeds:** Show all pairs inline in "Pairs" section
> - **4+ seeds:** Show "Top Combinations (by synergy)" with top 5 pairs sorted by pair_synergy descending
>
> **Pair Synergy Calculation (in widget):**
> ```python
> ind1 = individuals.get(s1, 0)
> ind2 = individuals.get(s2, 0)
> pair_synergy = pair_contribution - ind1 - ind2
> ```
>
> **Ablation-Only Mode:** Pair contributions are NOT available in ablation_only mode (lines 173-174). The widget shows "(Pair interactions available at episode end)" message instead.
>
> **For Two Seeds:** When n_seeds=2, the "all enabled" configuration IS the pair, so `pair_contributions()` correctly returns it. The panel displays this in the Pairs section rather than duplicating it in Combined.
