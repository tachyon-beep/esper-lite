# Telemetry Record: [TELE-532] Shapley Ranked Slots

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-532` |
| **Name** | Shapley Ranked Slots |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which slots are contributing the most to ensemble performance, and which are dragging it down?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `list[tuple[str, float]]` |
| **Units** | (slot_id, mean_contribution) tuples |
| **Range** | Contribution typically `[-10.0, +10.0]` percentage points |
| **Precision** | 3 decimal places for display |
| **Default** | `[]` (empty list before first computation) |

### Semantic Meaning

> A sorted list of (slot_id, mean_contribution) tuples, ordered by contribution descending. This is a convenience method on ShapleySnapshot that transforms the `values` dict into a ranked list.
>
> **Structure:**
> ```python
> [("r0c1", 3.2), ("r0c0", 1.5), ("r0c2", -0.3)]
> #  ^ best       ^ second       ^ worst (negative)
> ```
>
> **Use cases:**
> 1. Display slots in contribution order (most impactful first)
> 2. Quickly identify top contributors and detractors
> 3. Support "show top N" filtering in compact views

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | All contributions positive | All slots helping |
| **Mixed** | Some positive, some negative | Ensemble has dead weight |
| **Concerning** | Multiple negative contributors | Consider pruning |

**Threshold Source:** Derived from individual slot thresholds (TELE-530)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed by `ShapleySnapshot.ranked_slots()` method from `values` dict |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `ShapleySnapshot.ranked_slots()` |
| **Line(s)** | 295-301 |

```python
def ranked_slots(self) -> list[tuple[str, float]]:
    """Return slots ranked by mean contribution (descending)."""
    return sorted(
        [(slot_id, est.mean) for slot_id, est in self.values.items()],
        key=lambda x: x[1],
        reverse=True,
    )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Not emitted separately - computed on demand | N/A |
| **2. Collection** | Part of ShapleySnapshot object | `karn/sanctum/schema.py` |
| **3. Aggregation** | Method call on stored ShapleySnapshot | N/A |
| **4. Delivery** | Accessed via `shapley_snapshot.ranked_slots()` | N/A |

```
[ShapleySnapshot.values populated]
  --method call-->
  [ranked_slots() computes sorted list]
  --returns-->
  [Widget uses for display]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ShapleySnapshot` |
| **Field** | Method: `ranked_slots()` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].shapley_snapshot.ranked_slots()` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 295-301 |
| **Default Value** | Returns `[]` when `values` is empty |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ShapleyPanel | `widgets/shapley_panel.py` (lines 70-112) | Iterates ranked slots for display in contribution order |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Method exists on ShapleySnapshot
- [x] **Transport works** - Derived from populated `values` dict
- [x] **Schema field exists** - `ranked_slots()` method at lines 295-301
- [x] **Default is correct** - Returns empty list when no values
- [x] **Consumer reads it** - ShapleyPanel calls `ranked_slots()` at line 70
- [x] **Display is correct** - Renders slots in contribution order
- [x] **Thresholds applied** - Individual slot thresholds from TELE-530

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (method) | `tests/karn/sanctum/test_schema.py` | `test_shapley_ranked_slots` | `[ ]` |
| Widget (ShapleyPanel) | `tests/karn/sanctum/widgets/test_shapley_panel.py` | Ranking display | `[ ]` |

### Manual Verification Steps

1. Populate ShapleySnapshot with test values
2. Call `ranked_slots()` method
3. Verify returns list sorted by mean descending
4. Verify empty values dict returns empty list

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| ShapleySnapshot.values | field | Source data for ranking |
| TELE-530 Shapley Values | telemetry | Must be populated first |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ShapleyPanel display order | widget | Shows slots ranked by contribution |
| Top-N contributor display | future | Could show only top N slots |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** The `ranked_slots()` method exists and works correctly when `values` is populated. The underlying `values` dict (TELE-530) is NOT YET WIRED to an emitter.
>
> **Design Rationale:** This is a derived metric (method) rather than a stored field. Computing on-demand avoids redundant storage and ensures consistency with the source `values` dict.
>
> **Performance:** O(n log n) sorting on each call. For typical slot counts (4-8), this is negligible. If performance becomes an issue, could cache the sorted result.
