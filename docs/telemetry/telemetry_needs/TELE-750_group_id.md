# Telemetry Record: [TELE-750] group_id

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-750` |
| **Name** | group_id |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which A/B testing group is this training run assigned to (A/B/C)?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [x] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `str \| None` |
| **Units** | N/A (categorical label) |
| **Range** | `{"A", "B", "C"}` or `None` (when not in A/B testing mode) |
| **Precision** | N/A |
| **Default** | `None` (no A/B testing) |

### Semantic Meaning

> A/B testing group label used to identify which variant of a training experiment a run belongs to. Used in dual-policy or multi-variant evaluation scenarios (e.g., comparing different reward modes, exploration strategies, or network architectures).
>
> Values are: "A", "B", "C" (case-sensitive uppercase), or None when not in A/B testing mode.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Value is set and consistent | Run is properly tagged with A/B group |
| **Warning** | Value is None during A/B testing | Group not properly set, telemetry may be incomplete |
| **Critical** | Value changes mid-run | Indicates data corruption or concurrent A/B test collision |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | A/B training harness or multi-policy training loop |
| **File** | `/home/john/esper-lite/src/esper/simic/training/dual_ab.py` |
| **Function/Method** | `train_dual_ab_comparison()` or `train_ppo_vectorized()` |
| **Line(s)** | dual_ab.py:213, vectorized.py:597, 3628 |

```python
# In dual_ab.py:213 - group_id passed to train_ppo_vectorized
group_id=group_id,  # Tag telemetry events with group

# In vectorized.py:597 - train_ppo_vectorized function signature
group_id: str = "default",  # A/B testing group identifier
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Passed via `TelemetryEvent.group_id` field | `leyline/telemetry.py` |
| **2. Collection** | Event.group_id extracted from telemetry event | `karn/sanctum/aggregator.py:785` |
| **3. Aggregation** | Validated and assigned to TamiyoState | `karn/sanctum/aggregator.py:786-787` |
| **4. Delivery** | Written to `snapshot.tamiyo.group_id` | `karn/sanctum/schema.py:971` |

```
[training loop (dual_ab.py:213)]
--> [train_ppo_vectorized(group_id="A")]
--> [TelemetryEvent(group_id=event.group_id)]
--> [aggregator.handle_ppo_update()]
--> [TamiyoState.group_id]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `group_id` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.group_id` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 971 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Color-coded group label display (lines 85-119) |
| DecisionsColumn | `widgets/tamiyo_brain/decisions_column.py` | Passes group_id to decision cards (line 390) |
| DecisionDetailScreen | `widgets/tamiyo_brain/decision_detail_screen.py` | Displays group_id in decision detail view header (line 70) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `train_ppo_vectorized()` accepts and passes group_id
- [x] **Transport works** — `TelemetryEvent.group_id` field properly defined
- [x] **Schema field exists** — `TamiyoState.group_id: str | None = None`
- [x] **Default is correct** — `None` appropriate when not in A/B testing
- [x] **Consumer reads it** — StatusBanner, DecisionsColumn, DecisionDetailScreen all access field
- [x] **Display is correct** — Values render with color coding and emoji labels
- [x] **Thresholds applied** — Group colors map A→green, B→cyan, C→magenta

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/training/test_dual_ab.py` | `test_group_id_passed_through` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_group_id_set_from_event` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_ab_telemetry_flow.py` | `test_group_id_reaches_ui` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification with dual_ab training | `[x]` |

### Manual Verification Steps

1. Start A/B training: `uv run esper dual-ab --group-a=A --group-b=B --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Verify StatusBanner displays group label with color coding (🟢 A or 🔵 B)
4. Open Decision Details screen and verify `[A]` or `[B]` appears in title
5. Verify group_id remains consistent throughout run

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| A/B training harness | system | Only populated when using dual_ab.py or explicit group_id parameter |
| TelemetryEvent emission | system | Must use `TelemetryEvent(group_id=...)` syntax |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner styling | widget | Color and emoji selection based on group_id value |
| DecisionDetailScreen header | widget | Group label displayed in title bar |
| Telemetry filtering | analytics | Users can filter/compare runs by group_id |
| Pareto frontier computation | system | May stratify results by group_id |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation, verified wiring complete |
| | | |

---

## 8. Notes

### Design Decision

Group_id is **optional** (defaults to None) to support both A/B testing and single-run modes. When None, StatusBanner skips group label display entirely. This allows the same training harness to be used for both modes without conditional code.

### Wiring Status: COMPLETE

The group_id metric has **full end-to-end wiring**:

1. **Source:** Passed from dual_ab.py training harness → vectorized.py → TelemetryEvent
2. **Transport:** Flows through aggregator event handler
3. **Schema:** Stored in TamiyoState as `str | None`
4. **Display:** Consumed by StatusBanner (primary), DecisionsColumn, DecisionDetailScreen

All three consumer widgets properly access the field and apply formatting.

### Color Coding Implementation

StatusBanner implements group-specific styling in two places:

1. **CSS Classes** (lines 85-88): Adds `group-{a,b,c}` class for stylesheet-based styling
2. **Inline Labels** (lines 116-119): Maps group → (emoji, color) with fallback for unknown groups

```python
GROUP_COLORS = {
    "A": "bright_green",      # 🟢
    "B": "bright_cyan",       # 🔵
    "C": "bright_magenta",    # 🟣
}
```

### Known Limitation

**Emission Gap:** The `emit_ppo_update_event()` function signature accepts `group_id` parameter (line 751), but VectorizedEmitter.on_ppo_update() does NOT pass this parameter when calling it (line 337). This means **group_id is not propagated through regular PPO update events**.

However, group_id IS propagated through the TelemetryEvent wrapper at line 877, which means the event still carries the group_id from the parent context. This works because:

1. TelemetryEvent is created with `group_id=group_id` (line 877)
2. Aggregator reads from `event.group_id` (line 785)
3. The emitter function's `group_id` parameter has default="default" (line 751) and is only used if explicitly passed

**Status:** FUNCTIONAL but could be cleaner - VectorizedEmitter should store group_id and pass it explicitly to emit_ppo_update_event() for clarity.

### Testing Recommendations

- Add unit test to verify group_id persists through aggregator handler
- Add integration test for full dual_ab flow
- Add snapshot test for StatusBanner rendering with each group label
- Verify CSS classes for styling are actually applied in TUI
