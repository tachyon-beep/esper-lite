# Telemetry Record: [TELE-301] Inf Gradient Count

> **Status:** `[x] Planned` `[x] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-301` |
| **Name** | Inf Gradient Count |
| **Category** | `gradient` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Are any layers experiencing Inf (infinity) gradients, indicating numerical instability?"

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
| **Type** | `int` |
| **Units** | count of gradient elements |
| **Range** | `[0, total_params]` — non-negative |
| **Precision** | integer count |
| **Default** | `0` |

### Semantic Meaning

> Count of gradient elements that contain Inf (infinity) values across all network layers.
>
> Computed as: `sum(torch.isinf(param.grad).sum() for param in network.parameters())`
>
> Inf gradients indicate numerical overflow in computation and are critical training failures.
> Unlike NaN, Inf can sometimes be recovered with adjustment, but requires immediate intervention.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `inf_grad_count == 0` | No numerical instability detected |
| **Warning** | `0 < inf_grad_count <= 5` | Minor Inf elements detected, trend should be monitored |
| **Critical** | `inf_grad_count > 5` | Significant Inf presence, training is unstable |

**Display Thresholds in StatusBanner:**
- `>0`: Display as red bold indicator (e.g., "Inf:2")
- `>5`: Display as red bold reverse (maximum visibility) per spec requirement

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Per-layer gradient collection during PPO update |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/debug_telemetry.py` |
| **Function/Method** | `collect_per_layer_gradients()` |
| **Line(s)** | ~80-130 |

```python
# Computed via torch.isinf() for each layer's gradient tensor
inf_count = torch.isinf(grad_tensor).sum()
# Result stored in LayerGradientStats.inf_count (int)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Collected in LayerGradientStats objects | `simic/telemetry/debug_telemetry.py` |
| **2. Collection** | `aggregate_layer_gradient_health()` should sum counts | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Should be in metrics dict as "inf_grad_count" | `simic/telemetry/emitters.py` |
| **4. Delivery** | Should be passed to PPOUpdatePayload.inf_grad_count | `simic/telemetry/emitters.py` |

```
[collect_per_layer_gradients()] --> [LayerGradientStats.inf_count] --X--> [aggregate_layer_gradient_health()] --> [emit_ppo_update_event()] --> [TamiyoState.inf_grad_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `inf_grad_count` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.inf_grad_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 877 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Critical status indicator with severity graduation |
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | [Investigation needed] |
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | [Investigation needed] |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `collect_per_layer_gradients()` computes inf_count per layer
- [ ] **Transport works** — `aggregate_layer_gradient_health()` does NOT sum inf_counts (BROKEN)
- [ ] **Metrics dict updated** — `emit_ppo_update_event()` hardcodes inf_grad_count=0 (BROKEN)
- [x] **Schema field exists** — `TamiyoState.inf_grad_count: int = 0` defined at schema.py:877
- [x] **Default is correct** — Default 0 appropriate before first detection
- [x] **Consumer reads it** — StatusBanner accesses `snapshot.tamiyo.inf_grad_count`
- [x] **Display is correct** — StatusBanner renders with threshold-based styling
- [x] **Thresholds applied** — StatusBanner applies >0=red bold, >5=red bold reverse

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/` | Gradient collector tests | `[ ]` |
| Unit (aggregator) | `tests/simic/telemetry/` | Layer aggregation tests | `[ ]` |
| Integration (end-to-end) | `tests/integration/` | Telemetry flow tests | `[ ]` |
| Widget (StatusBanner) | `tests/karn/sanctum/widgets/tamiyo_brain/test_status_banner.py` | `test_inf_triggers_critical_status`, `test_high_inf_count_triggers_reverse_style` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe StatusBanner at top
4. Verify no "Inf" indicator appears during healthy training
5. [CANNOT CURRENTLY VERIFY] Artificially trigger Inf gradients to verify display
   - Would require modifying gradient computation to inject Inf values
   - Current metrics pipeline doesn't propagate inf_grad_count correctly

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Gradient computation | computation | Requires valid backprop pass |
| Per-layer gradient collection | system | `collect_per_layer_gradients()` must be called |
| Layer stats aggregation | system | Currently broken - returns dict without inf_grad_count |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner critical status | display | `inf_grad_count > 0` triggers "Inf DETECTED" status |
| HealthStatusPanel | display | Likely displays inf_grad_count (verify implementation) |
| Training intervention system | system | May trigger training halt on critical threshold |

---

## 7. Wiring Issues (Broken Path)

### CRITICAL BUG #1: Missing inf_grad_count aggregation

**Location:** `src/esper/simic/telemetry/emitters.py:687-738`

**Function:** `aggregate_layer_gradient_health()`

**Issue:** The function computes `nan_count` (line 709) and includes it in return dict (line 736), but does NOT compute `inf_grad_count` from `layer_stats` even though `LayerGradientStats` contains `inf_count` field.

**Current Code:**
```python
nan_count = sum(s.nan_count for s in layer_stats)
# ... no equivalent for inf_count ...
return {
    "dead_layers": dead,
    "exploding_layers": exploding,
    "nan_grad_count": nan_count,
    "layer_gradient_health": per_layer_health,
}
```

**Expected:** Should include `"inf_grad_count": sum(s.inf_count for s in layer_stats)`

---

### CRITICAL BUG #2: Hardcoded zero in emit_ppo_update_event()

**Location:** `src/esper/simic/telemetry/emitters.py:827`

**Function:** `emit_ppo_update_event()`

**Issue:** The function hardcodes `inf_grad_count=0` instead of retrieving it from metrics dict.

**Current Code:**
```python
inf_grad_count=0,
```

**Expected:** Should use `inf_grad_count=metrics.get("inf_grad_count", 0)` or similar.

---

## 8. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Created telemetry record; identified broken wiring in emitter pipeline |
| | | Bug #1: aggregate_layer_gradient_health() missing inf_grad_count sum |
| | | Bug #2: emit_ppo_update_event() hardcodes inf_grad_count=0 |

---

## 9. Notes

### Current Status

The metric is **partially wired**:
- ✓ Computation layer: `LayerGradientStats` correctly computes `inf_count` via `torch.isinf()`
- ✓ Schema layer: `TamiyoState` correctly defines `inf_grad_count` field
- ✓ Display layer: `StatusBanner` correctly reads and displays it with proper thresholds
- ✗ **BROKEN:** Transport layer - metrics dict never receives aggregated inf_grad_count value

### Why This Matters

- StatusBanner can detect inf_grad_count=0 (default) and display correctly when nothing is wrong
- But if Inf gradients occur, they are computed but never aggregated and never reach the display
- This creates false confidence: "No Inf detected" even when numerical instability is occurring

### Test Note

The test file `test_status_banner.py` includes comprehensive tests for inf_grad_count display behavior (lines 85-141), but these tests only verify display logic, not the actual metric flow from emitter to schema.

### Future Verification

Once bugs are fixed:
1. Re-run telemetry flow integration tests
2. Trigger training scenario that produces Inf gradients
3. Verify StatusBanner displays "Inf:N" with correct styling
4. Verify count increases as expected with more Inf elements

### Design Decision

The separation of `inf_grad_count` and `nan_grad_count` is intentional per PyTorch expert recommendation. While they're similar, Inf has different recovery characteristics than NaN and should be tracked separately.
