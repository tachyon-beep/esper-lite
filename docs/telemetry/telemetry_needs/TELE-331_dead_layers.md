# Telemetry Record: [TELE-331] Dead Layers Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-331` |
| **Name** | Dead Layers Count |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many network layers have vanishing gradients, indicating gradient flow blockage?"

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
| **Units** | count of layers |
| **Range** | `[0, total_layers]` — non-negative |
| **Precision** | integer count |
| **Default** | `0` |

### Semantic Meaning

> Count of network layers where gradient norms are effectively zero (vanishing gradients).
>
> Computed as: `sum(1 for layer in layer_stats if layer.zero_fraction > 0.9)`
>
> A layer is considered "dead" when more than 90% of its gradient elements are zero. This indicates:
> - Gradient flow is blocked at that layer
> - Earlier layers cannot receive learning signal
> - Network capacity is underutilized
>
> Common causes include:
> - ReLU activation saturation (dying ReLU problem)
> - Vanishing gradient in deep networks
> - Improper weight initialization
> - Learning rate too low for deep layers

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `dead_layers == 0` | All layers receiving gradient signal |
| **Warning** | `1 <= dead_layers <= 2` | Minor gradient flow issues, monitor trend |
| **Critical** | `dead_layers >= 3` | Significant gradient blockage, training compromised |

**Display Behavior:**
- `== 0`: Green text in ActionHeadsPanel, excluded from AnomalyStrip
- `> 0`: Yellow in AnomalyStrip ("dead layers: N"), red in ActionHeadsPanel

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Per-layer gradient statistics collection during PPO update |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py` |
| **Function/Method** | `aggregate_layer_gradient_health()` |
| **Line(s)** | 687-738 |

```python
# Dead layer detection: >90% zero gradients indicates vanishing gradient
dead = sum(1 for s in layer_stats if s.zero_fraction > 0.9)
# ...
return {
    "dead_layers": dead,
    # ...
}
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `aggregate_layer_gradient_health()` returns dict with dead_layers | `simic/telemetry/emitters.py:734` |
| **2. Collection** | Included in metrics dict passed to emit function | `simic/telemetry/emitters.py:828` |
| **3. Aggregation** | `PPOUpdatePayload.dead_layers` carries value | `leyline/telemetry.py:663` |
| **4. Delivery** | Aggregator writes to `TamiyoState.dead_layers` | `karn/sanctum/aggregator.py:860` |

```
[collect_per_layer_gradients()] --> [LayerGradientStats.zero_fraction] --> [aggregate_layer_gradient_health()] --> [metrics dict] --> [emit_ppo_update_event()] --> [PPOUpdatePayload.dead_layers] --> [aggregator._handle_ppo_update()] --> [TamiyoState.dead_layers]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `dead_layers` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.dead_layers` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 874 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| AnomalyStrip | `widgets/anomaly_strip.py:40,107,137-138` | Displayed as "dead layers: N" in yellow when > 0 |
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py:769-774` | Shows Dead:N/total with green (0) or red (>0) styling |
| PolicyDiagnostics (Web) | `overwatch/web/src/components/PolicyDiagnostics.vue:39,127` | Health indicator with warning class when > 0 |
| GradientHeatmap (Web) | `overwatch/web/src/components/GradientHeatmap.vue:61` | Computed property for dead layer display |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `aggregate_layer_gradient_health()` computes dead layer count
- [x] **Transport works** — Value propagates through PPOUpdatePayload to aggregator
- [x] **Schema field exists** — `TamiyoState.dead_layers: int = 0` at schema.py:874
- [x] **Default is correct** — Default 0 appropriate before first PPO update
- [x] **Consumer reads it** — AnomalyStrip, ActionHeadsPanel, and Overwatch web components read field
- [x] **Display is correct** — Yellow in AnomalyStrip, red in ActionHeadsPanel when > 0
- [x] **Thresholds applied** — Color coding applied: 0=green/hidden, >0=yellow/red

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_telemetry_fields.py` | `test_dead_layer_detection` | `[x]` |
| Unit (emitter) | `tests/simic/test_telemetry_fields.py` | `test_empty_list_returns_defaults` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py:129-137` | `test_gradient_health_metrics` | `[x]` |
| Unit (backend) | `tests/karn/sanctum/test_backend.py:104,116` | PPO update populates dead_layers | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py:183` | ActionHeadsPanel snapshot | `[x]` |
| Web (Vue) | `overwatch/web/src/components/__tests__/PolicyDiagnostics.spec.ts:178-194` | `dead_layers with warning/good class` | `[x]` |
| Web (Vue) | `overwatch/web/src/components/__tests__/GradientHeatmap.spec.ts` | Multiple dead_layers scenarios | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe AnomalyStrip at bottom of screen
4. Verify "dead layers" does NOT appear during healthy training
5. Observe ActionHeadsPanel "Flow" row - should show "Dead:0/N" in green
6. To verify warning display: Would require artificially creating vanishing gradients
   - Reduce learning rate dramatically or use poor initialization
   - Observe "dead layers: N" appears in yellow in AnomalyStrip
   - Observe Dead:N/total changes to red styling

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Gradient computation | computation | Requires valid backprop pass with gradient retention |
| Per-layer gradient collection | system | `collect_per_layer_gradients()` must be called during PPO update |
| `LayerGradientStats.zero_fraction` | computation | Must accurately measure fraction of zero gradients |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| AnomalyStrip has_anomalies | display | `dead_layers > 0` contributes to anomaly detection |
| ActionHeadsPanel gradient flow display | display | Shows layer health status |
| PolicyDiagnostics web component | display | Drives warning indicator styling |
| TELE-332 exploding_layers | related | Combined provide complete gradient health picture |
| TELE-300 nan_grad_count | related | All three gradient health metrics used together |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Created telemetry record; verified full wiring from emitter to display |

---

## 8. Notes

### Implementation Quality

This metric is **fully wired** end-to-end:
- Computation layer: `aggregate_layer_gradient_health()` correctly counts dead layers via zero_fraction > 0.9
- Transport layer: Value correctly propagates through PPOUpdatePayload
- Aggregator layer: `_handle_ppo_update()` correctly writes to TamiyoState
- Display layer: Both TUI widgets (AnomalyStrip, ActionHeadsPanel) and Web components correctly consume and display

### Detection Threshold

The 90% zero fraction threshold is intentionally conservative:
- Below 90%: Layer may have sparse gradients but is still learning
- Above 90%: Layer is effectively not receiving gradient signal
- This prevents false positives from normal gradient sparsity patterns

### Relationship to Other Gradient Metrics

`dead_layers` is one of three network-level gradient health metrics:
- `dead_layers`: Vanishing gradients (too small)
- `exploding_layers`: Exploding gradients (too large)
- `nan_grad_count` / `inf_grad_count`: Numerical instability

Together these provide comprehensive gradient health monitoring. All are computed in the same `aggregate_layer_gradient_health()` function for efficiency.

### Visual Indicator Philosophy

- **AnomalyStrip**: Only shows when > 0 (hide good news, show problems)
- **ActionHeadsPanel**: Always shows Dead:N/total (consistent visibility, color-coded health)
- **PolicyDiagnostics (Web)**: Shows count with warning class styling when > 0

This follows the project's "dim when good, bright when bad" visibility philosophy.
