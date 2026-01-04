# Telemetry Record: [TELE-332] Exploding Layers Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-332` |
| **Name** | Exploding Layers Count |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many network layers have exploding gradients, indicating gradient flow instability?"

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

> Count of network layers where gradient norms are excessively large (exploding gradients).
>
> Computed as: `sum(1 for layer in layer_stats if layer.large_fraction > 0.1)`
>
> A layer is considered "exploding" when more than 10% of its gradient elements exceed the large threshold (default: 10.0). This indicates:
> - Gradient magnitudes are too large for stable training
> - Updates will be dominated by these layers (gradient imbalance)
> - Numerical instability is developing (potential precursor to NaN/Inf)
>
> Common causes include:
> - Learning rate too high
> - Reward scale issues in RL
> - Insufficient gradient clipping
> - Deep network without proper normalization
> - Exploding gradients in recurrent architectures (LSTM/GRU)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `exploding_layers == 0` | All layers have stable gradient magnitudes |
| **Warning** | `1 <= exploding_layers <= 2` | Minor gradient instability, monitor trend |
| **Critical** | `exploding_layers >= 3` | Significant gradient explosion, training compromised |

**Display Behavior:**
- `== 0`: Green text in ActionHeadsPanel, excluded from AnomalyStrip
- `> 0`: Red in AnomalyStrip ("exploding: N"), red in ActionHeadsPanel
- Web PolicyDiagnostics: `critical` class styling when > 0

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
# Exploding layer detection: >10% of gradients exceed large threshold (10.0)
exploding = sum(1 for s in layer_stats if s.large_fraction > 0.1)
# ...
return {
    "dead_layers": dead,
    "exploding_layers": exploding,
    # ...
}
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `aggregate_layer_gradient_health()` returns dict with exploding_layers | `simic/telemetry/emitters.py:735` |
| **2. Collection** | Included in metrics dict passed to emit function | `simic/telemetry/emitters.py:829` |
| **3. Aggregation** | `PPOUpdatePayload.exploding_layers` carries value | `leyline/telemetry.py:664` |
| **4. Delivery** | Aggregator writes to `TamiyoState.exploding_layers` | `karn/sanctum/aggregator.py:861` |

```
[collect_per_layer_gradients()] --> [LayerGradientStats.large_fraction] --> [aggregate_layer_gradient_health()] --> [metrics dict] --> [emit_ppo_update_event()] --> [PPOUpdatePayload.exploding_layers] --> [aggregator._handle_ppo_update()] --> [TamiyoState.exploding_layers]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `exploding_layers` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.exploding_layers` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 875 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| AnomalyStrip | `widgets/anomaly_strip.py:41,108,135-136` | Displayed as "exploding: N" in red when > 0 |
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py:770-774` | Shows Exploding:N/total with green (0) or red (>0) styling |
| PolicyDiagnostics (Web) | `overwatch/web/src/components/PolicyDiagnostics.vue:42-43,130-136` | Health indicator with critical class when > 0 |
| GradientHeatmap (Web) | `overwatch/web/src/components/GradientHeatmap.vue:62-63` | Computed property for exploding layer display, used in hasIssues |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `aggregate_layer_gradient_health()` computes exploding layer count
- [x] **Transport works** — Value propagates through PPOUpdatePayload to aggregator
- [x] **Schema field exists** — `TamiyoState.exploding_layers: int = 0` at schema.py:875
- [x] **Default is correct** — Default 0 appropriate before first PPO update
- [x] **Consumer reads it** — AnomalyStrip, ActionHeadsPanel, and Overwatch web components read field
- [x] **Display is correct** — Red in AnomalyStrip, red in ActionHeadsPanel when > 0
- [x] **Thresholds applied** — Color coding applied: 0=green/hidden, >0=red

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_telemetry_fields.py:66-91` | `test_exploding_layer_detection` | `[x]` |
| Unit (emitter) | `tests/simic/test_telemetry_fields.py:15-22` | `test_empty_list_returns_defaults` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py:129-137` | `test_gradient_health_metrics` | `[x]` |
| Unit (backend) | `tests/karn/sanctum/test_backend.py:105,117` | PPO update populates exploding_layers | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py:184` | ActionHeadsPanel snapshot | `[x]` |
| Web (Vue) | `overwatch/web/src/components/__tests__/PolicyDiagnostics.spec.ts:202-223` | `exploding_layers with critical/good class` | `[x]` |
| Web (Vue) | `overwatch/web/src/components/__tests__/GradientHeatmap.spec.ts:341-427` | Multiple exploding_layers scenarios | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe AnomalyStrip at bottom of screen
4. Verify "exploding" does NOT appear during healthy training
5. Observe ActionHeadsPanel "Flow" row - should show "Exploding:0/N" in green
6. To verify warning display: Would require artificially creating exploding gradients
   - Increase learning rate dramatically or use improper reward scaling
   - Observe "exploding: N" appears in red in AnomalyStrip
   - Observe Exploding:N/total changes to red styling

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Gradient computation | computation | Requires valid backprop pass with gradient retention |
| Per-layer gradient collection | system | `collect_per_layer_gradients()` must be called during PPO update |
| `LayerGradientStats.large_fraction` | computation | Must accurately measure fraction of large gradients (> 10.0 threshold) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| AnomalyStrip has_anomalies | display | `exploding_layers > 0` contributes to anomaly detection |
| ActionHeadsPanel gradient flow display | display | Shows layer health status |
| PolicyDiagnostics web component | display | Drives critical indicator styling |
| GradientHeatmap hasIssues | display | Combined with dead_layers for issue detection |
| TELE-331 dead_layers | related | Combined provide complete gradient health picture |
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
- Computation layer: `aggregate_layer_gradient_health()` correctly counts exploding layers via large_fraction > 0.1
- Transport layer: Value correctly propagates through PPOUpdatePayload
- Aggregator layer: `_handle_ppo_update()` correctly writes to TamiyoState
- Display layer: Both TUI widgets (AnomalyStrip, ActionHeadsPanel) and Web components correctly consume and display

### Detection Threshold

The 10% large fraction threshold is calibrated for PPO:
- `LayerGradientStats.large_fraction` tracks gradients > 10.0 (configured via `large_threshold` parameter)
- When > 10% of gradient elements exceed this threshold, the layer is considered "exploding"
- This threshold catches gradient instability before it leads to NaN/Inf

### Relationship to Gradient Clipping

Exploding layers are detected **before** gradient clipping is applied:
- The default exploding threshold (10.0) is 20x the default clip norm (0.5)
- Gradients at this magnitude will be heavily scaled down by clipping
- Detection provides early warning that clipping is working overtime

### Relationship to Other Gradient Metrics

`exploding_layers` is one of three network-level gradient health metrics:
- `dead_layers` (TELE-331): Vanishing gradients (too small)
- `exploding_layers`: Exploding gradients (too large)
- `nan_grad_count` / `inf_grad_count`: Numerical instability

Together these provide comprehensive gradient health monitoring. All are computed in the same `aggregate_layer_gradient_health()` function for efficiency.

### Visual Indicator Philosophy

- **AnomalyStrip**: Only shows when > 0 (hide good news, show problems) - displayed in RED (more severe than dead_layers)
- **ActionHeadsPanel**: Always shows Exploding:N/total (consistent visibility, color-coded health)
- **PolicyDiagnostics (Web)**: Shows count with `critical` class styling when > 0 (vs `warning` for dead_layers)

This follows the project's "dim when good, bright when bad" visibility philosophy, with exploding gradients treated as more severe than vanishing gradients (red vs yellow).

### Why Exploding is More Severe Than Dead

Dead layers (vanishing gradients) cause slow learning but are stable. Exploding layers indicate active numerical instability that can cascade to NaN/Inf. Hence:
- `dead_layers > 0`: Yellow/warning severity
- `exploding_layers > 0`: Red/critical severity
