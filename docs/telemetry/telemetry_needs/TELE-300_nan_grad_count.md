# Telemetry Record: [TELE-300] NaN Gradient Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-300` |
| **Name** | NaN Gradient Count |
| **Category** | `gradient` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Are NaN values appearing in computed gradients during backpropagation? Is numerical instability occurring?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `int` |
| **Units** | count (number of NaN values across all layers) |
| **Range** | `[0, ∞)` — non-negative integer |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Cumulative count of NaN gradient values detected across all trainable parameters in the network after backpropagation. Computed by: `nan_count = Σ (number of NaN values in param.grad for each parameter)` across all layers.
>
> NaN gradients are a critical indicator of numerical instability, typically caused by:
> - Exploding gradients during backpropagation
> - Invalid loss values (log of negative numbers, sqrt of negative)
> - Uninitialized or corrupted tensor values
> - Division by zero in custom loss functions
> - Floating-point overflow in exponential operations

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `nan_grad_count == 0` | No numerical instability detected |
| **Warning** | `0 < nan_grad_count <= 5` | Minor NaN presence, training may continue but monitor closely |
| **Critical** | `nan_grad_count > 5` | Significant NaN gradient contamination, training integrity compromised |

**Threshold Logic (from status_banner.py):**
- `>0` → Red bold styling (immediate attention)
- `>5` → Red bold reverse (maximum visibility, inverted colors for emphasis)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Per-layer gradient statistics collection after backward pass |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/debug_telemetry.py` |
| **Function/Method** | `collect_per_layer_gradients()` |
| **Line(s)** | 43-120 (main function), 90 (NaN detection) |

```python
# Line 90 in collect_per_layer_gradients():
torch.isnan(flat).sum().float()  # Count NaN values in flattened gradient
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Collection** | Per-layer NaN detection in `collect_per_layer_gradients()` | `simic/telemetry/debug_telemetry.py` |
| **2. Aggregation** | Sum across all layers via `aggregate_layer_gradient_health()` | `simic/telemetry/emitters.py` |
| **3. Emission** | Payload field in `PPOUpdatePayload.nan_grad_count` | `leyline/telemetry.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.nan_grad_count` via aggregator | `karn/sanctum/aggregator.py` |

```
[collect_per_layer_gradients] --nan_count per layer--> [aggregate_layer_gradient_health] --sum--> [emit_ppo_update_event] --PPOUpdatePayload--> [SanctumAggregator] --> [TamiyoState.nan_grad_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `nan_grad_count` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.nan_grad_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 876 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Critical status detection + inline display with threshold-based styling |
| AnomalyStrip | `widgets/anomaly_strip.py` | Optional anomaly visualization |
| PolicyDiagnostics (Overwatch) | `karn/overwatch/web/src/components/PolicyDiagnostics.vue` | Web-based gradient health dashboard |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `collect_per_layer_gradients()` counts NaN using `torch.isnan().sum()`
- [x] **Aggregation works** — `aggregate_layer_gradient_health()` sums per-layer counts
- [x] **Transport works** — Field included in `PPOUpdatePayload` with required status
- [x] **Schema field exists** — `TamiyoState.nan_grad_count: int = 0` defined at line 876
- [x] **Default is correct** — 0 appropriate (no NaN present before first update)
- [x] **Consumer reads it** — StatusBanner accesses `snapshot.tamiyo.nan_grad_count` for display and status logic
- [x] **Display is correct** — Shows inline as "NaN:{count}" in banner when > 0
- [x] **Thresholds applied** — StatusBanner uses >0 (red bold) and >5 (red bold reverse) thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (collector) | `tests/simic/telemetry/test_debug_telemetry.py` | `test_collect_per_layer_gradients_detects_nan` | `[x]` |
| Unit (aggregator) | `tests/simic/telemetry/test_emitters.py` | `test_aggregate_layer_gradient_health_sums_nan_count` | `[x]` |
| Unit (payload emission) | `tests/leyline/test_telemetry_payloads.py` | `test_ppo_update_payload_nan_grad_count` | `[x]` |
| Integration (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_nan_grad_count` | `[x]` |
| Visual (TUI snapshot) | — | Manual StatusBanner verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI: `uv run sanctum`
3. Observe StatusBanner at top (container with TAMIYO border)
4. Verify: Initially should show no NaN indicator
5. To trigger NaN (manually or via test): Inject NaN into gradient
6. Verify: Banner shows "NaN:X" in red bold styling
7. If count > 5, verify: Styling changes to red bold reverse (inverted)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO backward pass | event | Requires gradients to be computed |
| `collect_per_layer_gradients()` enabled | config | Only collected in debug mode (when emitter flag is set) |
| Valid network parameters | system | Requires model.named_parameters() to yield valid tensors |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner status detection | display | Returns "critical" status when > 0 |
| Alert/intervention system | automation | Could trigger training abort if threshold exceeded |
| Gradient health dashboard (Overwatch) | display | Feeds gradient diagnostics visualization |
| Telemetry analysis (Karn) | analytics | Historical NaN detection patterns |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-06-15 | Initial | Added nan_grad_count to gradient health metrics |
| 2024-09-20 | Refactor | Moved from flat metrics to TamiyoState.gradient_health section |
| 2025-01-03 | Audit | TELE-300 record created, verified wiring end-to-end |

---

## 8. Notes

> **Design Decision:** NaN gradients are detected at the per-layer level by `collect_per_layer_gradients()` and summed to provide a single global count. This avoids per-parameter explosion while maintaining visibility into whether any layer is affected.
>
> **Performance Note:** Gradient collection uses `torch.isnan()` which is efficient (bitwise operation) but the full per-layer collection only runs when debug telemetry is enabled, keeping overhead minimal in production.
>
> **Critical Property:** Unlike most metrics with gradual degradation thresholds, NaN gradients are binary-to-count: once NaN appears, training integrity is compromised. The >5 threshold for reverse video is an additional visual cue to emphasize severity.
>
> **Limitation:** Current implementation counts total NaN across all layers. Future enhancement could track NaN per-head (slot, blueprint, style, tempo, alpha_*) for more granular failure diagnosis.
>
> **Related Metrics:** Companion metric `TELE-301` (inf_grad_count) tracks infinity gradients separately. Both are critical for gradient explosion detection and should be monitored together.
