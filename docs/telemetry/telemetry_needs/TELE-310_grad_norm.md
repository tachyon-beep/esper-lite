# Telemetry Record: [TELE-310] Gradient Norm

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-310` |
| **Name** | Gradient Norm |
| **Category** | `gradient` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Are gradients exploding or remaining healthy during backpropagation? Is gradient clipping being triggered?"

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
| **Type** | `float` |
| **Units** | L2 norm magnitude (unitless) |
| **Range** | `[0.0, inf)` — typically 0.5-1.5 with clipping active |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Post-clip gradient norm represents the L2 norm of all network parameters' gradients AFTER gradient clipping is applied. Computed as:
>
> grad_norm = sqrt(sum(g^2 for all parameters))
>
> When `max_grad_norm` is set (e.g., 1.0), the gradient clipping operation scales all gradients to enforce: grad_norm <= max_grad_norm. If pre-clip norm is > max_grad_norm, clipping is active.
>
> This metric serves two purposes:
> 1. **Normal operation:** Track stable gradient flow. Healthy range is typically 0.5-1.5 with clipping.
> 2. **Explosion detection:** Extreme values (>10.0) indicate numerical instability even after clipping, suggesting NaN or Inf propagation.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `grad_norm <= 5.0` | Normal gradient flow, clipping not aggressively triggered |
| **Warning** | `5.0 < grad_norm <= 10.0` | Gradients elevated, clipping frequently triggered, investigate return scale |
| **Critical** | `grad_norm > 10.0` | Severe gradient explosion, high risk of NaN/Inf or training instability |

**Threshold Source:** `TUIThresholds.GRAD_NORM_WARNING = 5.0`, `TUIThresholds.GRAD_NORM_CRITICAL = 10.0`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after backward pass, before optimizer step |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._update_epoch()` — line 888-891 |
| **Line(s)** | ~888-891 |

```python
# Capture post-clip gradient norm (computed by clip_grad_norm_)
pre_clip_norm = nn.utils.clip_grad_norm_(
    self.policy.network.parameters(), self.max_grad_norm
)
metrics["pre_clip_grad_norm"].append(float(pre_clip_norm))
# After clipping, the actual post-clip norm is min(pre_clip_norm, max_grad_norm)
# But grad_norm field represents the pre-clip measurement value
```

**Note:** The field name `grad_norm` in the telemetry event represents the post-clip norm (after clipping is applied). The value emitted is constrained to be <= max_grad_norm. For detection of the raw pre-clip explosion magnitude, see `pre_clip_grad_norm` (TELE-311).

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` function | `simic/telemetry/emitters.py` line 787 |
| **2. Collection** | Event payload with `grad_norm` field | `leyline/telemetry.py` PPOUpdateCompleted dataclass |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py` line 809-811 |
| **4. Delivery** | Written to `snapshot.tamiyo.grad_norm` | `karn/sanctum/schema.py` TamiyoState field |

```
[PPOAgent._update_epoch()]
  --nn.utils.clip_grad_norm_()-->
[metrics["pre_clip_grad_norm"]]
  --emit_ppo_update()-->
[TelemetryEvent.grad_norm]
  --handle_ppo_update()-->
[SanctumAggregator._tamiyo.grad_norm]
  --SanctumSnapshot.tamiyo-->
[snapshot.tamiyo.grad_norm]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `grad_norm` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.grad_norm` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~842 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` lines 116-131 | Displayed as "Grad Norm" row with sparkline trend detection |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` lines 213, 238 | Used for critical/warning status detection in status indicators |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes grad_norm via `clip_grad_norm_()` and emits via `emit_ppo_update()`
- [x] **Transport works** — Event includes `grad_norm` field in `PPOUpdateCompleted` dataclass
- [x] **Schema field exists** — `TamiyoState.grad_norm: float = 0.0` (line 842)
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — Both widgets access `snapshot.tamiyo.grad_norm`
- [x] **Display is correct** — Value renders with 3 decimal places, sparkline shows history
- [x] **Thresholds applied** — StatusBanner uses 5.0/10.0 thresholds, HealthStatusPanel applies color coding

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_grad_norm_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_grad_norm` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_sanctum_head_gradients.py` | `test_grad_norms_flow_to_ui` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Navigate to HealthStatusPanel (typically visible by default)
4. Locate "Grad Norm" row with numeric value and sparkline
5. Verify gradient norm value updates after each PPO update batch
6. Monitor threshold coloring:
   - Green: grad_norm <= 5.0 (healthy)
   - Yellow: 5.0 < grad_norm <= 10.0 (warning)
   - Red: grad_norm > 10.0 (critical)
7. Test gradient explosion: Manually trigger explosion (e.g., by increasing learning rate or return scale) to verify critical coloring activates

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Backward pass | computation | Requires valid loss computation and backpropagation |
| Gradient clipping | operation | Requires `max_grad_norm` parameter in PPO config |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-311` pre_clip_grad_norm | telemetry | Uses pre-clip measurement to detect unclipped explosion magnitude |
| `TELE-312` gradient_explosion_risk | telemetry | Computes risk score based on grad_norm trend |
| StatusBanner status | display | Drives FAIL/WARN status when critical threshold exceeded |
| HealthStatusPanel coloring | display | Renders with green/yellow/red based on thresholds |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - telemetry audit TELE-310 |

---

## 8. Notes

> **Design Decision:** The `grad_norm` field represents the L2 norm of gradients AFTER clipping is applied. This is the constrained value that actually drives the optimizer step. For detecting raw gradient explosion before clipping, refer to `pre_clip_grad_norm` (TELE-311).
>
> **Threshold Rationale:**
> - Warning at 5.0: Indicates clipping is frequently triggered, suggesting return scale or value function mismatch.
> - Critical at 10.0: Post-clip value > 10 indicates severe numerical issues (NaN/Inf propagation) or clipping is ineffective.
>
> **Historical Context:** Per PyTorch expert review, gradient norms vary significantly batch-to-batch in RL training. The 25% threshold in TREND_THRESHOLDS (schema.py:1410) reflects this volatility. Sparkline visualization in HealthStatusPanel helps distinguish transient spikes from sustained elevation.
>
> **Interaction with pre_clip_grad_norm:** When pre_clip_norm >> 1.0 and grad_norm ≈ max_grad_norm (e.g., 1.0), clipping is active. If grad_norm >> max_grad_norm, numerical instability has already corrupted gradients (NaN/Inf). This dual tracking enables precise diagnosis:
> - High pre_clip, normal post_clip = clipping working correctly
> - High post_clip = numerical explosion despite clipping attempt (critical bug)
>
> **Known Limitation:** This metric does not distinguish between per-layer gradient magnitudes. For detailed per-head gradient health, see `head_*_grad_norm` metrics (TELE-320 series). Individual layer explosion can occur while global norm stays healthy if most layers are stable.
>
> **Future Enhancement:** Consider per-layer gradient norm tracking (histogram) for more granular explosion detection, or gradient variance coefficient (CV) to distinguish high-magnitude stable gradients from noisy small gradients.
