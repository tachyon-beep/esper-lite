# Telemetry Record: [TELE-311] Gradient Norm History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-311` |
| **Name** | Gradient Norm History |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the gradient norm trend over recent batches? Are gradients exploding or stable?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `deque[float]` |
| **Units** | L2 norm (gradient magnitude) |
| **Range** | `[0.0, ∞)` non-negative |
| **Precision** | 3 decimal places for display |
| **Default** | Empty deque with maxlen=10 |

### Semantic Meaning

> Gradient norm history maintains a rolling window of the last 10 post-clipping gradient norms. Each entry represents the L2 norm of the full policy network's gradient vector after the clipping operation (`clip_grad_norm_`). This serves as a sparkline visualization of gradient health trends across recent batch updates.
>
> The gradient norm measures total gradient magnitude: `||∇L||₂ = sqrt(Σᵢ (∂L/∂θᵢ)²)`
>
> Typical values:
> - `< 0.1`: Vanishing gradients (learning stalled)
> - `0.1 - 1.0`: Healthy range (clipping active)
> - `1.0 - 5.0`: Healthy to elevated (may indicate aggressive learning)
> - `> 5.0`: Warning level (gradient instability)
> - `> 10.0`: Critical (gradient explosion)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.1 <= grad_norm <= 5.0` | Stable gradients, normal training |
| **Warning** | `5.0 < grad_norm <= 10.0` | Elevated gradient norm, monitor learning rate |
| **Critical** | `grad_norm > 10.0` | Gradient explosion, immediate intervention needed |

**Threshold Source:** `TUIThresholds.GRAD_NORM_WARNING = 5.0`, `TUIThresholds.GRAD_NORM_CRITICAL = 10.0` in `/home/john/esper-lite/src/esper/karn/constants.py`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after gradient clipping |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | `VectorizedTrainingLoop.run()` |
| **Line(s)** | ~3563, 3634 |

```python
# Compute gradient norm AFTER backward() but BEFORE clipping
ppo_grad_norm = compute_grad_norm_surrogate(agent.policy.network)

# Later, emit via telemetry
emitter.emit_ppo_update(
    ppo_grad_norm=ppo_grad_norm,
    ...
)
```

The gradient norm is computed using `compute_grad_norm_surrogate()` which:
1. Collects all parameter gradients after `loss.backward()`
2. Upcasts to float64 to prevent overflow
3. Computes L2 norm using fused `torch._foreach_norm()` followed by `torch.linalg.vector_norm()`
4. Returns the scalar norm value

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py` |
| **2. Collection** | Event payload with `grad_norm` field (post-clip value) | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` appends to deque | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.grad_norm_history` | `karn/sanctum/schema.py` |

```
[PPOAgent backward] --> [compute_grad_norm_surrogate] --> [TelemetryEmitter.emit_ppo_update()]
  --> [TelemetryEvent: grad_norm=X.XXX]
  --> [SanctumAggregator.handle_ppo_update()]
  --> [append to grad_norm_history deque]
  --> [TamiyoState.grad_norm_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `grad_norm_history` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.grad_norm_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~947 |

**Schema Definition:**
```python
grad_norm_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Rendered as 10-char sparkline with trend arrow (lines 123-133) |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Single current value used for critical/warning status detection |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_grad_norm_surrogate()` computes norm after backward pass
- [x] **Transport works** — `PPOUpdateEvent.grad_norm` carries value through telemetry hub
- [x] **Schema field exists** — `TamiyoState.grad_norm_history: deque[float]` with maxlen=10
- [x] **Default is correct** — Empty deque with maxlen=10 appropriate before first PPO update
- [x] **Consumer reads it** — Both HealthStatusPanel and StatusBanner access `snapshot.tamiyo.grad_norm_history`
- [x] **Display is correct** — Sparkline renders with width=10, matches deque maxlen
- [x] **Thresholds applied** — StatusBanner and HealthStatusPanel use 5.0/10.0 thresholds consistently

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_gradient_norm_computed` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_appends_grad_norm_history` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_grad_norm_history_reaches_tui` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Grad Norm" row with sparkline
4. Verify sparkline updates after each PPO batch (new point added to deque)
5. Verify deque maintains max 10 values (old values dropped)
6. Artificially increase learning rate or disable gradient clipping to trigger warning/critical coloring
7. Verify StatusBanner shows "Grad" status when grad_norm exceeds thresholds

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-310` | telemetry | `grad_norm` (single scalar) is appended to history each update |
| PPO update cycle | event | Only populated after first PPO update completes |
| `clip_grad_norm_()` | operation | Gradient clipping operation must complete successfully |
| Network gradient computation | operation | Requires valid policy forward pass and loss.backward() |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel sparkline | display | Visualizes trend with 10-point rolling window |
| StatusBanner status detection | display | Uses single current value for health status |
| Trend detection | display | `detect_trend()` analyzes deque to show ↑/↓/→ indicator |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation as part of telemetry audit TELE-2 |

---

## 8. Notes

> **Design Rationale:** The history deque with maxlen=10 provides a compact sparkline visualization of recent gradient norm trends without consuming significant memory. The maxlen=10 (width) matches the sparkline width in the TUI for 1:1 visual mapping.
>
> **Post-Clipping Values:** The `grad_norm` field in the telemetry event represents the gradient norm AFTER `clip_grad_norm_()` is applied, meaning values are typically clamped to `max_grad_norm` (default 40.0). This is the "post-clip" value, NOT the pre-clip value (which is separately tracked in `pre_clip_grad_norm` for diagnostics).
>
> **Gradient Computation:** Uses `compute_grad_norm_surrogate()` for efficient float64 overflow-safe computation. The function uses PyTorch's fused `_foreach_norm()` kernel for minimal GPU-CPU sync overhead.
>
> **Threshold Alignment:** The 5.0/10.0 thresholds represent post-clipping norm values. With default max_grad_norm=40.0, these thresholds imply the gradient magnitude before clipping exceeded 200-400x the clip limit (very unhealthy). Values > 5.0 indicate the clipping operation is aggressively suppressing gradients, suggesting learning rate misconfiguration or training instability.
>
> **Known Limitation:** The deque is transient (not persisted to disk) and resets between training sessions. For offline analysis of gradient trends across full training runs, refer to `TELE-310` (grad_norm scalar) which is logged via Karn MCP telemetry.
>
> **Future Improvement:** Consider adding exponential moving average (EMA) of gradient norm alongside raw history for smoothed trend detection.
