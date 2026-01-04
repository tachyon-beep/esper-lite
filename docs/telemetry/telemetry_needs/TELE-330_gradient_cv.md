# Telemetry Record: [TELE-330] Gradient Coefficient of Variation

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-330` |
| **Name** | Gradient Coefficient of Variation |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are gradients flowing uniformly across network layers, or is there uneven distribution indicating training instability?"

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
| **Type** | `float` |
| **Units** | dimensionless ratio (std/mean) |
| **Range** | `[0.0, +inf)` — typically 0.0 to 5.0 in practice |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Coefficient of Variation (CV) of gradient norms across action heads. Computed as:
>
> CV = std(grad_norms) / |mean(grad_norms)|
>
> Where grad_norms is the vector of per-head gradient norms collected during a PPO update.
>
> Low CV indicates uniform gradient flow across all heads (healthy).
> High CV indicates uneven gradient distribution — some heads may be dominating or starving the learning signal.
>
> This replaced the original "gradient SNR" metric per DRL expert review, as SNR was computed incorrectly (var/mean^2 is noise-to-signal, not signal-to-noise).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `CV < 0.5` | Uniform gradient flow, high signal quality |
| **Warning** | `0.5 <= CV < 2.0` | Moderate variance, worth monitoring |
| **Critical** | `CV >= 2.0` | Uneven gradient distribution, potential training instability |

**Threshold Source:** Widget logic in `action_heads_panel.py` lines 764-765

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after backward pass and before optimizer step |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | ~872-883 |

```python
# Gradient CV: coefficient of variation = std/|mean| (per DRL expert)
# Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy gradients
# PERF: Compute on CPU using already-transferred all_norms (avoids extra sync)
n = len(all_norms)
if n > 1 and any(v > 0 for v in all_norms):
    grad_mean = sum(all_norms) / n
    grad_var = sum((x - grad_mean) ** 2 for x in all_norms) / (n - 1)
    grad_std = grad_var ** 0.5
    grad_cv = grad_std / max(abs(grad_mean), 1e-8)
else:
    grad_cv = 0.0
metrics["gradient_cv"].append(grad_cv)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` with `gradient_cv` field | `simic/telemetry/emitters.py` L866 |
| **2. Collection** | `PPOUpdatePayload.gradient_cv` field | `leyline/telemetry.py` L751 |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py` L975 |
| **4. Delivery** | Written to `snapshot.tamiyo.gradient_quality.gradient_cv` | `karn/sanctum/schema.py` L777 |

```
[PPOAgent.update()] --metrics dict--> [emit_ppo_update()] --PPOUpdatePayload--> [Aggregator] --> [GradientQualityMetrics.gradient_cv]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `GradientQualityMetrics` (nested in `TamiyoState`) |
| **Field** | `gradient_cv` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.gradient_quality.gradient_cv` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 777 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` L763-767 | Displayed with status (stable/warn/BAD) and color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes CV from per-head gradient norms
- [x] **Transport works** — `emit_ppo_update()` includes `gradient_cv` in payload
- [x] **Schema field exists** — `GradientQualityMetrics.gradient_cv: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — HeadsPanel accesses `snapshot.tamiyo.gradient_quality.gradient_cv`
- [x] **Display is correct** — Value renders as "CV:X.XXX stable/warn/BAD"
- [x] **Thresholds applied** — Green (<0.5), yellow (0.5-2.0), red (>=2.0)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (payload) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_creation` | `[x]` |
| Unit (payload defaults) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_defaults` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_gradient_quality_metrics_defaults` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_gradient_quality_metrics` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py` | `test_heads_panel_renders_gradient_quality_diagnostics` | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HeadsPanel "Flow:" section
4. Verify CV value updates after each PPO batch
5. Verify color coding: green for <0.5, yellow for 0.5-2.0, red for >=2.0

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after PPO update completes |
| Per-head gradient norms | computation | Requires backward pass with gradients in each action head |
| `all_norms` collection | internal | Gradient norms from slot, blueprint, alpha, target, speed, curve, op heads |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HeadsPanel display | widget | Visual indicator of gradient flow health |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record created |

---

## 8. Notes

> **Design Decision:** CV is computed from per-head gradient norms, not per-layer or per-parameter. This provides action-head-level granularity which is most relevant for multi-head PPO training.
>
> **DRL Expert Review:** This metric replaced the original "gradient_snr" which was computed as var/mean^2 (noise-to-signal ratio, not signal-to-noise). CV = std/|mean| is the standard coefficient of variation and is self-explanatory.
>
> **Performance Note:** CV computation is done on CPU using already-transferred `all_norms` to avoid an extra GPU sync. The gradient norms are already transferred as part of per-head gradient tracking.
>
> **Edge Case:** If all gradient norms are zero or there's only one head with gradients, CV defaults to 0.0 to avoid division issues.
