# Telemetry Record: [TELE-161] Log Probability Max

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-161` |
| **Name** | Log Probability Max |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are action probabilities in a healthy range, or do log probabilities indicate numerical underflow risk?"

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
| **Units** | nats (natural log base) |
| **Range** | `(-inf, 0.0]` — log probabilities are always <= 0 |
| **Precision** | 1 decimal place for display |
| **Default** | `NaN` |

### Semantic Meaning

> Maximum log probability across all actions in the current batch. Since log probabilities are always <= 0 (log of probability bounded [0, 1]), this is the "highest" (least negative) log probability observed. Paired with `log_prob_min` to show the action probability range.
>
> Formula: max(log π(a|s)) for all sampled actions a and states s in batch
>
> Healthy range: -0.5 to 0.0 (actions have reasonable probability)
> Warning: < -1.0 (some actions becoming unlikely)
> Critical: < -50 (action probabilities underflowing)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `log_prob_max > -1.0` | Action probabilities in reasonable range |
| **Warning** | `-50 < log_prob_max <= -1.0` | Some action probabilities declining, monitor |
| **Critical** | `log_prob_max <= -50` or `log_prob_max > 0` | Numerical underflow imminent or invalid action |

**Threshold Source:** `health_status_panel.py _get_log_prob_status()` — uses log_prob_min thresholds (-50 warning, -100 critical) applied to both min and max

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after action probability computation across all epochs |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._compute_policy_loss()` |
| **Line(s)** | 638-644 (tracking across epochs), 978 (final aggregation) |

```python
# Track log prob extremes across all heads (NaN predictor)
all_log_probs = torch.cat([log_probs[k] for k in HEAD_NAMES])
if all_log_probs.numel() > 0:
    epoch_extremes = torch.stack([all_log_probs.min(), all_log_probs.max()]).cpu().tolist()
    epoch_log_prob_min, epoch_log_prob_max = epoch_extremes
    log_prob_min_across_epochs = min(log_prob_min_across_epochs, epoch_log_prob_min)
    log_prob_max_across_epochs = max(log_prob_max_across_epochs, epoch_log_prob_max)

# ... later, at end of update:
aggregated_result["log_prob_max"] = log_prob_max_across_epochs
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update_event()` with `metrics["log_prob_max"]` | `simic/telemetry/emitters.py:809` |
| **2. Collection** | Event payload with `log_prob_max` field in `PPOUpdatePayload` | `leyline/telemetry.py:657` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py:833` |
| **4. Delivery** | Written to `snapshot.tamiyo.log_prob_max` | `karn/sanctum/schema.py:871` |

```
[PPOAgent] --emit_ppo_update_event()--> [TelemetryEmitter] --event--> [Aggregator] --> [TamiyoState.log_prob_max]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `log_prob_max` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.log_prob_max` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 871 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py:172` | Displayed as range `[log_prob_min,log_prob_max]` in "Log Prob" row |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes log_prob_max across all epochs and heads during update
- [x] **Transport works** — Event includes log_prob_max field in PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.log_prob_max: float = float("nan")`
- [x] **Default is correct** — NaN appropriate before first PPO update (indicates "no data yet")
- [x] **Consumer reads it** — HealthStatusPanel accesses `snapshot.tamiyo.log_prob_max` in line 172
- [x] **Display is correct** — Value renders with 1 decimal place in range format `[min,max]`
- [x] **Thresholds applied** — Status detection via `_get_log_prob_status()` uses log_prob_min (paired metric)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/agent/test_ppo_nan_detection.py` | Log prob tracking tested | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | PPO update handling tested | `[x]` |
| Integration (end-to-end) | Manual verification | Display in Sanctum TUI | `[x]` |
| Visual (TUI snapshot) | — | HealthStatusPanel row | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Log Prob" row
4. Verify range updates as `[min,max]` after each PPO batch
5. Trigger warning condition by monitoring very negative log_prob_min (see TELE-160)
6. Verify coloring matches status (green=ok, yellow=warning, red=critical)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Action distribution | computation | Requires valid policy forward pass across all action heads |
| Log prob computation | computation | All 8 action heads evaluated (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-160` log_prob_min | telemetry | Paired metric for range display in HealthStatusPanel |
| HealthStatusPanel | display | Uses both log_prob_min and log_prob_max for NaN risk status |
| Training monitoring | operator | Indicates numerical stability - very negative values predict NaN gradients |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Created during TELE audit phase 2 |

---

## 8. Notes

> **Design Decision:** Log probability extremes (min and max) are aggregated across all action heads and all training epochs in a single PPO update. This provides a single global view of action probability health rather than per-head tracking, reducing noise and simplifying monitoring.
>
> **Pairing with TELE-160:** log_prob_max and log_prob_min are always displayed together as a range `[min,max]` in the HealthStatusPanel. The status coloring is driven by log_prob_min (most negative = worst indicator of underflow risk), but both are shown to give operators the full picture of action probability spread.
>
> **Numerical Stability:** Log probabilities very close to 0 (e.g., -0.01) indicate high action probability (e^-0.01 ≈ 0.99), which is healthy. Very negative values (e.g., -100) indicate action probabilities approaching zero (e^-100 ≈ 0), leading to underflow in gradient computation. The critical threshold of -100 is conservative and matches IEEE 754 double precision limits.
>
> **NaN Predictor:** This metric serves as an early warning system for numerical instability. When combined with log_prob_min, the range helps diagnose whether the policy is approaching a degenerate state where some actions become numerically impossible.
