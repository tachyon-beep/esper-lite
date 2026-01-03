# Telemetry Record: [TELE-160] Log Probability Min

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-160` |
| **Name** | Log Probability Min |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Are log probabilities approaching numerical underflow? Will gradient computation fail with NaN soon?"

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
| **Range** | `(-inf, 0.0]` — log probabilities are always ≤ 0 |
| **Precision** | 1 decimal place for display |
| **Default** | `NaN` (before first PPO update) |

### Semantic Meaning

> The minimum (most negative) log probability across all actions and timesteps in a PPO batch update.
>
> Log probability is the natural logarithm of an action probability: log_prob = log(π(a|s)).
> By definition, 0 ≤ π(a|s) ≤ 1, so log_prob ≤ 0.
>
> Very negative log probabilities indicate actions becoming nearly impossible:
> - log_prob = -1 → probability ≈ 0.37 (healthy action)
> - log_prob = -50 → probability ≈ e^-50 ≈ 2e-22 (nearly zero)
> - log_prob = -100 → probability ≈ e^-100 ≈ 4e-44 (numerical underflow imminent)
>
> At such extreme probabilities, floating-point arithmetic loses precision, causing:
> 1. exp(log_prob) → 0 (underflow)
> 2. Ratio computation: exp(log_prob_new - log_prob_old) becomes 0/0 → NaN
> 3. KL divergence computation → NaN gradients → training divergence
>
> This metric is a LEADING INDICATOR of NaN risk—shows the problem before it crashes training.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `log_prob_min > -50` | Actions have reasonable probabilities, no underflow risk |
| **Warning** | `-100 < log_prob_min ≤ -50` | Actions becoming rare, numerical precision degrading |
| **Critical** | `log_prob_min ≤ -100` | Severe numerical underflow imminent, NaN nearly guaranteed |

**Threshold Source:** HealthStatusPanel._get_log_prob_status() uses `-50` (warning) and `-100` (critical)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after computing action log probabilities across batch |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | ~636-644 (line 636: initialize as inf, lines 638-644: track min/max across all heads) |

```python
# Initialize at start of update
log_prob_min_across_epochs: float = float("inf")
log_prob_max_across_epochs: float = float("-inf")

# During each epoch, track extremes across all action heads
all_log_probs = torch.cat([log_probs[k] for k in HEAD_NAMES])
if all_log_probs.numel() > 0:
    epoch_extremes = torch.stack([all_log_probs.min(), all_log_probs.max()]).cpu().tolist()
    epoch_log_prob_min, epoch_log_prob_max = epoch_extremes
    log_prob_min_across_epochs = min(log_prob_min_across_epochs, epoch_log_prob_min)
    log_prob_max_across_epochs = max(log_prob_max_across_epochs, epoch_log_prob_max)

# After update completes, convert inf to NaN (lines 973-976)
if log_prob_min_across_epochs == float("inf"):
    log_prob_min_across_epochs = float("nan")

aggregated_result["log_prob_min"] = log_prob_min_across_epochs
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Computed in `PPOAgent.update()`, added to metrics dict | `simic/agent/ppo.py` |
| **2. Collection** | Passed through return value `PPOUpdateMetrics` | `simic/agent/types.py` |
| **3. Aggregation** | `emit_ppo_update_event()` reads from metrics dict, builds payload | `simic/telemetry/emitters.py` |
| **4. Delivery** | Written to `PPOUpdatePayload.log_prob_min` field | `leyline/telemetry.py` |

```
[PPOAgent.update()] --metrics dict--> [emit_ppo_update_event()] --PPOUpdatePayload--> [SanctumAggregator.handle_ppo_update()] --> [TamiyoState.log_prob_min]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `log_prob_min` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.log_prob_min` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~870 |
| **Default Value** | `float("nan")` (NaN indicates no data yet) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | "Log Prob" row (displays `[min,max]` range with color coding) |
| StatusBanner | (referenced in notes) | Uses critical/warning status for overall status detection |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPOAgent computes log probs during forward pass, tracks min/max
- [x] **Transport works** — Value flows through metrics dict → emit_ppo_update_event() → PPOUpdatePayload
- [x] **Schema field exists** — TamiyoState.log_prob_min: float = float("nan")
- [x] **Default is correct** — NaN appropriate before first PPO update (signals "no data")
- [x] **Consumer reads it** — HealthStatusPanel accesses snapshot.tamiyo.log_prob_min
- [x] **Display is correct** — Renders as `[min,max]` with status coloring
- [x] **Thresholds applied** — _get_log_prob_status() uses -50 warning / -100 critical

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_log_prob_extremes_tracked` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_log_prob_min_populates_tamiyo` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_log_prob_reaches_tui` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Log Prob" row
4. Verify log_prob_min updates after each PPO batch (should be negative or NaN early on)
5. Verify color changes: dim (no data) → green (healthy) → yellow (warning) → red (critical)
6. Trigger numerical issues to verify critical threshold (would require pushing policy far out of distribution)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Action distribution | computation | Requires valid policy forward pass with finite log probabilities |
| All action heads | computation | Min is computed across ALL 8 heads (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner status | display | Drives CRITICAL status when log_prob_min < -100 |
| HealthStatusPanel coloring | display | Shows warning/critical color in "Log Prob" row |
| NaN risk prediction | analysis | Primary indicator of imminent numerical underflow |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Created with PPO implementation; verified wiring |

---

## 8. Notes

> **Design Decision:** Log prob min/max are tracked separately from individual head entropies because they represent a GLOBAL SAFETY METRIC across the entire policy, not a per-head health metric. A single action with log_prob < -100 anywhere in the batch is a critical failure signal.
>
> **Why NaN Default:** NaN (not 0.0) signals "no data yet" rather than "probability = 1" (log(1) = 0). This prevents false positives early in training when no PPO updates have occurred. Consumers check `math.isnan(log_prob_min)` to distinguish "no data" from "safe values".
>
> **Numerical Underflow Explanation:** IEEE 754 float32 underflows to zero around e^-88 ≈ 1.4e-38. Log probabilities approaching -100 represent probabilities near 3.7e-44, which are WAY below the underflow threshold. At this point:
> - exp(log_prob) loses all precision
> - Ratio computation (PPO core operation) produces 0/0 → NaN
> - Any operation involving the ratio (loss, gradient) becomes NaN
> This explains why -100 is critical and -50 is warning.
>
> **Related Metrics:**
> - TELE-001 entropy: Measures policy spread; can collapse while log probs stay reasonable
> - TELE-???  ratio_max: Detects policy divergence; log_prob_min detects numerical underflow specifically
> - Head entropies (TELE-???): Per-head collapse; log_prob_min is global across all heads
>
> **Known Limitations:**
> - Only tracks batch minimum, not per-action statistics (would be too expensive)
> - No distinction between "rare but valid action" vs "impossible action from bad distribution"
> - True safety would require monitoring ratio computation directly, but log_prob_min is the leading indicator
>
> **Future Improvement:** Consider per-head log prob extremes (log_prob_min per head) to diagnose which head is causing numerical issues, similar to existing per-head entropy breakdown.
