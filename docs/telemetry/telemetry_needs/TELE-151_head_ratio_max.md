# Telemetry Record: [TELE-151] Per-Head Ratio Max

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-151` |
| **Name** | Per-Head Ratio Max |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is any individual action head experiencing disproportionate policy shifts that might be hidden when looking at aggregate ratio statistics?"

This metric enables detection of per-head ratio explosion. Individual head ratios can appear healthy while the joint (product) ratio exceeds the PPO clip range, causing training instability. This is the "Policy V2" multi-head ratio explosion detection pattern.

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
| **Type** | 8 `float` fields |
| **Units** | ratio (dimensionless) |
| **Range** | `[0.0, +inf)` — theoretically unbounded, practically clamped by log-ratio limits |
| **Precision** | 3 decimal places for display |
| **Default** | `1.0` each (neutral ratio = no policy change) |

### Fields

| Field Name | Head | Description |
|------------|------|-------------|
| `head_op_ratio_max` | op | Operation head (WAIT/GROW/MUTATE/KILL) |
| `head_slot_ratio_max` | slot | Slot selection head |
| `head_blueprint_ratio_max` | blueprint | Blueprint selection head |
| `head_style_ratio_max` | style | Style selection head |
| `head_tempo_ratio_max` | tempo | Tempo selection head |
| `head_alpha_target_ratio_max` | alpha_target | Alpha target head |
| `head_alpha_speed_ratio_max` | alpha_speed | Alpha speed head |
| `head_alpha_curve_ratio_max` | alpha_curve | Alpha curve head |

### Semantic Meaning

> The maximum value of the probability ratio `pi_new(a|s) / pi_old(a|s)` observed for each action head during the last PPO update.
>
> For each head, the ratio is computed as:
> ```
> ratio[head] = exp(log_prob_new[head] - log_prob_old[head])
> ```
>
> A ratio of 1.0 means the policy probability for that head's actions is unchanged. Ratios significantly above 1.0 indicate the policy is becoming more likely to take certain actions; ratios below 1.0 indicate decreased likelihood.
>
> The log-ratio is clamped to `[-20.0, +20.0]` before exponentiation to prevent numerical overflow (exp(88) overflows float32).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.8 <= ratio <= 1.2` | Within PPO clip range |
| **Warning** | `0.5 <= ratio <= 1.5` (outside clip) | Moderate policy shift |
| **Critical** | `ratio < 0.5` or `ratio > 1.5` | Significant policy divergence |

**Threshold Source:** Inline in `ActionHeadsPanel._ratio_color()` — uses PPO clip range (epsilon ~ 0.2) as the basis.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, per-head ratio computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | ~680-704 (computation), ~980-992 (aggregation) |

```python
# Per-head ratio computation (lines 680-695)
for key in HEAD_NAMES:
    log_ratio = log_probs[key] - old_log_probs[key]
    log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
    per_head_ratios[key] = torch.exp(log_ratio_clamped)

# Track max across epochs (lines 689-695)
with torch.inference_mode():
    per_head_ratio_max_tensors = [per_head_ratios[k].max() for k in HEAD_NAMES]
    per_head_ratio_max_values = torch.stack(per_head_ratio_max_tensors).cpu().tolist()
    for key, max_val in zip(HEAD_NAMES, per_head_ratio_max_values):
        head_ratio_max_across_epochs[key] = max(head_ratio_max_across_epochs[key], max_val)

# Aggregation into result dict (lines 980-992)
for key in HEAD_NAMES:
    ratio_key = f"head_{key}_ratio_max"
    max_val = head_ratio_max_across_epochs[key]
    aggregated_result[ratio_key] = max_val if max_val != float("-inf") else 1.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `src/esper/simic/telemetry/emitters.py:851-858` |
| **2. Collection** | `PPOUpdatePayload` dataclass | `src/esper/leyline/telemetry.py:703-710` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `src/esper/karn/sanctum/aggregator.py:842-849` |
| **4. Delivery** | Written to `TamiyoState.head_*_ratio_max` | `src/esper/karn/sanctum/schema.py:927-934` |

```
[PPOAgent.update()]
    --> aggregated_result["head_{head}_ratio_max"]
    --> [TelemetryEmitter.emit_ppo_update(metrics=...)]
    --> [PPOUpdatePayload]
    --> [SanctumAggregator._handle_ppo_update()]
    --> [TamiyoState.head_*_ratio_max]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Fields** | `head_slot_ratio_max`, `head_blueprint_ratio_max`, `head_style_ratio_max`, `head_tempo_ratio_max`, `head_alpha_target_ratio_max`, `head_alpha_speed_ratio_max`, `head_alpha_curve_ratio_max`, `head_op_ratio_max` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.head_{head}_ratio_max` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 927-934 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py:383-410` | Row 5-6: Ratio values with color coding + bar visualization |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes per-head ratio max
- [x] **Transport works** — Values passed through `metrics` dict to emitter
- [x] **Schema field exists** — 8 fields defined in `TamiyoState` with defaults
- [x] **Default is correct** — 1.0 appropriate (neutral ratio before first PPO update)
- [x] **Consumer reads it** — `ActionHeadsPanel` accesses via dynamic `getattr()`
- [x] **Display is correct** — Values rendered with 3 decimal places and ratio bar
- [x] **Thresholds applied** — `_ratio_color()` applies PPO clip-based thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | — | `[ ]` |
| Unit (aggregator) | — | — | `[ ]` |
| Integration (end-to-end) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe ActionHeadsPanel "Ratio" row showing per-head ratio values
4. Verify ratio values update after each PPO batch
5. Verify color coding: green for [0.8, 1.2], yellow for [0.5, 0.8) or (1.2, 1.5], red outside
6. Verify ratio bars reflect deviation from 1.0 (fuller bar = more deviation)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Per-head log probabilities | computation | Requires valid policy forward pass with factored action heads |
| Old log probabilities | data | Stored from rollout collection |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `joint_ratio_max` | telemetry | Product of per-head ratios (computed in log-space) |
| ActionHeadsPanel display | widget | Visual ratio health indicator per head |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record creation |

---

## 8. Notes

> **Design Decision:** Per-head ratio tracking was added as "Policy V2" to detect multi-head ratio explosion. The key insight is that with 8 independent action heads, each head could have a ratio of 1.2 (within clip range), but the joint ratio would be 1.2^8 = 4.3, far exceeding any reasonable clip range. This makes aggregate ratio statistics (ratio_min, ratio_max, ratio_std) insufficient for detecting policy instability in factored action spaces.
>
> **Log-Ratio Clamping:** The log-ratio is clamped to `[-20, +20]` before exponentiation. This prevents `exp(88)` overflow while still allowing detection of extreme policy shifts. A clamped value of 20 corresponds to ratio = exp(20) ~ 485 million, which is already indicative of catastrophic policy divergence.
>
> **Default Value Semantics:** The default of 1.0 means "no policy change" which is correct for:
> - Before first PPO update (no policy has been updated yet)
> - If no updates occurred in an epoch (ratio tracking was initialized to -inf, converted to 1.0)
> Consumers can distinguish "healthy 1.0" from "no data 1.0" by checking `ppo_updates_count` or `ppo_data_received`.
>
> **Related Metrics:**
> - `TELE-150` (if exists): Per-head entropy (measures exploration per head)
> - `ratio_max`, `ratio_min`, `ratio_std`: Aggregate ratio statistics (global, not per-head)
> - `joint_ratio_max`: Product of per-head ratios (detects multi-head explosion)
