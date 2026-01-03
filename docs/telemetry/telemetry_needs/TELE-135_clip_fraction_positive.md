# Telemetry Record: [TELE-135] Clip Fraction Positive

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-135` |
| **Name** | Clip Fraction Positive |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are policy probability increases being clipped? What proportion of actions are hitting the PPO clip ceiling when the policy is trying to increase their likelihood?"

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
| **Units** | fraction (0.0-1.0) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 1 decimal place for display (percentage) |
| **Default** | `0.0` |

### Semantic Meaning

> Clip fraction positive measures the proportion of policy updates where the probability ratio exceeded the upper PPO clip boundary. Computed as:
>
> clip_fraction_positive = mean(joint_ratio > 1.0 + clip_ratio)
>
> Where:
> - `joint_ratio` = product of probability ratios across all action heads (new_prob / old_prob)
> - `clip_ratio` = PPO clipping epsilon (typically 0.2)
>
> High positive clip fraction indicates the policy is aggressively increasing action probabilities for certain actions. This is normal during early training but persistent high values may indicate:
> - Policy is confidently improving on discovered strategies
> - Potential for policy collapse if combined with low entropy
>
> **Interpretation with clip_fraction_negative:**
> - Symmetric high values (both ~equal): Normal PPO training with balanced updates
> - Asymmetric high positive: Policy is predominantly increasing probabilities (encouraging actions)
> - Asymmetric high negative: Policy is predominantly decreasing probabilities (discouraging actions)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value < 0.3` | Normal clipping behavior |
| **Warning** | `0.3 <= value < 0.5` | Elevated positive clipping, monitor entropy |
| **Critical** | `value >= 0.5` | Aggressive probability increases, check for policy drift |

Note: These thresholds are heuristic; asymmetry between positive and negative is more diagnostic than absolute values.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after joint ratio computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | ~753-758 |

```python
# Directional clip fractions (per DRL expert recommendation)
# Tracks WHERE clipping occurs: asymmetry indicates directional policy drift
clip_pos = (joint_ratio > 1.0 + self.clip_ratio).float().mean()
clip_neg = (joint_ratio < 1.0 - self.clip_ratio).float().mean()
# Single GPU->CPU sync for all 3 clip metrics
clip_metrics = torch.stack([clip_fraction_t, clip_pos, clip_neg]).cpu().tolist()
metrics["clip_fraction"].append(clip_metrics[0])
metrics["clip_fraction_positive"].append(clip_metrics[1])
metrics["clip_fraction_negative"].append(clip_metrics[2])
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `PPOAgent.update()` returns `PPOUpdateMetrics` dict | `simic/agent/ppo.py` |
| **2. Collection** | `TelemetryEmitter.emit_ppo_update()` calls `emit_ppo_update_event()` | `simic/telemetry/emitters.py` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.gradient_quality.clip_fraction_positive` | `karn/sanctum/schema.py` |

```
[PPOAgent.update()] --> [PPOUpdateMetrics dict] --> [TelemetryEmitter.emit_ppo_update()]
    --> [emit_ppo_update_event()] --> [PPOUpdatePayload] --> [TelemetryHub]
    --> [SanctumAggregator._handle_ppo_update()] --> [TamiyoState.gradient_quality.clip_fraction_positive]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `GradientQualityMetrics` (nested in `TamiyoState`) |
| **Field** | `clip_fraction_positive` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.gradient_quality.clip_fraction_positive` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~784 |

```python
# Directional Clip Fraction (per DRL expert recommendation)
# These track WHERE clipping occurs, not WHETHER policy improved:
# clip+ = r > 1+epsilon (probability increases were capped)
# clip- = r < 1-epsilon (probability decreases were capped)
# Asymmetry indicates directional policy drift; symmetric high values are normal
clip_fraction_positive: float = 0.0
clip_fraction_negative: float = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Displayed with directional arrows after Clip Frac gauge |

```python
# Add directional breakdown with arrows (always show, dim when zero)
clip_pos = tamiyo.gradient_quality.clip_fraction_positive
clip_neg = tamiyo.gradient_quality.clip_fraction_negative
# Style: dim when both zero, otherwise show direction that's active
dir_style = "dim" if clip_pos == 0 and clip_neg == 0 else "cyan"
result.append(f" (up_arrow{clip_pos:.1%} down_arrow{clip_neg:.1%})", style=dir_style)
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Computed in `PPOAgent.update()` at line ~753-758
- [x] **Transport works** - Value reaches aggregator via `emit_ppo_update_event()`
- [x] **Schema field exists** - `GradientQualityMetrics.clip_fraction_positive: float = 0.0`
- [x] **Default is correct** - 0.0 appropriate before first PPO update
- [x] **Consumer reads it** - `PPOLossesPanel` accesses via `tamiyo.gradient_quality.clip_fraction_positive`
- [x] **Display is correct** - Rendered as percentage with up arrow
- [x] **Thresholds applied** - Dimmed when zero, cyan otherwise (visual feedback not health-based)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (payload) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_gradient_quality_fields` | `[x]` |
| Unit (payload defaults) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_gradient_quality_defaults` | `[x]` |
| Unit (from_dict) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_from_dict` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_gradient_quality_metrics_defaults` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_gradient_quality_fields` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | `test_directional_clip_fraction_display` | `[x]` |
| Integration (end-to-end) | - | - | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPOLossesPanel "Clip Frac" row
4. Verify directional breakdown shows up/down arrows with percentages
5. Confirm values update after each PPO batch
6. When no updates have occurred, verify display is dimmed

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| `joint_ratio` | computation | Product of probability ratios across all action heads |
| `clip_ratio` | hyperparameter | PPO clipping epsilon (default 0.2) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-136` clip_fraction_negative | telemetry | Paired metric for directional analysis |
| Directional drift detection | analysis | Asymmetry between pos/neg indicates policy direction |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and verification |

---

## 8. Notes

> **Design Decision:** Directional clip fractions were added per DRL expert recommendation to distinguish WHERE clipping occurs. The original `clip_fraction` only shows WHETHER clipping occurred, but not the direction of policy drift.
>
> **Related Metrics:** This metric should always be analyzed alongside `clip_fraction_negative` (TELE-136). The ratio between them indicates whether the policy is predominantly encouraging or discouraging actions.
>
> **Interpretation Guidance:**
> - If `clip_fraction_positive` >> `clip_fraction_negative`: Policy is aggressively increasing probabilities
> - If `clip_fraction_negative` >> `clip_fraction_positive`: Policy is aggressively decreasing probabilities
> - If both are high and similar: Balanced PPO training with active clipping
> - If both are low: Small policy updates, possibly due to low learning rate or converged policy
>
> **Performance Note:** The three clip metrics (total, positive, negative) are batched into a single GPU-to-CPU transfer via `torch.stack()` to minimize synchronization overhead.
