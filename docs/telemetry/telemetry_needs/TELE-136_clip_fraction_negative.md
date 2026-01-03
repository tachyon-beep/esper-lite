# Telemetry Record: [TELE-136] Clip Fraction Negative

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-136` |
| **Name** | Clip Fraction Negative |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How often is PPO clipping preventing the policy from further decreasing action probabilities? Are actions being discouraged hitting the clip floor?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `float` |
| **Units** | fraction (0-1) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 3 decimal places for computation, 1 decimal as percentage for display |
| **Default** | `0.0` |

### Semantic Meaning

> Clip fraction negative measures what proportion of policy updates had their probability ratio clipped because the ratio fell below the PPO clip floor (1 - epsilon).
>
> Computed as: `clip_neg = (joint_ratio < 1.0 - clip_ratio).float().mean()`
>
> Where:
> - `joint_ratio` = product of probability ratios across all action heads (pi_new/pi_old)
> - `clip_ratio` = PPO epsilon parameter (typically 0.2)
>
> A ratio < 1 - epsilon means the new policy assigns significantly LOWER probability to the action than the old policy. PPO clipping prevents the update from pushing the probability even lower.
>
> **Interpretation:**
> - High clip_neg = policy is trying to strongly discourage many actions (hitting the floor)
> - Asymmetry with clip_pos indicates directional policy drift
> - Symmetric high values (both pos and neg elevated) indicate normal learning with large updates

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `clip_neg < 0.15` | Normal clipping behavior |
| **Warning** | `0.15 <= clip_neg < 0.30` | Elevated negative clipping, monitor asymmetry |
| **Critical** | `clip_neg >= 0.30` | Policy may be aggressively unlearning actions |

**Note:** No explicit thresholds defined in `TUIThresholds` for directional clip fractions. The overall `CLIP_WARNING = 0.25` applies to total clip_fraction. Asymmetry detection is more important than absolute values.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, during policy loss computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` inner loop |
| **Line(s)** | 750-759 |

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
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `src/esper/simic/telemetry/emitters.py:865` |
| **2. Collection** | `PPOUpdatePayload` dataclass with field | `src/esper/leyline/telemetry.py:748` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `src/esper/karn/sanctum/aggregator.py:974` |
| **4. Delivery** | Written to `TamiyoState.gradient_quality.clip_fraction_negative` | `src/esper/karn/sanctum/schema.py:785` |

```
[PPOAgent.update()]
    --> metrics["clip_fraction_negative"].append(clip_neg)
    --> TelemetryEmitter.emit_ppo_update(metrics)
    --> PPOUpdatePayload(clip_fraction_negative=...)
    --> SanctumAggregator._handle_ppo_update()
    --> TamiyoState.gradient_quality.clip_fraction_negative
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `GradientQualityMetrics` (nested in `TamiyoState`) |
| **Field** | `clip_fraction_negative` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.gradient_quality.clip_fraction_negative` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 785 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/ppo_losses_panel.py:146-149` | Displayed as directional breakdown with down-arrow indicator |

**Display Format:**
```python
clip_neg = tamiyo.gradient_quality.clip_fraction_negative
# Style: dim when both zero, otherwise show direction that's active
dir_style = "dim" if clip_pos == 0 and clip_neg == 0 else "cyan"
result.append(f" (up_arrow{clip_pos:.1%} down_arrow{clip_neg:.1%})", style=dir_style)
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - PPOAgent computes clip_neg and adds to metrics dict
- [x] **Transport works** - TelemetryEmitter includes field in PPOUpdatePayload
- [x] **Schema field exists** - `GradientQualityMetrics.clip_fraction_negative: float = 0.0`
- [x] **Default is correct** - 0.0 appropriate before first PPO update
- [x] **Consumer reads it** - PPOLossesPanel accesses `gradient_quality.clip_fraction_negative`
- [x] **Display is correct** - Rendered as percentage with down-arrow indicator
- [ ] **Thresholds applied** - No explicit color coding for directional clip fractions (uses dim/cyan styling)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (payload) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_gradient_quality_fields` | `[x]` |
| Unit (payload defaults) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_defaults_gradient_quality` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_gradient_quality_metrics_defaults` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_handle_ppo_update_populates_gradient_quality` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | `test_ppo_losses_panel_renders_directional_clip` | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPOLossesPanel "Clip Frac" row
4. Verify directional breakdown shows "(up_arrow X.X% down_arrow Y.Y%)" after clip fraction
5. During early training, expect both values to be low (dim styling)
6. As training progresses, observe non-zero values appear (cyan styling)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| `joint_ratio` computation | value | Requires product of probability ratios across all action heads |
| `clip_ratio` hyperparameter | config | Typically 0.2, determines clip threshold |
| `TELE-135 clip_fraction_positive` | telemetry | Computed together in same code block |
| `TELE-130 clip_fraction` | telemetry | Total clip fraction computed alongside |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| PPOLossesPanel display | widget | Uses for directional breakdown display |
| Asymmetry detection (implicit) | analysis | Difference from clip_fraction_positive indicates drift direction |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record created during audit |

---

## 8. Notes

> **Design Decision:** Directional clip fractions were added per DRL expert recommendation to track WHERE clipping occurs, not just WHETHER it occurs. Asymmetry between positive and negative clipping indicates directional policy drift.
>
> **Interpretation Guide:**
> - `clip_pos >> clip_neg`: Policy is increasingly confident in certain actions (probability increases being capped)
> - `clip_neg >> clip_pos`: Policy is increasingly discouraging certain actions (probability decreases being capped)
> - `clip_pos ~ clip_neg`: Balanced exploration/exploitation, normal healthy training
>
> **Performance Note:** All three clip metrics (total, positive, negative) are batched into a single GPU->CPU transfer via `torch.stack()` to minimize synchronization overhead.
>
> **MCP View Access:** Available in `ppo_updates` view via `clip_fraction_negative` column for SQL queries.
