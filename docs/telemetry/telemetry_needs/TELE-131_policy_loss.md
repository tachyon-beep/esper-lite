# Telemetry Record: [TELE-131] Policy Loss

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-131` |
| **Name** | Policy Loss |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "What is the current magnitude of the policy gradient loss? Is the policy learning signal healthy or degenerating?"

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
| **Units** | loss units (dimensionless) |
| **Range** | `(-inf, inf)` — typically small positive values |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> The PPO policy loss (also called surrogate objective loss) measures the clipped surrogate objective used in Proximal Policy Optimization. Computed as:
>
> L^{CLIP}(theta) = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
>
> Where r_t is the probability ratio between new and old policy, A_t is the advantage estimate, and eps is the clipping parameter.
>
> The policy loss is summed across all action heads (slot, blueprint, style, tempo, op) using masked means to avoid bias from masked positions.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Stable, small positive values | Normal learning signal |
| **Warning** | Large spikes or oscillation | Potential training instability |
| **Critical** | NaN or extremely large | Training failure |

**Note:** Policy loss does not have fixed thresholds in `TUIThresholds`. The Value/Policy loss ratio is monitored instead (healthy: 0.2-5.0, warning: 0.1-0.2 or 5.0-10.0, critical: <0.1 or >10.0).

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed as sum of clipped surrogate losses across heads |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | ~778-792, ~899, ~914 |

```python
# Compute policy loss per head and sum
# Use masked mean to avoid bias from averaging zeros with real values
policy_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
for key in HEAD_NAMES:
    ratio = per_head_ratios[key]
    adv = per_head_advantages[key]
    mask = head_masks[key]
    # ... clipping logic ...
    # Masked mean: only average over causally-relevant positions
    n_valid = mask.sum().clamp(min=1)
    head_loss = -(clipped_surr * mask).sum() / n_valid
    policy_loss = policy_loss + head_loss

# Later in the function:
metrics["policy_loss"].append(logging_tensors[0])
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update_completed()` with `metrics["policy_loss"]` | `simic/telemetry/emitters.py` |
| **2. Collection** | `PPOUpdatePayload.policy_loss` field | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `TamiyoState.policy_loss` and appended to `policy_loss_history` | `karn/sanctum/schema.py` |

```
[PPOAgent.update()] --> [emit_ppo_update_completed()] --> [PPOUpdatePayload] --> [Aggregator.handle_ppo_update()] --> [TamiyoState.policy_loss]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `policy_loss` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.policy_loss` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 839 |

Additional related fields:
- `policy_loss_history: deque[float]` at line 945 (last 10 values for sparkline)
- `TREND_THRESHOLDS["policy_loss"] = 0.20` at line 1406 (20% threshold for trend detection)

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/ppo_losses_panel.py` | Displayed with sparkline, value, and trend indicator (lines 211-221) |
| PPOLossesPanel | Same file | Used in Value/Policy loss ratio calculation (lines 240-273) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes policy_loss and adds to metrics dict
- [x] **Transport works** — `emit_ppo_update_completed()` includes `policy_loss=metrics["policy_loss"]` in `PPOUpdatePayload`
- [x] **Schema field exists** — `TamiyoState.policy_loss: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — PPOLossesPanel accesses `tamiyo.policy_loss` and `tamiyo.policy_loss_history`
- [x] **Display is correct** — Value renders with 3 decimal places, sparkline, and trend arrow
- [x] **Thresholds applied** — Value/Policy ratio uses thresholds (0.1-10.0 healthy range)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_policy_loss_computed` (implicit in metrics assertions) | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_histories` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | `test_ppo_losses_panel_renders_sparklines` | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPO LOSSES panel "P.Loss" row
4. Verify policy_loss value updates after each PPO batch
5. Check sparkline shows history of last 10 values
6. Verify trend arrow appears (up/down/stable)
7. Check Lv/Lp ratio row shows Value/Policy ratio with health status

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Per-head ratio computation | computation | Requires valid policy forward pass for all heads |
| Advantage estimates | computation | Uses normalized advantages from GAE |
| Head masks | computation | Causal masks for multi-head action space |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `policy_loss_history` | derived | Deque of last 10 values for sparkline |
| Value/Policy loss ratio | display | PPOLossesPanel computes `value_loss / policy_loss` |
| Trend detection | display | Uses 20% threshold from `TREND_THRESHOLDS` |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-06-15 | Initial | Created with PPO implementation |
| 2024-09-20 | Refactor | Moved to TamiyoState from flat snapshot |
| 2025-01-03 | Audit | Created telemetry record, verified full wiring |

---

## 8. Notes

> **Design Decision:** Policy loss is computed as a sum across all action heads (slot, blueprint, style, tempo, op) rather than averaged. Each head uses masked mean to avoid bias from masked positions.
>
> **Trend Threshold:** Policy loss uses a 20% threshold for trend detection (`TREND_THRESHOLDS["policy_loss"] = 0.20`), which is relatively high because policy loss is noisy.
>
> **Value/Policy Ratio:** The PPOLossesPanel displays the Lv/Lp (value/policy loss) ratio as a key diagnostic. Healthy range is 0.2-5.0; values outside this indicate one loss dominating the other, which can cause training instability.
>
> **No Direct Thresholds:** Unlike entropy or clip_fraction, policy_loss does not have fixed critical/warning thresholds because the absolute magnitude depends on advantage scaling and learning rate. The ratio to value_loss is the more meaningful diagnostic.
