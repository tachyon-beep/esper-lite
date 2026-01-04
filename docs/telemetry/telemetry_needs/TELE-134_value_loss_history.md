# Telemetry Record: [TELE-134] Value Loss History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-134` |
| **Name** | Value Loss History |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the value loss trending in a healthy direction over recent PPO updates?"

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
| **Units** | MSE loss (unitless) |
| **Range** | `[0.0, inf)` - non-negative |
| **Precision** | 3 decimal places for display |
| **Default** | Empty deque with maxlen=10 |

### Semantic Meaning

> Rolling history of the 10 most recent value loss values from PPO updates. Value loss measures how well the value function predicts actual returns, computed as:
>
> L_v = MSE(V(s), R) = (V(s) - R)^2
>
> With optional clipping:
> L_v = 0.5 * max((V(s) - R)^2, (V_clipped(s) - R)^2)
>
> Used for sparkline visualization and trend detection in the TUI. A decreasing trend indicates the value function is improving its predictions; an increasing trend may signal instability or poor learning.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Decreasing or stable trend | Value function converging |
| **Warning** | Sharply increasing trend | Value function may be destabilizing |
| **Critical** | N/A | No critical threshold defined (uses trend direction only) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after value loss computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._ppo_inner_loop()` |
| **Line(s)** | ~802-806 |

```python
# Value loss computation (with optional clipping)
if self.clip_value:
    values_clipped = valid_old_values + torch.clamp(
        values - valid_old_values, -self.value_clip, self.value_clip
    )
    value_loss_unclipped = (values - valid_returns) ** 2
    value_loss_clipped = (values_clipped - valid_returns) ** 2
    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
else:
    value_loss = F.mse_loss(values, valid_returns)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Accumulated in metrics dict, emitted via `emit_ppo_update()` | `simic/telemetry/emitters.py:785` |
| **2. Collection** | `PPOUpdatePayload.value_loss` field | `leyline/telemetry.py:618` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` appends to history | `karn/sanctum/aggregator.py:796` |
| **4. Delivery** | Written to `snapshot.tamiyo.value_loss_history` | `karn/sanctum/schema.py:946` |

```
[PPOAgent] --metrics["value_loss"]--> [emit_ppo_update()] --PPOUpdatePayload-->
[Aggregator._handle_ppo_update()] --append()--> [TamiyoState.value_loss_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `value_loss_history` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_loss_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 946 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Sparkline + trend indicator for value loss |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes value_loss during PPO inner loop
- [x] **Transport works** — Value emitted via `emit_ppo_update()` in PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.value_loss_history: deque[float]`
- [x] **Default is correct** — Empty deque with maxlen=10
- [x] **Consumer reads it** — PPOLossesPanel reads and renders sparkline
- [x] **Display is correct** — Sparkline + trend arrow rendered
- [x] **Thresholds applied** — Trend direction determines styling (up=bad for loss)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `TestTamiyoState.test_history_deques` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_history` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | Uses `value_loss_history` in fixture | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPOLossesPanel "V.Loss" row
4. Verify sparkline appears after first PPO update
5. Watch sparkline update and trend arrow change as training progresses

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after PPO_UPDATE_COMPLETED events |
| `value_loss` scalar | telemetry | Each update appends current value_loss to history |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| PPOLossesPanel sparkline | display | Renders visual trend of last 10 values |
| Trend detection | display | `detect_trend()` determines up/down/stable arrow |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** The history deque has maxlen=10 to provide a short rolling window suitable for sparkline visualization (5-character width). This balances visual density with trend detection accuracy.
>
> **Related Metrics:** This is the history version of `TELE-130 value_loss` (scalar). The scalar holds the current value while the history holds the rolling window for trend visualization.
>
> **Trend Interpretation:** For loss metrics, decreasing trends are healthy (shown with down arrow in green), while increasing trends are concerning (shown with up arrow in red). This is inverted from reward metrics.
