# Telemetry Record: [TELE-132] Value Loss

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-132` |
| **Name** | Value Loss |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "How well is the value function learning to predict expected returns? Is the critic providing useful baselines for advantage estimation?"

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
| **Units** | MSE loss (squared error) |
| **Range** | `[0.0, +inf)` — typically 0.01 to 10.0 during healthy training |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Value loss measures how accurately the critic (value function) predicts expected returns. Computed as Mean Squared Error between predicted values V(s) and computed returns G:
>
> L_value = 0.5 * MSE(V(s), G)
>
> With value clipping enabled (default), uses pessimistic max:
> ```
> V_clipped = V_old + clamp(V - V_old, -clip, +clip)
> L_value = 0.5 * max(MSE(V, G), MSE(V_clipped, G))
> ```
>
> Lower value loss indicates better value predictions, which improves advantage estimation quality. Very low value loss may indicate the value function is overfitting to training data.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.01 < value_loss < 5.0` | Normal operating range for RL training |
| **Warning** | `value_loss > 5.0` or `value_loss < 0.001` | Value function may be struggling or overfitting |
| **Critical** | `value_loss > 20.0` or `NaN` | Value function failure, training unstable |

**Note:** Value loss magnitude depends on reward scale. The Lv/Lp ratio (value/policy loss ratio) in the TUI uses thresholds: critical if `< 0.1` or `> 10.0`, warning if `< 0.2` or `> 5.0`.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, during value function loss computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._update_inner_loop()` |
| **Line(s)** | 802-806 |

```python
# With value clipping (default)
value_loss_unclipped = (values - valid_returns) ** 2
value_loss_clipped = (values_clipped - valid_returns) ** 2
value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

# Without value clipping
value_loss = F.mse_loss(values, valid_returns)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Collected in `metrics["value_loss"]` list during PPO epochs | `simic/agent/ppo.py:915` |
| **2. Collection** | `TelemetryEmitter.emit_ppo_update()` with `PPOUpdatePayload` | `simic/telemetry/emitters.py:785` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py:794-796` |
| **4. Delivery** | Written to `TamiyoState.value_loss` and appended to `value_loss_history` | `karn/sanctum/schema.py:840,946` |

```
[PPOAgent] --metrics dict--> [TelemetryEmitter.emit_ppo_update()] --PPOUpdatePayload-->
[SanctumAggregator._handle_ppo_update()] --> [TamiyoState.value_loss + value_loss_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `value_loss: float = 0.0` |
| **History Field** | `value_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_loss` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 840 (value_loss), 946 (value_loss_history) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py:223-233` | Value loss with sparkline and trend indicator |
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py:236,240-264` | Value/Policy loss ratio with threshold coloring |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes value_loss during update (lines 802-806)
- [x] **Transport works** — `emit_ppo_update()` includes value_loss in PPOUpdatePayload (line 785)
- [x] **Schema field exists** — `TamiyoState.value_loss: float = 0.0` (line 840)
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — PPOLossesPanel accesses `tamiyo.value_loss` and `tamiyo.value_loss_history`
- [x] **Display is correct** — Value renders with 3 decimal places, sparkline, and trend
- [x] **Thresholds applied** — Lv/Lp ratio uses 0.1/0.2/5.0/10.0 thresholds for coloring

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_ppo_update` (line 888) | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_history` (lines 34,48-52) | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_tamiyo_state_histories` (line 169,178) | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | Uses value_loss in fixtures (lines 42,46,90) | `[x]` |
| Integration (end-to-end) | `tests/integration/test_q_values_telemetry.py` | Emits value_loss (line 139) | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPOLossesPanel "V.Loss" row
4. Verify value_loss updates after each PPO batch with sparkline
5. Observe Lv/Lp ratio row for value/policy loss ratio
6. Verify ratio coloring: cyan for healthy (0.2-5.0), yellow for warning, red for critical

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Value predictions V(s) | computation | Requires valid critic forward pass |
| Computed returns G | computation | Requires GAE/TD returns calculation |
| TELE-131 policy_loss | telemetry | Used together for Lv/Lp ratio display |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Lv/Lp ratio display | widget | PPOLossesPanel computes ratio for health indication |
| Trend detection | display | Uses TREND_THRESHOLDS["value_loss"] = 0.20 (20% threshold) |
| `value_loss_history` | derived | Deque of last 10 values for sparkline |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation during TELE audit |

---

## 8. Notes

> **Design Decision:** Value loss uses the pessimistic max of clipped and unclipped loss when value clipping is enabled. This prevents the value function from changing too rapidly, similar to how policy clipping stabilizes policy updates.
>
> **Value Clipping:** The `value_clip` parameter (default 10.0) is separate from the policy `clip_ratio` (default 0.2) because value predictions can range from -10 to +50, making a 0.2 clip too restrictive for value function updates.
>
> **Lv/Lp Ratio:** The value/policy loss ratio is a useful diagnostic. Healthy training typically shows ratios between 0.2 and 5.0. Extreme ratios (< 0.1 or > 10.0) may indicate:
> - Very low ratio: Value function learning too slowly relative to policy
> - Very high ratio: Policy updates too small, or value function struggling
>
> **Trend Threshold:** TREND_THRESHOLDS["value_loss"] = 0.20 means 20% change is required before trend is flagged as improving/declining, accounting for natural batch-to-batch noise in value loss.
