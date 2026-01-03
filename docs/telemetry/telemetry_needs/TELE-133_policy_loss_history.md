# Telemetry Record: [TELE-133] Policy Loss History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-133` |
| **Name** | Policy Loss History |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the policy loss trending in a healthy direction over recent PPO updates?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
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
| **Units** | Clipped surrogate loss (dimensionless) |
| **Range** | Typically `(-0.5, 0.5)`, depends on advantage scaling |
| **Precision** | 3 decimal places for display |
| **Default** | Empty deque with `maxlen=10` |

### Semantic Meaning

> Rolling history of the last 10 policy loss values from PPO updates. Used for sparkline visualization and trend detection.
>
> Policy loss in PPO is the clipped surrogate objective:
> L_CLIP = E[min(r_t * A_t, clip(r_t, 1-e, 1+e) * A_t)]
>
> Where r_t is the probability ratio and A_t is the advantage estimate. The loss is negated (we minimize -objective). Values typically oscillate around a stable point during healthy training.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Stable or decreasing trend | Policy learning normally |
| **Warning** | Rapid increase over 5+ values | Potential instability |
| **Critical** | Sustained high values with NaN | Training collapse |

**Note:** Unlike scalar metrics, this is a history buffer. Trend detection uses the `detect_trend()` function which compares first-half to second-half averages.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed from clipped surrogate loss per action head |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._ppo_update()` inner loop |
| **Line(s)** | ~776-792 |

```python
# Compute policy loss per head and sum
policy_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
for key in HEAD_NAMES:
    ratio = per_head_ratios[key]
    adv = per_head_advantages[key]
    mask = head_masks[key]

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
    clipped_surr = torch.min(surr1, surr2)

    n_valid = mask.sum().clamp(min=1)
    head_loss = -(clipped_surr * mask).sum() / n_valid
    policy_loss = policy_loss + head_loss
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` packages `policy_loss` in `PPOUpdatePayload` | `simic/telemetry/emitters.py:784` |
| **2. Collection** | `TelemetryEvent` with `PPO_UPDATE_COMPLETED` type | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` appends to history | `karn/sanctum/aggregator.py:792` |
| **4. Delivery** | Written to `TamiyoState.policy_loss_history` deque | `karn/sanctum/schema.py:945` |

```
[PPOAgent] --emit_ppo_update()--> [TelemetryEmitter] --PPOUpdatePayload--> [Aggregator.handle_ppo_update()] --> [TamiyoState.policy_loss_history.append()]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `policy_loss_history` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.policy_loss_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 945 |

```python
# History for trend sparklines (last 10 values)
policy_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/ppo_losses_panel.py:213-218` | Sparkline visualization with trend arrow |

```python
# Policy loss with sparkline (compact for narrow panel)
result.append("P.Loss ", style="dim")
if tamiyo.policy_loss_history:
    pl_sparkline = render_sparkline(tamiyo.policy_loss_history, width=5)
    pl_trend = detect_trend(list(tamiyo.policy_loss_history))
    result.append(pl_sparkline)
    result.append(f" {tamiyo.policy_loss:.3f}", style="bright_cyan")
    result.append(pl_trend, style=trend_style(pl_trend, "loss"))
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - PPOAgent computes policy_loss and emits via `emit_ppo_update()`
- [x] **Transport works** - `PPOUpdatePayload.policy_loss` field populated at line 784
- [x] **Schema field exists** - `TamiyoState.policy_loss_history: deque[float]` at line 945
- [x] **Default is correct** - Empty deque with `maxlen=10` appropriate before first update
- [x] **Consumer reads it** - PPOLossesPanel accesses and renders sparkline at lines 213-218
- [x] **Display is correct** - Value renders with sparkline characters and trend arrow
- [x] **Thresholds applied** - Trend styling uses `trend_style(pl_trend, "loss")` (red=increasing, green=decreasing)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_history_deques` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_all_history_deques` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | `test_ppo_losses_shows_sparklines` | `[x]` |
| Isolation (snapshot copy) | `tests/karn/sanctum/test_aggregator.py` | `test_get_snapshot_returns_isolated_copy` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPOLossesPanel in the Tamiyo Brain section
4. Verify "P.Loss" row shows sparkline characters after 2+ PPO updates
5. Verify trend arrow appears and changes color based on loss direction
6. Confirm sparkline updates with each PPO update

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| `TELE-???` policy_loss (scalar) | telemetry | Each update's scalar value is appended to history |
| Per-head advantages | computation | Required for clipped surrogate loss |
| Probability ratios | computation | Required for policy gradient |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| PPOLossesPanel sparkline | display | Visual trend representation |
| `detect_trend()` output | derived | Trend arrow direction (up/down/stable) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation during TELE-133 audit |

---

## 8. Notes

> **Design Decision:** The deque has `maxlen=10` to provide enough history for meaningful trend detection while keeping memory bounded. The sparkline width in PPOLossesPanel is set to 5 characters for compactness.
>
> **Implementation Detail:** The history is populated in `handle_ppo_update()` on the aggregator side, not the emitter side. This keeps the emitter lightweight and allows the aggregator to manage state.
>
> **Sparkline Rendering:** Uses `render_sparkline()` from `sparkline_utils.py` which maps values to Unicode block characters. Values are normalized within the visible window, so the sparkline shows relative change not absolute scale.
>
> **Trend Detection:** The `detect_trend()` function compares first-half average to second-half average. For losses, a decreasing trend is considered healthy (green), increasing is concerning (red).
>
> **Related History Fields:** This is one of 7 parallel history deques in TamiyoState (policy_loss, value_loss, grad_norm, entropy, explained_variance, kl_divergence, clip_fraction), all populated by the same `handle_ppo_update()` handler.
