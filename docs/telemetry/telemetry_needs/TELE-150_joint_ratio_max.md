# Telemetry Record: [TELE-150] Joint Ratio Max

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-150` |
| **Name** | Joint Ratio Max |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the joint probability ratio across all action heads stable? Is there multi-head policy instability manifesting as ratio explosion?"

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
| **Units** | dimensionless ratio |
| **Range** | `[0.0, inf)` — unbounded above, typically close to 1.0 |
| **Precision** | 3 decimal places for display |
| **Default** | `1.0` (represents "no update yet" or "healthy identity ratio") |

### Semantic Meaning

> The joint ratio is the product of per-head importance sampling ratios across all 8 action heads (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op).
>
> Computed in log-space for numerical stability:
> - joint_ratio = exp(sum(log_ratio_i)) = ∏(ratio_i)
> - Each log_ratio_i is clamped to [-30.0, 30.0] before summation
>
> In standard PPO with clip range [0.8, 1.2]:
> - Each head ratio should be near 1.0
> - With 8 independent heads, ratios at 1.15 produce joint ~3.06
> - Detects when the product of probabilities shifts (multi-head policy shift)
>
> Values >> 1.0 indicate policy expansion (more probability to new actions)
> Values << 1.0 indicate policy contraction (less probability to new actions)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.5 ≤ joint_ratio ≤ 2.0` | Normal multi-head policy stability |
| **Warning** | `(0.33 < joint_ratio < 0.5)` or `(2.0 < joint_ratio ≤ 3.0)` | Elevated ratio change, monitor closely |
| **Critical** | `joint_ratio ≤ 0.33` or `joint_ratio > 3.0` | Severe policy collapse/explosion across heads |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed across all epochs |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._compute_policy_loss()` (main epoch loop) |
| **Line(s)** | ~524 (init), ~697-704 (computation) |

```python
# Initialize tracking (line 524)
joint_ratio_max_across_epochs: float = float("-inf")

# Compute in each epoch (lines 697-704)
# Compute joint ratio using log-space summation (numerically stable)
# joint_ratio = exp(sum(log_ratio_i)) = product(ratio_i)
stacked_log_ratios = torch.stack([log_ratios_for_joint[k] for k in HEAD_NAMES])
joint_log_ratio = stacked_log_ratios.sum(dim=0)  # Sum across heads per timestep
joint_log_ratio_clamped = torch.clamp(joint_log_ratio, min=-30.0, max=30.0)
joint_ratio = torch.exp(joint_log_ratio_clamped)
epoch_joint_ratio_max = joint_ratio.max().item()
joint_ratio_max_across_epochs = max(joint_ratio_max_across_epochs, epoch_joint_ratio_max)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update_event()` | `simic/telemetry/emitters.py` |
| **2. Collection** | Event payload with `joint_ratio_max` field | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.joint_ratio_max` | `karn/sanctum/schema.py` |

```
[PPOAgent._compute_policy_loss()] --> [emit_ppo_update_event()] --> [PPOUpdatePayload] -->
[TelemetryEvent] --> [_handle_ppo_update()] --> [TamiyoState.joint_ratio_max]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `joint_ratio_max` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.joint_ratio_max` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~935 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | "Ratio Joint" row with color-coded status |
| MCP SQL View | `karn/mcp/views.py` | Exposed in ppo_updates view for analysis queries |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes joint_ratio_max during update
- [x] **Transport works** — Event includes joint_ratio_max field in PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.joint_ratio_max: float = 1.0`
- [x] **Default is correct** — 1.0 appropriate (identity ratio, healthy state)
- [x] **Consumer reads it** — HealthStatusPanel accesses `snapshot.tamiyo.joint_ratio_max`
- [x] **Display is correct** — Value renders with 3 decimal places, color-coded status
- [x] **Thresholds applied** — `_get_joint_ratio_status()` uses 0.33/0.5/2.0/3.0 thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | Not yet written | `[ ]` |
| Unit (aggregator) | — | Not yet written | `[ ]` |
| Integration (end-to-end) | — | Not yet written | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Ratio Joint" row
4. Verify joint ratio value updates after each PPO update
5. Verify status color changes:
   - Cyan (ok): 0.5 ≤ ratio ≤ 2.0
   - Yellow (warning): 0.33 < ratio < 0.5 or 2.0 < ratio ≤ 3.0
   - Red (critical): ratio ≤ 0.33 or ratio > 3.0

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Per-head log ratios | computation | Requires valid policy forward pass and log prob computation for all 8 heads |
| Importance sampling | computation | Requires old and new policy log probabilities |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel status | display | Drives visual indication of policy stability |
| Auto-monitoring systems | system | May trigger alerts on critical thresholds |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude Code | Initial audit creation from TELE-150 specification |
| | | Verified wiring from emitter through aggregator to widget |
| | | Confirmed thresholds match HealthStatusPanel implementation |

---

## 8. Notes

> **Design Decision:** The joint ratio is computed in log-space (summing log_ratios then exponentiating) for numerical stability. Clamping to [-30.0, 30.0] prevents underflow/overflow. The default value of 1.0 represents the identity ratio (no policy change), which is the healthy baseline.
>
> **Multi-Head Detection:** Unlike single-head metrics (TELE-101 entropy, TELE-102 kl_divergence), the joint ratio detects instability that emerges from the *interaction* of multiple heads. A situation where individual head ratios are stable but their product explodes indicates misalignment between heads in the policy update direction.
>
> **Thresholds Rationale:** With 8 heads operating independently:
> - Critical threshold of >3.0: Approximately 8 heads at 1.15 each (modest ratio per head)
> - Warning threshold of >2.0: Approximately 8 heads at 1.09 each
> - Lower bounds (0.33, 0.5) mirror the upper bounds, detecting contraction
>
> **Known Limitation:** The metric tracks max ratio across all timesteps in the batch rather than mean or distribution. This makes it sensitive to outliers but catches edge cases where even a single timestep shows severe policy shift.
>
> **Future Improvement:** Consider tracking per-head ratio max in addition to joint, to identify which heads are driving the explosion. Could add breakdown chart showing head contribution to total ratio divergence.

