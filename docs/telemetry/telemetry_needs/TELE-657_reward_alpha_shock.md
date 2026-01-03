# Telemetry Record: [TELE-657] Reward Alpha Shock

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-657` |
| **Name** | Reward Alpha Shock |
| **Category** | `reward` |
| **Priority** | `P1-critical` |

## 2. Purpose

### What question does this answer?

> "Is the agent gaming the system by rapidly oscillating alpha values to exploit reward mechanics?"

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
| **Units** | reward units (penalty) |
| **Range** | `[-inf, 0]` — always negative or zero (penalty) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Alpha shock is a convex penalty applied when the agent makes large alpha (blending coefficient) changes. This is an anti-gaming mechanism that discourages reward hacking through rapid alpha oscillation.
>
> The penalty is computed as: `-alpha_shock_coef * sum(alpha_delta^2)` where alpha_delta is the change in blending coefficients. The quadratic form makes large changes increasingly expensive.
>
> - **Zero:** No alpha changes or changes below threshold (healthy)
> - **Negative:** Penalty applied for large alpha oscillations (gaming detected)
> - **Larger magnitude:** More aggressive alpha manipulation
>
> When triggered, the UI displays "Gaming: X% (SHOCK)" to alert operators that anti-gaming measures are active.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == 0` | No gaming detected |
| **Warning** | `-0.05 < value < 0` | Minor alpha oscillation detected |
| **Critical** | `value <= -0.05` | Significant gaming behavior, may need intervention |

**Display Color Logic:** Red styling when triggered (penalty), contributes to "Gaming" indicator

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Alpha shock computation in contribution reward |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 753-758 |

```python
alpha_shock = 0.0
if alpha_delta_sq_sum > 0 and config.alpha_shock_coef != 0.0 and not config.disable_anti_gaming:
    alpha_shock = -config.alpha_shock_coef * alpha_delta_sq_sum
    reward += alpha_shock

    components.alpha_shock = alpha_shock
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.alpha_shock` | `simic/rewards/reward_telemetry.py` (line 32) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1467) |
| **4. Delivery** | Written to `env.reward_components.alpha_shock` | `karn/sanctum/schema.py` (line 1116) |

```
[compute_contribution_reward()]
  --components.alpha_shock-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.alpha_shock]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `alpha_shock` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.alpha_shock` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1116 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 579-583) | Displayed in "Signals" row as "Shock: -0.123" with red styling |
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 594-598) | Triggers "Gaming: X% (SHOCK)" indicator when non-zero |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.alpha_shock` at line 758
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.alpha_shock: float = 0.0` at line 1116
- [x] **Default is correct** — `0.0` is appropriate default (no penalty when not triggered)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.alpha_shock` at lines 579 and 594
- [x] **Display is correct** — Rendered with `.3f` format, red styling, triggers SHOCK indicator

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Alpha shock penalty tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Gaming indicator rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for any environment
4. Observe "Signals" row — if gaming detected, shows "Shock: -X.XXX" with red styling
5. Observe gaming indicator — shows "Gaming: X% (SHOCK)" when alpha_shock is non-zero
6. After training, query telemetry: `SELECT alpha_shock FROM rewards WHERE alpha_shock != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Alpha delta computation | state | Requires tracking of blending coefficient changes |
| Anti-gaming config | config | Controlled by `alpha_shock_coef` (default 0.1958) |
| `disable_anti_gaming` flag | config | Can be disabled for debugging |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Signals row | display | Displayed as "Shock: -X.XXX" |
| EnvDetailScreen gaming indicator | display | Triggers "(SHOCK)" state in gaming display |
| Gaming rate calculation | derived | Contributes to `gaming_trigger_count` when non-zero |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Alpha shock uses a convex (quadratic) penalty to make large alpha changes increasingly expensive. This allows small adjustments while severely penalizing oscillation patterns that could exploit reward mechanics.
>
> **Anti-Gaming Mechanism:** Together with `ratio_penalty`, alpha shock forms the anti-gaming defense. Alpha shock targets alpha manipulation, while ratio penalty targets contribution inflation. Both are displayed in the "Signals" row and contribute to the gaming rate.
>
> **Coefficient Tuning:** The default `alpha_shock_coef = 0.1958` was tuned empirically. The value balances sensitivity (detecting gaming) with avoiding false positives on legitimate alpha adjustments during blending.
>
> **Sign Convention:** Always negative or zero. A non-zero value always represents a penalty (never a bonus).
