# Telemetry Record: [TELE-664] Reward Holding Warning

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-664` |
| **Name** | Reward Holding Warning |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "Is a seed in HOLDING stage being held too long, suggesting it should proceed to fossilization or be pruned?"

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
| **Units** | reward units (penalty) |
| **Range** | `[-inf, 0]` — always negative or zero (penalty) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Holding warning is a penalty signal applied during the HOLDING stage when the agent continues to WAIT instead of advancing the seed. This encourages timely integration and prevents seeds from being held indefinitely.
>
> The HOLDING stage is a decision point: the seed can proceed to fossilization or be pruned. Holding for too long incurs an escalating penalty that nudges the agent toward making a decision.
>
> - **Zero:** Seed not in HOLDING, or just entered HOLDING (healthy)
> - **Negative:** Seed in HOLDING with continued WAIT actions (warning)
> - **Larger magnitude:** Longer holding period, stronger nudge toward action
>
> The penalty is calculated as a per-epoch cost that accumulates during HOLDING.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == 0` | No holding issues |
| **Warning** | `-0.05 < value < 0` | Minor holding delay |
| **Critical** | `value <= -0.05` | Seed held too long, needs decision |

**Display Color Logic:** Yellow styling (warning category)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Holding warning computation in contribution reward |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 682-707 |

```python
holding_warning = 0.0
...
# Escalating penalty for WAITing in HOLDING
if (
    seed_info.total_improvement is not None
    ...
):
    ...
    holding_warning = -per_epoch_penalty
    reward += holding_warning

components.holding_warning = holding_warning
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.holding_warning` | `simic/rewards/reward_telemetry.py` (line 34) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (implicit via component mapping) |
| **4. Delivery** | Written to `env.reward_components.holding_warning` | `karn/sanctum/schema.py` (line 1125) |

```
[compute_contribution_reward()]
  --components.holding_warning-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.holding_warning]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `holding_warning` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.holding_warning` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1125 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 641-645) | Displayed in "Warnings" row as "Hold: -0.03" with yellow styling |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.holding_warning` at line 707
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.holding_warning: float = 0.0` at line 1125
- [x] **Default is correct** — `0.0` is appropriate default (no warning when not triggered)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.holding_warning` at line 641
- [x] **Display is correct** — Rendered with `.3f` format, yellow styling, check for `< 0`

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Holding warning tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Holding warning rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for an environment with a seed in HOLDING stage
4. If the agent continues to WAIT, observe "Warnings" row — should show "Hold: -X.XXX" with yellow styling
5. Verify the warning appears only for HOLDING seeds with continued WAIT actions
6. After training, query telemetry: `SELECT holding_warning FROM rewards WHERE holding_warning != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed stage | state | Only applies to HOLDING stage |
| Action taken | action | Warning triggered when WAIT during HOLDING |
| Epochs in HOLDING | duration | Penalty may escalate with duration |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Warnings row | display | Displayed as "Hold: -X.XXX" |
| Total reward | derived | Contributes to total reward sum |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Holding warning encourages timely decision-making. The HOLDING stage is meant to be a brief evaluation period before fossilization, not an indefinite waiting room. The escalating penalty prevents the agent from exploiting HOLDING as a "safe" state.
>
> **Complementary to PBRS:** Together with the stage bonus (PBRS), holding warning creates a push-pull dynamic: the agent is rewarded for advancing to HOLDING (PBRS) but penalized for staying there too long (holding warning).
>
> **Botanical Metaphor:** In the plant metaphor, HOLDING is like a seedling that's ready to be planted but hasn't been yet. Keeping it in a pot too long is inefficient — it should either be planted (fossilized) or discarded (pruned).
>
> **Sign Convention:** Always negative or zero. A non-zero value always represents a penalty (never a bonus). The display checks `rc.holding_warning < 0` to determine visibility.
