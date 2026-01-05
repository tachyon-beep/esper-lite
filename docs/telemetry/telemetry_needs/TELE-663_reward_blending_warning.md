# Telemetry Record: [TELE-663] Reward Blending Warning

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-663` |
| **Name** | Reward Blending Warning |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "Is a seed in BLENDING stage showing negative trajectory, suggesting it should be pruned rather than fossilized?"

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

> Blending warning is a penalty signal applied during the BLENDING stage when a seed shows negative total improvement. This nudges the agent toward pruning rather than fossilizing underperforming seeds.
>
> When a seed enters BLENDING, it is close to fossilization. If its total_improvement is negative at this point, the agent should reconsider — the warning penalty escalates with the magnitude of degradation.
>
> - **Zero:** Seed not in BLENDING, or BLENDING with positive trajectory (healthy)
> - **Negative:** Seed in BLENDING with negative trajectory (warning)
> - **Larger magnitude:** More severe degradation, stronger nudge toward pruning
>
> The penalty is calculated as: `-0.1 - escalation` where escalation increases with degradation magnitude.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == 0` | No blending issues |
| **Warning** | `-0.1 < value < 0` | Minor blending concern |
| **Critical** | `value <= -0.1` | Seed should likely be pruned |

**Display Color Logic:** Yellow styling (warning category)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Blending warning computation in contribution reward |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 657-667 |

```python
blending_warning = 0.0
...
total_imp = seed_info.total_improvement
if total_imp < 0:
    # Escalating penalty for negative trajectory during BLENDING
    escalation = min(abs(total_imp) * 0.1, 0.1)
    blending_warning = -0.1 - escalation
    reward += blending_warning

components.blending_warning = blending_warning
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.blending_warning` | `simic/rewards/reward_telemetry.py` (line 33) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (implicit via component mapping) |
| **4. Delivery** | Written to `env.reward_components.blending_warning` | `karn/sanctum/schema.py` (line 1124) |

```
[compute_contribution_reward()]
  --components.blending_warning-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.blending_warning]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `blending_warning` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.blending_warning` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1124 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 638-640) | Displayed in "Warnings" row as "Blend: -0.02" with yellow styling |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.blending_warning` at line 667
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.blending_warning: float = 0.0` at line 1124
- [x] **Default is correct** — `0.0` is appropriate default (no warning when not triggered)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.blending_warning` at line 638
- [x] **Display is correct** — Rendered with `.3f` format, yellow styling, check for `< 0`

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Blending warning tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Blending warning rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for an environment with a seed in BLENDING stage
4. If the seed has negative trajectory, observe "Warnings" row — should show "Blend: -X.XXX" with yellow styling
5. Verify the warning appears only for BLENDING seeds with negative total_improvement
6. After training, query telemetry: `SELECT blending_warning FROM rewards WHERE blending_warning != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed stage | state | Only applies to BLENDING stage |
| total_improvement | metric | Warning triggered when negative |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Warnings row | display | Displayed as "Blend: -X.XXX" |
| Total reward | derived | Contributes to total reward sum |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Blending warning is a "soft gate" before fossilization. Unlike hard pruning, it nudges the agent toward reconsidering without forcing the decision. The escalating penalty structure makes larger degradations progressively more costly.
>
> **Stage-Specific:** This warning only applies to the BLENDING stage. Earlier stages (TRAINING) don't receive this penalty because it's expected that seeds may have temporary negative trajectories during training.
>
> **Botanical Metaphor:** In the plant metaphor, BLENDING is like a seedling that's almost ready to be permanently grafted. A negative trajectory at this stage is a warning sign that the graft may fail.
>
> **Sign Convention:** Always negative or zero. A non-zero value always represents a penalty (never a bonus). The display checks `rc.blending_warning < 0` to determine visibility.
