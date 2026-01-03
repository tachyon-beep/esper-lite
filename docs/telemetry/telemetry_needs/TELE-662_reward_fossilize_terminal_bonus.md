# Telemetry Record: [TELE-662] Reward Fossilize Terminal Bonus

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-662` |
| **Name** | Reward Fossilize Terminal Bonus |
| **Category** | `reward` |
| **Priority** | `P1-critical` |

## 2. Purpose

### What question does this answer?

> "How much terminal bonus was awarded for successful fossilization(s) this step?"

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
| **Units** | reward units (terminal bonus) |
| **Range** | `[0, +inf]` — always positive or zero (bonus) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Fossilize terminal bonus is a large reward signal given when a seed successfully completes its lifecycle by fossilizing (permanently integrating with the host model).
>
> This is a terminal reward that incentivizes the agent to guide seeds to successful completion. The bonus is proportional to the number of contributing fossilized seeds (seeds with `total_improvement >= MIN_FOSSILIZE_CONTRIBUTION`).
>
> - **Zero:** No fossilization occurred this step (normal operation)
> - **Positive:** One or more seeds successfully fossilized
> - **Higher values:** Multiple successful fossilizations or high-value fossilizations
>
> The bonus is calculated as: `num_contributing_fossilized * fossilize_terminal_scale`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Normal (any value is valid) |
| **Note** | `value > 0` | Successful fossilization occurred |
| **Info** | `value > 1.0` | Multiple or high-value fossilizations |

**Display Color Logic:** Blue styling (bonus category)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Terminal bonus computation in contribution reward |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 796-809 |

```python
fossilize_terminal_bonus = 0.0
...
# Seeds with total_improvement < DEFAULT_MIN_FOSSILIZE_CONTRIBUTION get no terminal bonus.
...
fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
terminal_bonus += fossilize_terminal_bonus
...
components.fossilize_terminal_bonus = fossilize_terminal_bonus
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.fossilize_terminal_bonus` | `simic/rewards/reward_telemetry.py` (line 42) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (implicit via component mapping) |
| **4. Delivery** | Written to `env.reward_components.fossilize_terminal_bonus` | `karn/sanctum/schema.py` (line 1121) |

```
[compute_contribution_reward()]
  --components.fossilize_terminal_bonus-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.fossilize_terminal_bonus]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `fossilize_terminal_bonus` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.fossilize_terminal_bonus` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1121 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 626-630) | Displayed in "Credits" row as "Foss: +0.50" with blue styling |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.fossilize_terminal_bonus` at line 809
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.fossilize_terminal_bonus: float = 0.0` at line 1121
- [x] **Default is correct** — `0.0` is appropriate default (no bonus when no fossilization)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.fossilize_terminal_bonus` at line 626
- [x] **Display is correct** — Rendered with `+.3f` format, blue styling

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Fossilize terminal bonus tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Fossilize bonus rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for an environment where fossilization occurs
4. Observe "Credits" row — should show "Foss: +X.XXX" when fossilization bonus is awarded
5. Verify the bonus appears at the moment of fossilization
6. After training, query telemetry: `SELECT fossilize_terminal_bonus FROM rewards WHERE fossilize_terminal_bonus != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Fossilization events | event | Bonus only awarded when seed fossilizes |
| num_contributing_fossilized | count | Number of seeds meeting contribution threshold |
| fossilize_terminal_scale | config | Multiplier for bonus magnitude |
| MIN_FOSSILIZE_CONTRIBUTION | threshold | Seeds below this don't get terminal bonus |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Credits row | display | Displayed as "Foss: +X.XXX" |
| Total reward | derived | Contributes to total reward sum |
| Terminal bonus tracking | aggregate | Part of terminal_bonus computation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** The fossilize terminal bonus is a critical incentive for the RL agent to complete seed lifecycles successfully. Without this terminal reward, the agent might keep seeds in training indefinitely without fossilizing them.
>
> **Quality Gate:** Seeds with `total_improvement < MIN_FOSSILIZE_CONTRIBUTION` (typically 0.5%) don't receive terminal bonus. This prevents rewarding fossilization of low-value or harmful seeds.
>
> **Botanical Metaphor:** In the plant lifecycle metaphor, fossilization is like a seed successfully growing into a tree and becoming part of the forest (host model). The terminal bonus rewards this successful completion.
>
> **Sign Convention:** Always positive or zero. A non-zero value always represents a bonus (never a penalty). Fossilization is always rewarded (assuming the seed met contribution thresholds).
