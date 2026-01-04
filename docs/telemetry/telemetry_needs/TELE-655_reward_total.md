# Telemetry Record: [TELE-655] Reward Total

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-655` |
| **Name** | Reward Total |
| **Category** | `reward` |
| **Priority** | `P1-critical` |

## 2. Purpose

### What question does this answer?

> "What is the total combined reward signal for this action step?"

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
| **Units** | reward units (sum of all reward components) |
| **Range** | `[-inf, +inf]` — typically small values around -1.0 to +1.0 |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> The total reward is the sum of all reward components: base accuracy delta, bounded attribution, compute rent, stage bonus, ratio penalty, alpha shock, hindsight credit, warnings, and terminal bonuses.
>
> This is the primary signal that the RL agent (Tamiyo) optimizes. The total reward directly shapes policy learning and determines whether actions are reinforced or discouraged.
>
> - **Positive values:** Net positive outcome - actions are being reinforced
> - **Negative values:** Net negative outcome - actions are being discouraged
> - **Zero:** Neutral step (rare in practice)
>
> The total reward is displayed prominently with bold green/red styling based on sign, making it immediately visible to operators monitoring training health.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value > 0` | Positive reinforcement signal |
| **Warning** | `-0.1 < value <= 0` | Weak or neutral signal |
| **Critical** | `value <= -0.1` | Consistent negative rewards may indicate reward hacking or environment issues |

**Display Color Logic:** Bold green if >= 0, bold red if < 0

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Reward computation in contribution reward functions |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 814 |

```python
components.total_reward = reward
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.total_reward` | `simic/rewards/reward_telemetry.py` (line 58) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1469) |
| **4. Delivery** | Written to `env.reward_components.total` | `karn/sanctum/schema.py` (line 1105) |

```
[compute_contribution_reward()]
  --components.total_reward-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.total]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `total` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.total` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1105 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 546-563) | Displayed as "+0.123" or "-0.456" with bold green/red styling in "Reward Total" row |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.total_reward` at line 814
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.total: float = 0.0` at line 1105
- [x] **Default is correct** — `0.0` is appropriate default before first reward computation
- [x] **Consumer reads it** — EnvDetailScreen directly accesses `rc.total` at line 546
- [x] **Display is correct** — Rendered with `+.3f` format, bold green if >= 0, bold red if < 0

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Reward computation tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Total reward rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for any environment
4. Observe "Reward Total" row — should show current total reward with appropriate coloring
5. Verify color coding: bold green for positive values, bold red for negative values
6. After training, query telemetry: `SELECT total_reward FROM rewards LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| All reward components | computation | Total is sum of base_acc_delta, bounded_attribution, compute_rent, stage_bonus, penalties, bonuses |
| Reward computation | function | Only populated when reward is computed (action step) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Reward Total row | display | Primary consumer for TUI display |
| PBRS fraction calculation | derived | Used as denominator: `abs(stage_bonus) / abs(total)` |
| Reward health assessment | analysis | Aggregate totals used to assess training health |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** `total` is the authoritative reward signal that the RL agent optimizes. It is the sum of all reward components and represents the final outcome of the reward function.
>
> **PBRS Fraction Display:** The EnvDetailScreen displays a PBRS (Potential-Based Reward Shaping) fraction alongside the total, calculated as `abs(stage_bonus) / abs(total)`. A healthy PBRS fraction is 10-40% (DisplayThresholds.PBRS_HEALTHY_MIN/MAX). This helps operators detect reward shaping dominance.
>
> **Aggregator Mapping:** Note that the telemetry field is `total_reward` but the schema field is `total`. The aggregator maps between them at line 1469: `env.reward_components.total = rc.total_reward`.
