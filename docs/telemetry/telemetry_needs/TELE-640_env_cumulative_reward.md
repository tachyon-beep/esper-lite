# Telemetry Record: [TELE-640] Environment Cumulative Reward

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-640` |
| **Name** | Environment Cumulative Reward |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What is the total reward accumulated by this environment throughout the current episode?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

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
| **Units** | reward units (unbounded) |
| **Range** | Typically `[-50.0, +50.0]` per episode, but can exceed |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Cumulative reward is the running sum of all rewards received during the current episode. This is the primary RL performance metric:
>
> - **Positive cumulative:** Episode is net positive, policy is working
> - **Negative cumulative:** Episode is net negative, policy needs improvement
> - **Highly negative (<-5):** Episode is struggling significantly
>
> Reset to 0.0 at the start of each episode (BATCH_EPOCH_COMPLETED).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `cumulative_reward > 0` | Positive episode return |
| **Warning** | `-5 <= cumulative_reward <= 0` | Slightly negative, normal variance |
| **Critical** | `cumulative_reward < -5` | Significantly negative episode |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_cumulative_reward()` (lines 588-602)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Accumulated from ANALYTICS_SNAPSHOT(kind=last_action) events |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_analytics_snapshot()` |
| **Line(s)** | 1444 |

```python
# Update reward tracking
total_reward = payload.total_reward if payload.total_reward is not None else 0.0
env.reward_history.append(total_reward)
env.cumulative_reward += total_reward
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits ANALYTICS_SNAPSHOT with total_reward | `simic/training.py` |
| **2. Collection** | Aggregator extracts total_reward | `aggregator.py` (line 1442) |
| **3. Aggregation** | Added to env.cumulative_reward | `aggregator.py` (line 1444) |
| **4. Delivery** | Available at `snapshot.envs[env_id].cumulative_reward` | `schema.py` (line 488) |

```
[AnalyticsSnapshotPayload.total_reward]
  --ANALYTICS_SNAPSHOT(kind=last_action)-->
  [SanctumAggregator._handle_analytics_snapshot()]
  --env.cumulative_reward += total_reward-->
  [EnvState.cumulative_reward]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].cumulative_reward]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `cumulative_reward` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].cumulative_reward` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 488 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 588-602) | "Sum Rwd" column with color coding |
| EnvOverview (aggregate) | `widgets/env_overview.py` (lines 443, 449) | Total cumulative across all envs |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — ANALYTICS_SNAPSHOT(kind=last_action) includes total_reward
- [x] **Transport works** — Aggregator accumulates to cumulative_reward in _handle_analytics_snapshot()
- [x] **Schema field exists** — `EnvState.cumulative_reward: float = 0.0` at line 488
- [x] **Default is correct** — `0.0` at start of each episode
- [x] **Consumer reads it** — EnvOverview._format_cumulative_reward() reads the value
- [x] **Display is correct** — Color-coded: green (>0), white (0 to -5), red (<-5)
- [x] **Thresholds applied** — 0 and -5 thresholds in _format_cumulative_reward()

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_cumulative_reward_accumulation` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_cumulative_reward_reset_at_episode` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Cumulative reward formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe "Sum Rwd" column — should accumulate throughout each episode
4. Verify color coding: green (>0), red (<-5)
5. Watch for reset to 0 when new episode starts

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| ANALYTICS_SNAPSHOT events | event | Provides total_reward each step |
| BATCH_EPOCH_COMPLETED | event | Resets cumulative_reward to 0 |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview "Sum Rwd" column | display | Per-env episode return |
| Aggregate total | computation | Sum across all envs |
| Episode return tracking | analysis | Post-hoc episode analysis |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** cumulative_reward is reset at each BATCH_EPOCH_COMPLETED (aggregator line 1302). This provides a clean per-episode return metric rather than an ever-growing total.
>
> **RL Significance:** Cumulative reward (episode return) is the primary metric for RL algorithm evaluation. PPO maximizes expected cumulative reward.
>
> **Threshold Rationale:** -5 as critical threshold is based on typical episode lengths (75 epochs) and reward scales. An episode averaging -0.07/step would hit -5.
>
> **Wiring Status:** Fully wired via ANALYTICS_SNAPSHOT(kind=last_action) events.
