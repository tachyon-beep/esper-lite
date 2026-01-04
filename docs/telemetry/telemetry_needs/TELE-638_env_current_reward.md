# Telemetry Record: [TELE-638] Environment Current Reward

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-638` |
| **Name** | Environment Current Reward |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What was the most recent reward signal received by this environment?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `float` (computed property) |
| **Units** | reward units (unbounded) |
| **Range** | Typically `[-5.0, +5.0]` but can exceed |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` (when reward_history is empty) |

### Semantic Meaning

> Current reward is the most recent reward value from the reward_history deque. It represents the immediate feedback from the last training step:
>
> - **Positive:** Training action was beneficial (accuracy improved, good decisions)
> - **Negative:** Training action was harmful (accuracy dropped, compute rent, penalties)
> - **Zero:** Neutral action or no reward computed
>
> This is a computed property, not a stored field: `reward_history[-1] if reward_history else 0.0`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `current_reward > 0` | Positive feedback, good progress |
| **Warning** | `-0.5 <= current_reward <= 0` | Neutral or slightly negative |
| **Critical** | `current_reward < -0.5` | Significantly negative, issues present |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_reward()` (lines 656-672)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | ANALYTICS_SNAPSHOT(kind=last_action) telemetry event |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_analytics_snapshot()` |
| **Line(s)** | 1442-1444 |

```python
# Update reward tracking
total_reward = payload.total_reward if payload.total_reward is not None else 0.0
env.reward_history.append(total_reward)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits ANALYTICS_SNAPSHOT(kind=last_action) with total_reward | `simic/training.py` |
| **2. Collection** | Aggregator extracts total_reward from payload | `aggregator.py` (line 1442) |
| **3. Aggregation** | Appended to env.reward_history deque | `aggregator.py` (line 1443) |
| **4. Delivery** | Computed via `snapshot.envs[env_id].current_reward` property | `schema.py` (lines 550-553) |

```
[AnalyticsSnapshotPayload.total_reward]
  --ANALYTICS_SNAPSHOT(kind=last_action)-->
  [SanctumAggregator._handle_analytics_snapshot()]
  --env.reward_history.append(total_reward)-->
  [EnvState.reward_history]
  --property-->
  [EnvState.current_reward = reward_history[-1]]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `current_reward` (property) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].current_reward` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 550-553 |
| **Default Value** | `0.0` (when reward_history empty) |

```python
@property
def current_reward(self) -> float:
    """Get most recent reward."""
    return self.reward_history[-1] if self.reward_history else 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 656-672) | Reward column with color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — ANALYTICS_SNAPSHOT(kind=last_action) includes total_reward
- [x] **Transport works** — Aggregator appends to reward_history in _handle_analytics_snapshot()
- [x] **Schema field exists** — `EnvState.current_reward` property at lines 550-553
- [x] **Default is correct** — `0.0` when reward_history is empty
- [x] **Consumer reads it** — EnvOverview._format_reward() accesses env.current_reward
- [x] **Display is correct** — Color-coded: green (>0), white (0 to -0.5), red (<-0.5)
- [x] **Thresholds applied** — 0 and -0.5 thresholds in _format_reward()

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_analytics_snapshot_updates_reward` | `[ ]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_current_reward_property` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Reward formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Reward column — should show current reward with color coding
4. Verify positive rewards are green, negative rewards below -0.5 are red
5. Note the format: "current (mean)" e.g., "+0.15 (+0.08)"

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| ANALYTICS_SNAPSHOT events | event | Provides total_reward each step |
| reward_history deque | storage | Stores recent rewards (maxlen=50) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Reward column | display | Shows current and mean reward |
| mean_reward (TELE-639) | derived | Computed from same reward_history |
| Reward sparkline | display | Built from reward_history |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** current_reward is a computed property (not stored field) to avoid storing redundant data. The reward_history deque is the source of truth.
>
> **Display Format:** The reward column shows both current and mean: `"{current:+.2f} ({mean:+.2f})"`. This provides immediate feedback plus trend context.
>
> **Wiring Status:** Fully wired via ANALYTICS_SNAPSHOT(kind=last_action) events. The total_reward field is populated by the reward computation in simic.
