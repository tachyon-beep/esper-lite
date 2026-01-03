# Telemetry Record: [TELE-639] Environment Mean Reward

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-639` |
| **Name** | Environment Mean Reward |
| **Category** | `env` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What is the average reward signal over the recent history for this environment?"

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

> Mean reward is the arithmetic mean of the reward_history deque (up to 50 recent rewards). It provides a smoothed view of reward trends:
>
> - **Positive mean:** Training is generally positive, making progress
> - **Negative mean:** Training is struggling, likely paying more rent/penalties than earning bonuses
> - **Zero mean:** Balanced or no significant activity
>
> This is a computed property: `sum(reward_history) / len(reward_history)`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `mean_reward > 0` | Generally positive training |
| **Warning** | `mean_reward ~ 0` | Neutral, may need tuning |
| **Critical** | `mean_reward < 0` | Consistently negative, issues present |

**Threshold Source:** Not directly color-coded; shown alongside current_reward for context

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from reward_history populated by ANALYTICS_SNAPSHOT events |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState.mean_reward` property |
| **Line(s)** | 555-560 |

```python
@property
def mean_reward(self) -> float:
    """Mean reward over history."""
    if not self.reward_history:
        return 0.0
    return sum(self.reward_history) / len(self.reward_history)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Rewards appended via ANALYTICS_SNAPSHOT handling | `aggregator.py` (line 1443) |
| **2. Collection** | Stored in reward_history deque (maxlen=50) | `schema.py` (line 484) |
| **3. Aggregation** | Mean computed on-demand via property | `schema.py` (lines 555-560) |
| **4. Delivery** | Available at `snapshot.envs[env_id].mean_reward` | `schema.py` (lines 555-560) |

```
[reward_history deque updates]
  --property access-->
  [sum(self.reward_history) / len(self.reward_history)]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].mean_reward]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `mean_reward` (property) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].mean_reward` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 555-560 |
| **Default Value** | `0.0` (when reward_history empty) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (line 672) | Shown in parentheses after current: "+0.15 (+0.08)" |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed from reward_history populated by ANALYTICS_SNAPSHOT
- [x] **Transport works** — reward_history is updated each step; mean computed on access
- [x] **Schema field exists** — `EnvState.mean_reward` property at lines 555-560
- [x] **Default is correct** — `0.0` when reward_history is empty
- [x] **Consumer reads it** — EnvOverview._format_reward() accesses env.mean_reward
- [x] **Display is correct** — Shown as dimmed value in parentheses
- [x] **Thresholds applied** — Not directly thresholded; contextual display only

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_mean_reward_property` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Mean reward display | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Reward column — should show "current (mean)" format
4. Verify mean stabilizes as more history accumulates
5. Note mean is dimmed to distinguish from current reward

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| reward_history deque | storage | Source data for mean calculation |
| ANALYTICS_SNAPSHOT events | event | Populates reward_history |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Reward column | display | Shows mean in parentheses |
| aggregate_mean_reward | computation | Mean across all envs |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** mean_reward is a computed property to avoid storing redundant data. The calculation is O(n) where n is history length (max 50), which is negligible.
>
> **Display Context:** Mean reward is shown dimmed and in parentheses to distinguish it from the current (more immediate) reward. This provides trend context without visual dominance.
>
> **Window Size:** reward_history maxlen=50 provides a smoothing window of roughly 50 steps. This balances responsiveness with stability.
>
> **Wiring Status:** Fully wired and operational.
