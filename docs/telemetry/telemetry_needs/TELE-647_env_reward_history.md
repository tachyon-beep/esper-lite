# Telemetry Record: [TELE-647] Environment Reward History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-647` |
| **Name** | Environment Reward History |
| **Category** | `environment` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How have rewards evolved over the recent training window for this environment?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `deque[float]` |
| **Units** | reward values (unbounded, typically -1.0 to +1.0) |
| **Range** | Last 50 values (maxlen=50) |
| **Precision** | float |
| **Default** | Empty deque |

### Semantic Meaning

> Reward history is a rolling window of the most recent reward values for this environment:
>
> - Values represent total reward computed at each step/epoch
> - Used to generate sparklines for visual trend analysis
> - Enables quick identification of reward gaming, instability, or healthy learning
> - Complements accuracy_history to show signal vs. outcome correlation

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | Informational field, no thresholds |

**Threshold Source:** N/A (time series for visualization)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregator reward update from REWARD_COMPUTED telemetry |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `EnvState.add_reward()` or aggregator reward handling |
| **Line(s)** | Aggregator REWARD_COMPUTED event handling |

```python
# Update reward history
env.reward_history.append(reward_value)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits REWARD_COMPUTED event | `simic/training.py` |
| **2. Collection** | Aggregator extracts total reward from event | `aggregator.py` |
| **3. Aggregation** | Appended to env.reward_history deque | `aggregator.py` |
| **4. Delivery** | Available at `snapshot.envs[env_id].reward_history` | `schema.py` (line 484) |

```
[REWARD_COMPUTED telemetry event]
  --event-->
  [SanctumAggregator]
  --reward extraction-->
  [env.reward_history.append(total_reward)]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].reward_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `reward_history` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 484 |
| **Default Value** | Empty `deque(maxlen=50)` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (line 505) | Displayed as sparkline via make_sparkline() |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — REWARD_COMPUTED telemetry events include total reward
- [x] **Transport works** — Aggregator appends to reward_history
- [x] **Schema field exists** — `EnvState.reward_history: deque[float]` at line 484
- [x] **Default is correct** — Empty deque with maxlen=50
- [x] **Consumer reads it** — EnvDetailScreen._render_metrics() reads reward_history
- [x] **Display is correct** — Shows sparkline using make_sparkline(env.reward_history, width=40)
- [x] **Thresholds applied** — N/A (informational)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_reward_history_tracking` | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/widgets/test_env_detail_screen.py` | Reward sparkline rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI and select an environment
3. Navigate to EnvDetailScreen detail view
4. Observe "Reward History" row — should show sparkline visualization
5. Verify sparkline updates as training progresses

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| REWARD_COMPUTED events | event | Provides total reward values each step |
| make_sparkline() | function | Converts deque to sparkline string |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen sparkline | display | Visual trend for reward |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Display Details:** The reward history is rendered as a sparkline using `make_sparkline(env.reward_history, width=40)`. If the deque is empty, a dim placeholder "--" is displayed instead.
>
> **Maxlen:** The deque is limited to 50 entries to keep the sparkline focused on recent trends without excessive memory usage.
>
> **Complementary Field:** Works alongside `accuracy_history` (TELE-646) to provide dual sparklines showing both training signal (reward) and outcome (accuracy) trends.
>
> **Reward Interpretation:** Positive sparkline trends indicate the policy is receiving positive reinforcement. Erratic patterns may indicate reward gaming or unstable training.
