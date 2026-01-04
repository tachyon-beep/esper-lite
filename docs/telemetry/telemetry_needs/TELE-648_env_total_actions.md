# Telemetry Record: [TELE-648] Environment Total Actions

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-648` |
| **Name** | Environment Total Actions |
| **Category** | `environment` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How many total actions has Tamiyo taken for this environment?"

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
| **Type** | `int` |
| **Units** | count of actions |
| **Range** | 0 to unbounded |
| **Precision** | N/A (integer) |
| **Default** | 0 |

### Semantic Meaning

> Total actions is the count of all actions taken by Tamiyo for this environment:
>
> - Serves as the denominator for calculating action distribution percentages
> - Includes all action types: WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE, ADVANCE
> - Incremented each time an action is recorded via add_action()
> - Used to compute per-action-type percentages in the action distribution display

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | Informational field, no thresholds |

**Threshold Source:** N/A (counter for percentage calculation)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregator action event handling |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `EnvState.add_action()` or aggregator action handling |
| **Line(s)** | Aggregator action event handling |

```python
# Increment total action count
env.total_actions += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits action telemetry event | `simic/training.py` |
| **2. Collection** | Aggregator extracts action from event | `aggregator.py` |
| **3. Aggregation** | Increments env.total_actions | `aggregator.py` |
| **4. Delivery** | Available at `snapshot.envs[env_id].total_actions` | `schema.py` (line 522) |

```
[Action telemetry event]
  --event-->
  [SanctumAggregator]
  --action counting-->
  [env.total_actions += 1]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].total_actions]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `total_actions` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].total_actions` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 522 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 525-542) | Denominator for action distribution percentages |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Action telemetry events trigger count increment
- [x] **Transport works** — Aggregator increments total_actions
- [x] **Schema field exists** — `EnvState.total_actions: int` at line 522
- [x] **Default is correct** — 0
- [x] **Consumer reads it** — EnvDetailScreen._render_metrics() uses total_actions as denominator
- [x] **Display is correct** — Used to calculate percentages: `(count / total_actions) * 100`
- [x] **Thresholds applied** — N/A (informational)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_total_actions_counting` | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/widgets/test_env_detail_screen.py` | Action distribution percentage calculation | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI and select an environment
3. Navigate to EnvDetailScreen detail view
4. Observe "Action Distribution" row — percentages should sum to ~100%
5. Verify total_actions > 0 when percentages are displayed (not dim placeholder)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Action telemetry events | event | Triggers increment for each action |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen action distribution | display | Denominator for percentage calculation |
| action_counts (TELE-649) | derived | Numerators for percentage calculation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Display Details:** When `total_actions` is 0, the "Action Distribution" row displays a dim placeholder "--" instead of attempting percentage calculation. When > 0, percentages are calculated for each action type.
>
> **Relationship to action_counts:** The `total_actions` field should equal the sum of all values in `action_counts` (TELE-649). Both are updated together when an action is recorded.
>
> **Episode Behavior:** The handling of episode boundaries (whether counts reset or accumulate) should be verified during testing.
