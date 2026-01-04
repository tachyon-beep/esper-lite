# Telemetry Record: [TELE-649] Environment Action Counts

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-649` |
| **Name** | Environment Action Counts |
| **Category** | `environment` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What is the distribution of action types Tamiyo has taken for this environment?"

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
| **Type** | `dict[str, int]` |
| **Units** | per-action-type counts |
| **Range** | 0 to unbounded for each key |
| **Precision** | N/A (integers) |
| **Default** | `{"WAIT": 0, "GERMINATE": 0, "SET_ALPHA_TARGET": 0, "PRUNE": 0, "FOSSILIZE": 0, "ADVANCE": 0}` |

### Semantic Meaning

> Action counts is a dictionary tracking the number of times each action type has been taken:
>
> - **WAIT:** No intervention, let training continue
> - **GERMINATE:** Spawn a new seed in a slot
> - **SET_ALPHA_TARGET:** Adjust blend alpha for a seed
> - **PRUNE:** Remove an underperforming seed
> - **FOSSILIZE:** Permanently integrate a successful seed
> - **ADVANCE:** Move seed to next lifecycle stage
>
> Actions are normalized before counting (e.g., GERMINATE_CONV_LIGHT -> GERMINATE).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | Informational field, no thresholds |

**Threshold Source:** N/A (distribution for visualization)

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
# Normalize action and increment count
action_name = normalize_action(raw_action)
env.action_counts[action_name] += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits action telemetry event | `simic/training.py` |
| **2. Collection** | Aggregator extracts and normalizes action | `aggregator.py` |
| **3. Aggregation** | Increments env.action_counts[action_name] | `aggregator.py` |
| **4. Delivery** | Available at `snapshot.envs[env_id].action_counts` | `schema.py` (lines 514-521) |

```
[Action telemetry event]
  --event-->
  [SanctumAggregator]
  --normalize_action()-->
  [env.action_counts[action_name] += 1]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].action_counts]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `action_counts` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].action_counts` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 514-521 |
| **Default Value** | Dict with all action types initialized to 0 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 527-539) | Displayed as color-coded percentages |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Action telemetry events trigger count increment
- [x] **Transport works** — Aggregator normalizes action and increments count
- [x] **Schema field exists** — `EnvState.action_counts: dict[str, int]` at lines 514-521
- [x] **Default is correct** — Dict with all action types initialized to 0
- [x] **Consumer reads it** — EnvDetailScreen._render_metrics() reads action_counts
- [x] **Display is correct** — Shows color-coded percentages for each action type
- [x] **Thresholds applied** — N/A (informational)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_action_counts_tracking` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_action_normalization_for_counts` | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/widgets/test_env_detail_screen.py` | Action distribution color-coded display | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI and select an environment
3. Navigate to EnvDetailScreen detail view
4. Observe "Action Distribution" row — should show color-coded percentages
5. Verify colors match action types: WAIT=dim, GERMINATE=cyan, SET_ALPHA_TARGET=yellow, FOSSILIZE=green, PRUNE=red

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Action telemetry events | event | Triggers count increment for each action |
| normalize_action() | function | Normalizes factored actions to base names |
| total_actions (TELE-648) | field | Denominator for percentage calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen action distribution | display | Color-coded percentage display |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Display Details:** Action counts are displayed as percentages using `total_actions` (TELE-648) as the denominator. Each action type has a designated color for visual distinction:
>
> | Action | Color |
> |--------|-------|
> | WAIT | dim |
> | GERMINATE | cyan |
> | SET_ALPHA_TARGET | yellow |
> | FOSSILIZE | green |
> | PRUNE | red |
> | ADVANCE | white (default) |
>
> **Action Normalization:** Factored action variants are normalized before counting:
> - GERMINATE_CONV_LIGHT -> GERMINATE
> - GERMINATE_DENSE_HEAVY -> GERMINATE
> - FOSSILIZE_R0C0 -> FOSSILIZE
> - PRUNE_R1C1 -> PRUNE
> - ADVANCE_R1C1 -> ADVANCE
>
> **Relationship to total_actions:** The sum of all values in `action_counts` should equal `total_actions` (TELE-648). Both are updated together when an action is recorded.
>
> **Zero Display:** When `total_actions` is 0, the "Action Distribution" row displays a dim placeholder "--" instead of attempting percentage calculation.
