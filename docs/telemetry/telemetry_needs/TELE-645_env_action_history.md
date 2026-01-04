# Telemetry Record: [TELE-645] Environment Action History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-645` |
| **Name** | Environment Action History |
| **Category** | `env` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What actions has Tamiyo taken recently for this environment?"

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
| **Type** | `deque[str]` |
| **Units** | list of action names |
| **Range** | Last 50 actions (maxlen) |
| **Precision** | N/A (categorical) |
| **Default** | Empty deque |

### Semantic Meaning

> Action history is a rolling window of the most recent actions taken by Tamiyo for this environment:
>
> - **WAIT:** No intervention, let training continue
> - **GERMINATE:** Spawn a new seed in a slot
> - **SET_ALPHA_TARGET:** Adjust blend alpha for a seed
> - **PRUNE:** Remove an underperforming seed
> - **FOSSILIZE:** Permanently integrate a successful seed
> - **ADVANCE:** Move seed to next lifecycle stage
>
> Actions are normalized (e.g., GERMINATE_CONV_LIGHT -> GERMINATE) before storage.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | Informational field, no thresholds |

**Threshold Source:** N/A (categorical sequence)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | ANALYTICS_SNAPSHOT(kind=last_action) telemetry event |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_analytics_snapshot()` |
| **Line(s)** | 1448-1450 |

```python
# Update action tracking (with normalization)
action_name = normalize_action(payload.action_name or "UNKNOWN")
env.action_history.append(action_name)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits ANALYTICS_SNAPSHOT with action_name | `simic/training.py` |
| **2. Collection** | Aggregator extracts and normalizes action_name | `aggregator.py` (line 1448) |
| **3. Aggregation** | Appended to env.action_history deque | `aggregator.py` (line 1449) |
| **4. Delivery** | Available at `snapshot.envs[env_id].action_history` | `schema.py` (line 487) |

```
[AnalyticsSnapshotPayload.action_name]
  --ANALYTICS_SNAPSHOT(kind=last_action)-->
  [SanctumAggregator._handle_analytics_snapshot()]
  --normalize_action()-->
  [env.action_history.append(action_name)]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].action_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `action_history` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].action_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 487 |
| **Default Value** | Empty `deque(maxlen=50)` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 780-796) | Last column shows most recent action (abbreviated) |
| BestRunRecord | `schema.py` (line 1262) | best_action_history captured at peak accuracy |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — ANALYTICS_SNAPSHOT(kind=last_action) includes action_name
- [x] **Transport works** — Aggregator normalizes and appends to action_history
- [x] **Schema field exists** — `EnvState.action_history: deque[str]` at line 487
- [x] **Default is correct** — Empty deque with maxlen=50
- [x] **Consumer reads it** — EnvOverview._format_last_action() reads action_history[-1]
- [x] **Display is correct** — Shows abbreviated last action (WAIT, GERM, ALPH, etc.)
- [x] **Thresholds applied** — N/A (informational)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_analytics_snapshot_updates_action_history` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_action_normalization` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Last action formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Last column — should show most recent action
4. Verify action abbreviations: WAIT, GERM, ALPH, FOSS, PRUN
5. Watch for action changes as policy makes decisions

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| ANALYTICS_SNAPSHOT events | event | Provides action_name each step |
| normalize_action() | function | Normalizes factored actions to base names |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Last column | display | Shows most recent action |
| action_counts tracking | derived | Incremented based on action type |
| BestRunRecord.best_action_history | data | Snapshot at peak accuracy |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Actions are normalized before storage to group factored action variants. For example, GERMINATE_CONV_LIGHT, GERMINATE_CONV_HEAVY, and GERMINATE_ATTENTION all become "GERMINATE". This simplifies counting and display.
>
> **Normalization Map:** See `ACTION_NORMALIZATION` dict and `normalize_action()` function in aggregator.py (lines 85-148).
>
> **Display Abbreviations:** The Last column uses 4-character abbreviations for space efficiency:
> - WAIT -> WAIT
> - GERMINATE -> GERM
> - SET_ALPHA_TARGET -> ALPH
> - FOSSILIZE -> FOSS
> - PRUNE -> PRUN
>
> **Episode Reset:** action_history is cleared at BATCH_EPOCH_COMPLETED (aggregator line 1318) for fresh tracking each episode.
>
> **Wiring Status:** Fully wired via ANALYTICS_SNAPSHOT(kind=last_action) events.
