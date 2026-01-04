# Telemetry Record: [TELE-802] Last Action Timestamp

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-802` |
| **Name** | Last Action Timestamp |
| **Category** | `ui` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "When did the last Tamiyo action occur? Should the action indicator still be visible?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [ ] Developer (debugging)
- [ ] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every action)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `datetime | None` |
| **Units** | UTC datetime |
| **Range** | Valid datetime or `None` if no actions yet |
| **Precision** | second-level for hysteresis calculation |
| **Default** | `None` (before first action) |

### Semantic Meaning

> The timestamp of when the most recent Tamiyo policy action was taken. Used in conjunction with `last_action_env_id` (TELE-801) to implement a 5-second hysteresis for the action targeting indicator in EnvOverview.
>
> The hysteresis works as follows:
> - When an action occurs, both `last_action_env_id` and `last_action_timestamp` are updated
> - The EnvOverview widget computes `age = now - last_action_timestamp`
> - If `age < 5.0` seconds, the cyan `>` indicator is shown on the targeted env row
> - If `age >= 5.0` seconds, the indicator is hidden (hysteresis expired)
>
> This prevents visual jitter when actions occur rapidly across different environments, giving operators time to observe each action target.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Show Indicator** | `age < 5.0` seconds | Action is recent, indicator visible |
| **Hide Indicator** | `age >= 5.0` seconds | Action is stale, indicator hidden |

**Note:** These are display thresholds, not health thresholds. The 5-second value was chosen for UX balance between visibility and responsiveness.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | SanctumAggregator on receiving ANALYTICS_SNAPSHOT(kind=last_action) |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `_handle_analytics_snapshot()` |
| **Line(s)** | 1455 |

```python
# Track last action target for EnvOverview row highlighting (with timestamp for hysteresis)
self._last_action_env_id = env_id
self._last_action_timestamp = datetime.now(timezone.utc)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | ANALYTICS_SNAPSHOT event triggers timestamp capture | `karn/sanctum/aggregator.py` (line 1455) |
| **2. Collection** | Stored in `_last_action_timestamp` field | `karn/sanctum/aggregator.py` (line 246) |
| **3. Aggregation** | Passed to snapshot in `_get_snapshot_unlocked()` | `karn/sanctum/aggregator.py` (line 564) |
| **4. Delivery** | Written to `snapshot.last_action_timestamp` | `karn/sanctum/schema.py` (line 1390) |

```
[ANALYTICS_SNAPSHOT(kind=last_action)]
  --SanctumAggregator._handle_analytics_snapshot()-->
  [_last_action_timestamp = datetime.now(timezone.utc)]
  --_get_snapshot_unlocked()-->
  [SanctumSnapshot.last_action_timestamp]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `last_action_timestamp` |
| **Path from SanctumSnapshot** | `snapshot.last_action_timestamp` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1390 |
| **Default Value** | `None` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 514-517) | Computes age for hysteresis check in `_format_env_id()` |

```python
# From env_overview.py _format_env_id():
if last_action_timestamp is not None:
    age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
    show_indicator = age < 5.0  # 5-second hysteresis
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Timestamp captured in aggregator on ANALYTICS_SNAPSHOT(last_action)
- [x] **Transport works** - Stored in `_last_action_timestamp` aggregator field
- [x] **Schema field exists** - `SanctumSnapshot.last_action_timestamp: datetime | None = None` at line 1390
- [x] **Default is correct** - `None` is appropriate before first action
- [x] **Consumer reads it** - EnvOverview._format_env_id() reads snapshot.last_action_timestamp
- [x] **Display is correct** - Used for 5-second hysteresis calculation
- [x] **Threshold applied** - `age < 5.0` seconds to show indicator (line 516)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Schema | `tests/telemetry/test_tele_action_targeting.py` | `TestTELE802LastActionTimestamp` | `[x]` |
| Aggregator | `tests/telemetry/test_tele_action_targeting.py` | `test_aggregator_sets_timestamp_on_action` | `[x]` |
| Hysteresis | `tests/telemetry/test_tele_action_targeting.py` | `TestHysteresisThreshold` | `[x]` |
| Widget logic | `tests/telemetry/test_tele_action_targeting.py` | `TestEnvOverviewFormatEnvId` | `[x]` |
| None handling | `tests/telemetry/test_tele_action_targeting.py` | `TestNoneHandling` | `[x]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Observe EnvOverview table rows
4. When an action occurs, note the cyan `>` indicator appears immediately
5. Wait ~5 seconds without new actions on that env
6. The indicator should disappear (hysteresis expired)
7. New action should reset the timer and show indicator again

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| ANALYTICS_SNAPSHOT(last_action) event | event | Triggers timestamp capture |
| System clock | infrastructure | Used for `datetime.now(timezone.utc)` |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| TELE-801 (last_action_env_id) | telemetry | Used together for indicator display |
| EnvOverview hysteresis | display | Controls indicator visibility duration |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** The timestamp is captured in the aggregator (not passed from the event) because we need wall-clock time for UI hysteresis, not the event timestamp which may have transport delay.
>
> **Hysteresis Value:** 5 seconds was chosen as a balance between:
> - Long enough: Operator can observe which env received action
> - Short enough: Indicator doesn't persist on stale data
> - Matches: Other Sanctum staleness thresholds (5s is the standard)
>
> **Timezone:** Always UTC (`datetime.now(timezone.utc)`) for consistent comparison across system boundaries.
>
> **Wiring Status:** Fully wired and operational. The timestamp capture, schema field, and consumer hysteresis logic are all correctly implemented.
