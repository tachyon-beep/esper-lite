# Telemetry Record: [TELE-801] Last Action Env ID

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-801` |
| **Name** | Last Action Env ID |
| **Category** | `ui` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "Which environment just received a Tamiyo action? Which row should I watch for the outcome?"

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
| **Type** | `int | None` |
| **Units** | env_id (integer identifier) |
| **Range** | `[0, num_envs-1]` or `None` if no actions yet |
| **Precision** | exact integer |
| **Default** | `None` (before first action) |

### Semantic Meaning

> The environment ID that received the most recent Tamiyo policy action. When an action is taken (GERMINATE, ADVANCE, FOSSILIZE, PRUNE, SET_ALPHA_TARGET, WAIT), this field records which environment the action targeted.
>
> Used by the EnvOverview widget to show a cyan `>` prefix on the targeted env row, helping operators visually track which environment just received attention.
>
> Works in conjunction with `last_action_timestamp` (TELE-802) for hysteresis: the indicator only shows if the action occurred within the last 5 seconds, preventing visual jitter as action focus moves between environments.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Normal** | Any valid env_id or None | Normal operation |

**Note:** This is a UI feedback field, not a health metric. No warning/critical thresholds apply.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Tamiyo policy step (ANALYTICS_SNAPSHOT kind=last_action) |
| **File** | `/home/john/esper-lite/src/esper/simic/policy/tamiyo_policy.py` |
| **Function/Method** | Policy step emitting ANALYTICS_SNAPSHOT |
| **Telemetry Event** | `ANALYTICS_SNAPSHOT(kind="last_action")` with `env_id` field |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | ANALYTICS_SNAPSHOT event with env_id in payload | `simic/policy/tamiyo_policy.py` |
| **2. Collection** | AnalyticsSnapshotPayload.env_id | `leyline/telemetry_events.py` |
| **3. Aggregation** | Extracted in `_handle_analytics_snapshot(kind=last_action)` handler | `karn/sanctum/aggregator.py` (line 1454) |
| **4. Delivery** | Written to `snapshot.last_action_env_id` | `karn/sanctum/schema.py` (line 1389) |

```
[TamiyoPolicy.step()]
  --ANALYTICS_SNAPSHOT(kind=last_action, env_id=X)-->
  [SanctumAggregator._handle_analytics_snapshot()]
  --_last_action_env_id = env_id-->
  [SanctumSnapshot.last_action_env_id]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `last_action_env_id` |
| **Path from SanctumSnapshot** | `snapshot.last_action_env_id` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1389 |
| **Default Value** | `None` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 491-524) | Shows cyan `>` prefix on env row via `_format_env_id()` |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - ANALYTICS_SNAPSHOT(kind=last_action) includes env_id
- [x] **Transport works** - env_id extracted from payload in aggregator line 1454
- [x] **Schema field exists** - `SanctumSnapshot.last_action_env_id: int | None = None` at line 1389
- [x] **Default is correct** - `None` is appropriate before first action
- [x] **Consumer reads it** - EnvOverview._format_env_id() reads snapshot.last_action_env_id
- [x] **Display is correct** - Cyan `>` prefix shown for 5 seconds after action
- [x] **Hysteresis applied** - 5-second decay using last_action_timestamp (TELE-802)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Schema | `tests/telemetry/test_tele_action_targeting.py` | `TestTELE801LastActionEnvId` | `[x]` |
| Aggregator | `tests/telemetry/test_tele_action_targeting.py` | `test_aggregator_stores_last_action_env_id` | `[x]` |
| Widget logic | `tests/telemetry/test_tele_action_targeting.py` | `TestEnvOverviewFormatEnvId` | `[x]` |
| Integration | `tests/telemetry/test_tele_action_targeting.py` | `TestActionTargetingIntegration` | `[x]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Observe EnvOverview table rows
4. When an action occurs, the targeted env row should show cyan `>` prefix
5. The indicator should fade (disappear) after ~5 seconds
6. New action on different env should move the indicator to that row

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| ANALYTICS_SNAPSHOT(last_action) event | event | Must emit env_id |
| Policy step execution | execution | Actions must be taken |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| TELE-802 (last_action_timestamp) | telemetry | Used together for hysteresis |
| EnvOverview indicator | display | Visual feedback for action targeting |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** The 5-second hysteresis prevents visual jitter when actions occur rapidly across different environments. Without hysteresis, the indicator would flash between rows too quickly to be useful.
>
> **UX Rationale:** The cyan color was chosen per UX accessibility review - it's distinct from the green/yellow/red status colors and provides clear visual contrast for the action targeting indicator.
>
> **Wiring Status:** Fully wired and operational. Emitter (policy), transport (aggregator), schema, and consumer (EnvOverview) are all correctly implemented.
