# Telemetry Record: [TELE-632] Environment Rolled Back

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-632` |
| **Name** | Environment Rolled Back |
| **Category** | `env` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Has this environment experienced a catastrophic failure requiring governor intervention and rollback?"

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
| **Type** | `bool` |
| **Units** | boolean flag |
| **Range** | `True`, `False` |
| **Precision** | N/A |
| **Default** | `False` |

### Semantic Meaning

> Rolled back indicates that the governor detected a catastrophic failure in this environment and triggered an emergency rollback. When `True`:
>
> - The environment row displays a red alert overlay instead of normal metrics
> - Training has been paused or reset for this environment
> - The `rollback_reason` field provides details about the failure type
>
> The flag is automatically cleared when training resumes (next EPOCH_COMPLETED event).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `rolled_back == False` | Environment operating normally |
| **Critical** | `rolled_back == True` | Catastrophic failure, governor intervention active |

**Threshold Source:** Binary flag (no intermediate thresholds)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | GOVERNOR_ROLLBACK telemetry event |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `GovernorRollbackPayload` |
| **Line(s)** | (varies) |

```python
@dataclass
class GovernorRollbackPayload:
    env_id: int
    reason: str  # "governor_nan", "governor_lobotomy", "governor_divergence"
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Governor detects failure and emits GOVERNOR_ROLLBACK | `simic/governor.py` |
| **2. Collection** | Aggregator receives GOVERNOR_ROLLBACK event | `aggregator.py` (line 352) |
| **3. Aggregation** | Handler sets `env.rolled_back = True` | `aggregator.py` (lines 1604-1631) |
| **4. Delivery** | Available at `snapshot.envs[env_id].rolled_back` | `schema.py` (line 546) |

```
[Governor failure detection]
  --GovernorRollbackPayload-->
  [emit GOVERNOR_ROLLBACK event]
  --event-->
  [SanctumAggregator._handle_governor_rollback()]
  --env.rolled_back = True-->
  [EnvState.rolled_back]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].rolled_back]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `rolled_back` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].rolled_back` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 546 |
| **Default Value** | `False` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 279-282, 365-398) | Displays red alert row overlay when rolled_back is True |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Governor emits GOVERNOR_ROLLBACK event on catastrophic failure
- [x] **Transport works** — Aggregator handles GOVERNOR_ROLLBACK and sets rolled_back flag
- [x] **Schema field exists** — `EnvState.rolled_back: bool = False` at line 546
- [x] **Default is correct** — `False` indicates normal operation
- [x] **Consumer reads it** — EnvOverview checks `env.rolled_back` before rendering row
- [x] **Display is correct** — Red alert overlay shows "CATASTROPHIC FAILURE - ROLLED BACK (reason)"
- [x] **Thresholds applied** — Binary flag, no intermediate thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| E2E (telemetry) | `tests/telemetry/test_tele_env_state.py` | `TestTELE632EnvRolledBack` (7 tests) | `[x]` |
| Unit (emitter) | `tests/simic/test_governor.py` | `test_governor_rollback_emission` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_governor_rollback_sets_flag` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Rollback alert rendering | `[ ]` |

### Manual Verification Steps

1. Start training with conditions likely to trigger NaN (e.g., very high learning rate)
2. Launch Sanctum TUI
3. Observe when governor detects NaN — affected env row should turn red
4. Verify alert message shows "CATASTROPHIC FAILURE - ROLLED BACK (NAN DETECTED)" or similar
5. Wait for env to resume training — alert should clear

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| GOVERNOR_ROLLBACK event | event | Sets the rolled_back flag |
| Governor failure detection | system | Must detect NaN/lobotomy/divergence |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview row rendering | display | Switches to alert overlay when True |
| rollback_reason (TELE-633) | data | Provides details when rolled_back is True |
| rollback_timestamp | data | Records when rollback occurred |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** The rolled_back flag is automatically cleared when the next EPOCH_COMPLETED event arrives for the env (line 696-698 in aggregator). This indicates training has resumed successfully.
>
> **Visual Treatment:** When rolled_back is True, the entire row is replaced with a red alert overlay. This is intentionally disruptive to ensure the operator notices the failure.
>
> **Related Fields:** Use in conjunction with TELE-633 (rollback_reason) for full context.
>
> **Wiring Status:** Fully wired and operational. The rollback flow from governor detection through TUI display is complete.
