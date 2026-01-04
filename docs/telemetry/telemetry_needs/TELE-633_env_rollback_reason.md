# Telemetry Record: [TELE-633] Environment Rollback Reason

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-633` |
| **Name** | Environment Rollback Reason |
| **Category** | `env` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Why did the governor trigger an emergency rollback for this environment?"

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
| **Type** | `str` |
| **Units** | categorical identifier |
| **Range** | `"governor_nan"`, `"governor_lobotomy"`, `"governor_divergence"`, `""` |
| **Precision** | N/A |
| **Default** | `""` (empty string) |

### Semantic Meaning

> Rollback reason identifies the type of catastrophic failure that triggered governor intervention:
>
> - **governor_nan:** NaN values detected in gradients, loss, or activations
> - **governor_lobotomy:** Severe accuracy drop (lobotomy) detected
> - **governor_divergence:** Training divergence detected (loss explosion)
> - **"":** No rollback has occurred (normal operation)
>
> This provides diagnostic context when `rolled_back == True`.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `rollback_reason == ""` | No rollback, normal operation |
| **Critical** | `rollback_reason != ""` | Catastrophic failure identified |

**Threshold Source:** N/A (categorical field with known values)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | GOVERNOR_ROLLBACK telemetry event |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `GovernorRollbackPayload.reason` |
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
| **1. Emission** | Governor includes reason in GOVERNOR_ROLLBACK event | `simic/governor.py` |
| **2. Collection** | Aggregator receives GOVERNOR_ROLLBACK event | `aggregator.py` (line 352) |
| **3. Aggregation** | Handler extracts `payload.reason` | `aggregator.py` (line 1624) |
| **4. Delivery** | Available at `snapshot.envs[env_id].rollback_reason` | `schema.py` (line 547) |

```
[Governor failure detection]
  --GovernorRollbackPayload(reason="governor_nan")-->
  [emit GOVERNOR_ROLLBACK event]
  --event-->
  [SanctumAggregator._handle_governor_rollback()]
  --env.rollback_reason = payload.reason-->
  [EnvState.rollback_reason]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].rollback_reason]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `rollback_reason` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].rollback_reason` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 547 |
| **Default Value** | `""` (empty string) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 365-398) | Shows reason in alert message: "CATASTROPHIC FAILURE - ROLLED BACK (NAN DETECTED)" |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — GovernorRollbackPayload includes reason field
- [x] **Transport works** — Aggregator extracts reason and stores in EnvState
- [x] **Schema field exists** — `EnvState.rollback_reason: str = ""` at line 547
- [x] **Default is correct** — Empty string indicates no rollback
- [x] **Consumer reads it** — EnvOverview._add_rollback_alert_row() displays formatted reason
- [x] **Display is correct** — Reason is mapped to human-readable text (NAN DETECTED, LOBOTOMY, DIVERGENCE)
- [x] **Thresholds applied** — N/A (categorical field)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| E2E (telemetry) | `tests/telemetry/test_tele_env_state.py` | `TestTELE633EnvRollbackReason` (8 tests) | `[x]` |
| Unit (emitter) | `tests/simic/test_governor.py` | `test_governor_rollback_reason` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_governor_rollback_captures_reason` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Reason formatting | `[ ]` |

### Manual Verification Steps

1. Trigger each failure type in training:
   - NaN: Use extreme learning rates or corrupt input
   - Lobotomy: Force severe accuracy drop
   - Divergence: Cause loss explosion
2. Launch Sanctum TUI
3. Verify alert message shows correct reason for each failure type
4. Confirm reason clears when env resumes training

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| GOVERNOR_ROLLBACK event | event | Provides the reason string |
| Governor failure classification | system | Must correctly identify failure type |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview alert message | display | Shows human-readable reason |
| rolled_back (TELE-632) | data | rollback_reason is only meaningful when rolled_back is True |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Rollback reason uses prefixed strings ("governor_*") to namespace the failure types. This allows future extension with additional reason codes.
>
> **Display Mapping:** The EnvOverview widget maps internal reason codes to user-friendly text:
> ```python
> reason_display = {
>     "governor_nan": "NAN DETECTED",
>     "governor_lobotomy": "LOBOTOMY",
>     "governor_divergence": "DIVERGENCE",
> }.get(env.rollback_reason, env.rollback_reason.upper())
> ```
>
> **Related Fields:** Always use in conjunction with TELE-632 (rolled_back) — rollback_reason is only meaningful when rolled_back is True.
>
> **Wiring Status:** Fully wired and operational.
