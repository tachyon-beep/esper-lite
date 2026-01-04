# Telemetry Record: [TELE-642] Environment Host Parameters

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-642` |
| **Name** | Environment Host Parameters |
| **Category** | `env` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How many parameters does the baseline host model have in this environment?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `int` |
| **Units** | parameter count |
| **Range** | `[0, billions]` typically 1K-100M |
| **Precision** | Integer |
| **Default** | `0` |

### Semantic Meaning

> Host parameters is the baseline parameter count of the host model before any seeds are fossilized. This serves as the denominator for growth_ratio calculation:
>
> - Used to compute `growth_ratio = (host_params + fossilized_params) / host_params`
> - Represents the "original" model size for efficiency comparisons
> - Captured once at TRAINING_STARTED and propagated to all environments
>
> All environments in a run share the same host_params value.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | This is an informational field, not a health metric |

**Threshold Source:** N/A (static configuration value)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | TRAINING_STARTED telemetry event |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `TrainingStartedPayload.host_params` |
| **Line(s)** | (varies) |

```python
@dataclass
class TrainingStartedPayload:
    ...
    host_params: int = 0  # Baseline host model parameter count
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training start captures host model param count | `simic/training.py` |
| **2. Collection** | Aggregator captures in _handle_training_started() | `aggregator.py` (lines 633-635) |
| **3. Aggregation** | Stored in aggregator._host_params, propagated to EnvState | `aggregator.py` (line 1688) |
| **4. Delivery** | Available at `snapshot.envs[env_id].host_params` | `schema.py` (line 454) |

```
[TrainingStartedPayload.host_params]
  --TRAINING_STARTED-->
  [SanctumAggregator._handle_training_started()]
  --self._host_params = payload.host_params-->
  [SanctumAggregator._ensure_env()]
  --host_params=self._host_params-->
  [EnvState.host_params]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].host_params]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `host_params` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].host_params` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 454 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| growth_ratio property | `schema.py` (lines 569-578) | Denominator for ratio calculation |
| BestRunRecord | `schema.py` (line 1265) | Captured for historical detail |
| SystemVitals | `schema.py` (line 1037) | Stored for system overview |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — TrainingStartedPayload includes host_params field
- [x] **Transport works** — Aggregator captures and propagates to all EnvState instances
- [x] **Schema field exists** — `EnvState.host_params: int = 0` at line 454
- [x] **Default is correct** — `0` before TRAINING_STARTED received
- [x] **Consumer reads it** — growth_ratio property uses host_params as denominator
- [x] **Display is correct** — Not directly displayed; used for ratio calculation
- [x] **Thresholds applied** — N/A (informational field)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_training_started_captures_host_params` | `[ ]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_growth_ratio_calculation` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Query telemetry: `SELECT host_params FROM runs LIMIT 1`
3. Verify host_params matches expected model size for preset
4. Check growth_ratio column shows 1.0x initially (no fossilized params yet)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TRAINING_STARTED event | event | Provides host_params configuration |
| Host model initialization | system | Must count params before training |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| growth_ratio (TELE-644) | derived | Uses host_params as denominator |
| BestRunRecord | data | Captured for historical comparison |
| SystemVitals.host_params | data | Stored for system overview |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** host_params is set once at TRAINING_STARTED and never changes. This provides a stable baseline for growth_ratio calculation throughout the run.
>
> **Shared Value:** All environments in a run share the same host_params value since they use the same base model architecture.
>
> **Related Fields:** Used in conjunction with fossilized_params (TELE-643) to compute growth_ratio (TELE-644).
>
> **Wiring Status:** Fully wired via TRAINING_STARTED event.
