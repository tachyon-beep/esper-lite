# Telemetry Record: [TELE-634] Environment Last Update

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-634` |
| **Name** | Environment Last Update |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "When was the last telemetry event received for this environment? Is the data stale?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
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
| **Type** | `datetime | None` |
| **Units** | UTC timestamp |
| **Range** | Any valid datetime, or `None` if never updated |
| **Precision** | Seconds |
| **Default** | `None` |

### Semantic Meaning

> Last update timestamp records when the most recent telemetry event was processed for this environment. Used to detect stale data:
>
> - **Fresh (< 2s):** Data is current, shown with green indicator
> - **Stale (2-5s):** Data is slightly old, shown with yellow warning
> - **Bad (> 5s):** Data is significantly stale, shown with red indicator
>
> A `None` value indicates no telemetry has been received for this environment yet.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `age < 2.0s` | Fresh data, no staleness |
| **Warning** | `2.0s <= age <= 5.0s` | Data is slightly stale |
| **Critical** | `age > 5.0s` or `None` | Data is significantly stale or missing |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_row_staleness()` (lines 835-850)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Set in aggregator when processing EPOCH_COMPLETED |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_epoch_completed()` |
| **Line(s)** | 760 |

```python
env.last_update = datetime.now(timezone.utc)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | EPOCH_COMPLETED event triggers handler | `simic/training.py` |
| **2. Collection** | Aggregator processes event | `aggregator.py` (lines 676-766) |
| **3. Aggregation** | Handler sets `env.last_update = datetime.now(timezone.utc)` | `aggregator.py` (line 760) |
| **4. Delivery** | Available at `snapshot.envs[env_id].last_update` | `schema.py` (line 531) |

```
[EPOCH_COMPLETED event]
  --EpochCompletedPayload-->
  [SanctumAggregator._handle_epoch_completed()]
  --datetime.now(timezone.utc)-->
  [EnvState.last_update]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].last_update]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `last_update` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].last_update` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 531 |
| **Default Value** | `None` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 835-850) | Stale column with icon and color: green OK, yellow WARN, red BAD |
| EnvOverview (aggregate row) | `widgets/env_overview.py` (lines 477-487) | Shows mean telemetry age across all envs |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Aggregator sets last_update in _handle_epoch_completed()
- [x] **Transport works** — last_update is set after each EPOCH_COMPLETED event
- [x] **Schema field exists** — `EnvState.last_update: datetime | None = None` at line 531
- [x] **Default is correct** — `None` indicates no telemetry received yet
- [x] **Consumer reads it** — EnvOverview._format_row_staleness() computes age from last_update
- [x] **Display is correct** — Icons and colors indicate staleness: green OK (<2s), yellow WARN (2-5s), red BAD (>5s)
- [x] **Thresholds applied** — 2s and 5s thresholds defined in _format_row_staleness()

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_epoch_completed_sets_last_update` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Staleness formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Stale column — should show green "OK" indicators during active training
4. Pause training (Ctrl+Z or kill process)
5. Verify Stale indicators turn yellow, then red as data ages
6. Resume training and verify indicators return to green

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED events | event | Each event updates last_update timestamp |
| System clock | system | Uses UTC for timezone-aware timestamps |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Stale column | display | Shows telemetry freshness indicator |
| Aggregate staleness mean | display | Computed for aggregate row |
| SanctumSnapshot.is_stale | property | Uses staleness_seconds (related) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Timestamps are always UTC (timezone.utc) to ensure consistent staleness calculation regardless of system timezone.
>
> **Staleness Thresholds:** The 2s/5s thresholds were chosen based on expected training loop timing. A healthy training run should emit EPOCH_COMPLETED events every ~100ms, so 2s indicates significant delay.
>
> **Visual Design:** Icons provide color-independent indication for accessibility:
> - green circle = fresh
> - yellow half-circle = warning
> - red empty circle = bad
>
> **Wiring Status:** Fully wired and operational.
