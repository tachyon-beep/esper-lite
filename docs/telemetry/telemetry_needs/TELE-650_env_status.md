# Telemetry Record: [TELE-650] Environment Status

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-650` |
| **Name** | Environment Status |
| **Category** | `environment` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is this environment making progress, or is it stalled/degraded and needs intervention?"

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
| **Type** | `str` |
| **Units** | categorical label |
| **Range** | `"initializing"`, `"healthy"`, `"excellent"`, `"stalled"`, `"degraded"` |
| **Precision** | N/A (string enum) |
| **Default** | `"initializing"` |

### Semantic Meaning

> Per-environment status indicating health state, computed from accuracy trends:
>
> - **initializing**: Training not yet started or first epoch not completed
> - **healthy**: Environment is making normal progress (epoch > 0, accuracy improving)
> - **excellent**: Environment has high accuracy (> 80%) and just improved
> - **stalled**: No accuracy improvement for > 10 epochs (after hysteresis check)
> - **degraded**: Accuracy dropped > 1% for 3 consecutive epochs (after hysteresis check)
>
> Status transitions use hysteresis (3 consecutive epochs meeting condition) to prevent flicker from transient spikes.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `status in ("healthy", "excellent")` | Environment progressing normally |
| **Warning** | `status == "stalled"` | No improvement for > 10 epochs, monitor for intervention |
| **Critical** | `status == "degraded"` | Accuracy regressing, may need rollback or hyperparameter adjustment |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EnvState._update_status() method, triggered by add_accuracy() |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState._update_status()` |
| **Line(s)** | ~685-718 |

```python
def _update_status(self, prev_acc: float, curr_acc: float) -> None:
    """Update env status based on metrics with hysteresis."""
    HYSTERESIS_THRESHOLD = 3

    # Check stall condition (>10 epochs without improvement)
    if self.epochs_since_improvement > 10:
        self.stall_counter += 1
        if self.stall_counter >= HYSTERESIS_THRESHOLD:
            self.status = "stalled"
    else:
        self.stall_counter = 0

    # Check degraded condition (accuracy dropped >1%)
    if curr_acc < prev_acc - 1.0:
        self.degraded_counter += 1
        if self.degraded_counter >= HYSTERESIS_THRESHOLD:
            self.status = "degraded"
    elif curr_acc > prev_acc:
        self.degraded_counter = 0

    # Positive status updates (no hysteresis needed)
    if self.epochs_since_improvement == 0:
        if curr_acc > 80.0:
            self.status = "excellent"
        elif self.current_epoch > 0:
            self.status = "healthy"
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.on_epoch_completed()` emits EPOCH_COMPLETED with val_accuracy | `simic/telemetry/emitters.py` |
| **2. Collection** | TelemetryHub receives EPOCH_COMPLETED event | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._handle_epoch_completed()` calls `env.add_accuracy()` which calls `_update_status()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Status computed and stored directly in `EnvState.status` field | `karn/sanctum/schema.py` |

```
[vectorized.py] --on_epoch_completed()--> [TelemetryEmitter] --EPOCH_COMPLETED-->
[Aggregator._handle_epoch_completed()] --env.add_accuracy()--> [EnvState._update_status()] --> [EnvState.status]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `status` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].status` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 530 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` | Status column with color-coded icons (lines 852-873) |
| EnvOverview | `widgets/env_overview.py` | Row filtering by status text (line 99) |
| EnvOverview | `widgets/env_overview.py` | Visual quieting for healthy/stalled rows (line 258) |
| EnvOverview | `widgets/env_overview.py` | Epochs-since-improvement formatting (line 813) |
| EnvDetailScreen | `widgets/env_detail_screen.py` | Header status display with color (lines 437-438) |
| AnomalyStrip | `widgets/anomaly_strip.py` | Stalled/degraded environment counts (lines 88-91) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EPOCH_COMPLETED event includes val_accuracy
- [x] **Transport works** — Aggregator handles event and calls add_accuracy()
- [x] **Schema field exists** — `EnvState.status: str = "initializing"`
- [x] **Default is correct** — "initializing" appropriate before first epoch
- [x] **Consumer reads it** — EnvOverview, EnvDetailScreen, AnomalyStrip all access status
- [x] **Display is correct** — Status renders with icons and colors
- [x] **Thresholds applied** — Color coding matches status values

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (status transitions) | `tests/karn/sanctum/test_schema.py` | `test_env_state_stalled_hysteresis` | `[x]` |
| Unit (degraded hysteresis) | `tests/karn/sanctum/test_schema.py` | `test_env_state_degraded_hysteresis` | `[x]` |
| Unit (counter persistence) | `tests/karn/sanctum/test_schema.py` | `test_degraded_counter_persists_on_stable` | `[x]` |
| Unit (aggregator wiring) | `tests/karn/sanctum/test_backend.py` | `test_env_status_updates_from_epoch_completed` | `[x]` |
| Unit (anomaly detection) | `tests/karn/sanctum/test_anomaly_strip.py` | `test_stalled_envs_are_counted` | `[x]` |
| Visual (TUI snapshot) | - | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe EnvOverview table Status column
4. Verify status shows "initializing" initially, then "healthy"/"excellent"
5. Observe environments that stall for > 10 epochs show "stalled" status
6. Check AnomalyStrip shows count of stalled/degraded environments

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED event | event | Provides val_accuracy needed for status calculation |
| `EnvState.epochs_since_improvement` | field | Used to detect stall condition (> 10 epochs) |
| `EnvState.accuracy_history` | field | Previous accuracy used to detect degradation (> 1% drop) |
| Hysteresis counters | field | `stall_counter`, `degraded_counter` prevent status flicker |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| AnomalyStrip stalled/degraded counts | display | Aggregates status across all envs for quick health view |
| EnvOverview row dimming | display | Uses status to visually quiet healthy/stalled rows |
| EnvOverview filtering | display | Users can filter env list by status text |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial documentation |

---

## 8. Notes

> **Design Decision:** Status uses hysteresis (3 consecutive epochs) to prevent visual flicker from transient accuracy spikes. This is especially important for the TUI where rapid status changes would be distracting.
>
> **Note on Values:** The specification mentioned values "initializing", "running", "stalled", "degraded", but the actual implementation uses "initializing", "healthy", "excellent", "stalled", "degraded". There is no "running" status - the code uses "healthy" for normal operation and "excellent" for high-accuracy (> 80%) states.
>
> **Status Reset:** Status resets to "initializing" at episode start (in aggregator's episode_started handler, line ~1297).
>
> **Positive Status Immediacy:** While negative statuses (stalled, degraded) require hysteresis, positive statuses (healthy, excellent) trigger immediately on improvement. This provides immediate positive feedback while protecting against false alarms.
