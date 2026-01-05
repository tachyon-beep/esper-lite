# Telemetry Record: [TELE-630] Environment Status

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-630` |
| **Name** | Environment Status |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What is the current health status of this training environment - is it improving, stalled, degraded, or just starting?"

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
| **Units** | categorical enum |
| **Range** | `"initializing"`, `"healthy"`, `"excellent"`, `"stalled"`, `"degraded"` |
| **Precision** | N/A |
| **Default** | `"initializing"` |

### Semantic Meaning

> Environment status is a derived state that summarizes the training health of a single environment. Status values have the following meanings:
>
> - **initializing:** Environment has just started, no meaningful data yet (epoch 0)
> - **healthy:** Environment is making progress, accuracy improving regularly
> - **excellent:** Environment has achieved >80% accuracy and is currently improving
> - **stalled:** Environment has gone >10 epochs without accuracy improvement (with hysteresis)
> - **degraded:** Environment has dropped >1% accuracy from previous level (with hysteresis)
>
> Status uses hysteresis (3 consecutive epochs meeting condition) to prevent flicker from transient spikes.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `status == "excellent"` or `status == "healthy"` | Environment actively improving or at high performance |
| **Good** | `status == "initializing"` | Environment still warming up |
| **Warning** | `status == "stalled"` | No improvement for >10 epochs |
| **Critical** | `status == "degraded"` | Accuracy actively declining |

**Threshold Source:** `src/esper/karn/sanctum/schema.py` — `EnvState._update_status()` method (lines 685-718)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from accuracy deltas within EnvState |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState._update_status()` |
| **Line(s)** | 685-718 |

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
| **1. Emission** | Computed during `add_accuracy()` call | `schema.py` (line 650) |
| **2. Collection** | EnvState.status field updated in-place | `schema.py` (line 530) |
| **3. Aggregation** | Included in EnvState, passed through to snapshot | `aggregator.py` (line 559) |
| **4. Delivery** | Available at `snapshot.envs[env_id].status` | `schema.py` (line 530) |

```
[EpochCompletedPayload]
  --val_accuracy-->
  [SanctumAggregator._handle_epoch_completed()]
  --env.add_accuracy()-->
  [EnvState._update_status()]
  --env.status-->
  [SanctumSnapshot.envs[env_id].status]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `status` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].status` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 530 |
| **Default Value** | `"initializing"` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 852-873) | Displayed in Status column with color-coded styling and icon |
| EnvDetailScreen | (planned) | Shows status with full context |
| HistoricalEnvDetail | (planned) | Historical status at snapshot time |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `EnvState._update_status()` computes status at lines 685-718
- [x] **Transport works** — Status is computed during `add_accuracy()` call triggered by EPOCH_COMPLETED
- [x] **Schema field exists** — `EnvState.status: str = "initializing"` at line 530
- [x] **Default is correct** — `"initializing"` is appropriate for new environments
- [x] **Consumer reads it** — EnvOverview._format_status() directly accesses `env.status`
- [x] **Display is correct** — Status rendered with color and icon (excellent=green star, healthy=green circle, stalled=yellow, degraded=red)
- [x] **Thresholds applied** — Hysteresis counters prevent flicker; 10 epoch stall threshold; 1% degradation threshold

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| E2E (telemetry) | `tests/telemetry/test_tele_env_state.py` | `TestTELE630EnvStatus` (11 tests) | `[x]` |
| Unit (emitter) | `tests/karn/sanctum/test_schema.py` | `test_env_state_status_transitions` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_epoch_completed_updates_status` | `[ ]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_backend.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Status formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically or `PYTHONPATH=src uv run python -m esper.karn.sanctum`)
3. Observe EnvOverview Status column — should show "INIT" at start, then "OK" or "EXCL" as training progresses
4. Verify color coding: green (excellent/healthy), yellow (stalled), red (degraded), dim (initializing)
5. Wait 10+ epochs without improvement to verify "STALL" status appears (requires hysteresis threshold)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED events | event | Triggers add_accuracy() which updates status |
| val_accuracy | metric | Status is computed from accuracy changes |
| epochs_since_improvement | counter | Used for stall detection |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Status column | display | Color-coded status indicator |
| Visual quieting logic | display | Dimmed rows for OK/STALLED status (unless top 5 accuracy) |
| _format_momentum_epochs | display | Momentum coloring changes based on status |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Status uses hysteresis (3 consecutive epochs meeting condition) to prevent status flicker from transient accuracy variations. This means a single bad epoch won't immediately show "degraded" status.
>
> **Positive Status Updates:** Unlike negative status transitions (stalled, degraded), positive transitions (to healthy, excellent) happen immediately without hysteresis. Immediate positive feedback is more valuable than preventing false positives.
>
> **Wiring Status:** Fully wired and operational. Status is computed in-place during accuracy updates, no external telemetry event required.
