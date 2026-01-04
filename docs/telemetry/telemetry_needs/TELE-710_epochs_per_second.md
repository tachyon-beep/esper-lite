# Telemetry Record: [TELE-710] Epochs per Second

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-710` |
| **Name** | Epochs per Second |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How fast is training progressing through the episode/epoch timeline? Is throughput degrading?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | epochs per second (e/s) |
| **Range** | `[0.0, inf)` — non-negative |
| **Precision** | 1 decimal place for display (e.g., "0.8e/s") |
| **Default** | `0.0` |

### Semantic Meaning

> Throughput metric computed as: `epochs_per_second = (current_episode * max_epochs) / elapsed_seconds`
>
> Measures the rate at which the training process progresses through episode/epoch structure.
> A metric for training speed and system efficiency (hardware + implementation).
> Higher is better (faster training). Used alongside batches_per_hour for dual throughput metrics.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `epochs_per_second > 0.5` | Normal training throughput |
| **Warning** | `0.1 < epochs_per_second <= 0.5` | Slow training, investigate |
| **Critical** | `epochs_per_second <= 0.1` | Training stalled or severely degraded |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed at BATCH_EPOCH_COMPLETED event, not emitted by simic |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_batch_epoch_completed()` (throughput calculation) |
| **Line(s)** | 1216-1222 |

```python
# Calculate throughput
now = time.time()
elapsed = now - self._start_time
if elapsed > 0:
    total_epochs = self._current_episode * self._max_epochs
    self._vitals.epochs_per_second = total_epochs / elapsed
    self._vitals.batches_per_hour = (self._batches_completed / elapsed) * 3600
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Not emitted by simic; computed in aggregator | `aggregator.py` |
| **2. Collection** | `_handle_batch_epoch_completed()` triggered on each BATCH_EPOCH_COMPLETED event | `aggregator.py` line 1343 |
| **3. Aggregation** | Computed from runtime and episode progress, stored in `_vitals` | `aggregator.py` line 1221 |
| **4. Delivery** | Delivered via `SanctumSnapshot.vitals.epochs_per_second` | `aggregator.py` line 570 |

```
[BATCH_EPOCH_COMPLETED event] --> [_handle_batch_epoch_completed()] --> [calculate elapsed/episodes] --> [_vitals.epochs_per_second] --> [SanctumSnapshot.vitals]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SystemVitals` |
| **Field** | `epochs_per_second` |
| **Path from SanctumSnapshot** | `snapshot.vitals.epochs_per_second` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1034 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Line 274: Displayed as "X.Xe/s" throughput metric in header |
| EsperStatus | `widgets/esper_status.py` | Line 88: Displayed in status table as "Epochs/sec: X.XX" |
| Sanctum Debug Output | `app.py` | Line 196: Debug telemetry output showing "Epochs/sec: X.XX" |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed in aggregator from runtime and episode counts
- [x] **Transport works** — Updated on BATCH_EPOCH_COMPLETED, included in snapshot
- [x] **Schema field exists** — `SystemVitals.epochs_per_second: float = 0.0` (line 1034)
- [x] **Default is correct** — 0.0 is appropriate (no progress until training starts)
- [x] **Consumer reads it** — Both RunHeader and EsperStatus read `vitals.epochs_per_second`
- [x] **Display is correct** — RunHeader formats as "0.8e/s", EsperStatus as "0.00"
- [ ] **Thresholds applied** — Health thresholds exist but NOT actively used for coloring/alerts

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (computation) | | | `[ ]` |
| Integration (flow) | | | `[ ]` |
| Visual (TUI snapshot) | | | `[ ]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader "Segment 7: Throughput" showing "X.Xe/s" metric
4. Watch value update after each BATCH_EPOCH_COMPLETED event
5. Verify EsperStatus panel (if visible) shows same metric in status table
6. Verify metric is non-zero after first batch completes (elapsed time > 0)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `BATCH_EPOCH_COMPLETED` event | event | Triggers throughput calculation; requires `episodes_completed` field |
| Training start time | context | Captured at TRAINING_STARTED in `_start_time` |
| Episode count | context | `_current_episode` updated by BATCH_EPOCH_COMPLETED payload |
| Max epochs per episode | config | `_max_epochs` captured at TRAINING_STARTED |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader widget | display | Formats and shows as throughput metric in header bar |
| EsperStatus widget | display | Shows in status table alongside other vitals |
| User monitoring | system | Used to assess training speed and detect stalls |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation, traced from schema to display |
| | | Verified wiring complete but thresholds not actively used |

---

## 8. Notes

> **Design Decision:** Throughput is computed on-demand in the aggregator rather than emitted from simic.
> This is appropriate because it's a derived metric (runtime-based) not a primary training signal.
> Computation happens at BATCH_EPOCH_COMPLETED to match the granularity of episode boundaries.
>
> **Formula:** `epochs_per_second = (current_episode * max_epochs) / (now - start_time)`
> This assumes `max_epochs` is constant per episode, which is true for the standard training loop.
>
> **Known Issue:** Health thresholds are defined (0.1 critical, 0.5 warning) but are NOT actively applied
> for color coding or alerts. RunHeader and EsperStatus show the raw metric without threshold-based styling.
> This is a minor wiring gap: the metric is fully functional but missing visual feedback for degraded throughput.
>
> **Display Format Consistency:** RunHeader uses "0.8e/s" format (1 decimal), while EsperStatus uses "0.00"
> (2 decimals). Both are readable but inconsistent. Consider standardizing to 1 decimal for all displays.
>
> **Telemetry Gap:** The metric resets to 0.0 at TRAINING_STARTED (new start_time), which is correct.
> However, there is no special handling for pause/resume scenarios. If training is paused and resumed,
> elapsed time is continuous (good for accuracy but may show artificially low throughput during pauses).
