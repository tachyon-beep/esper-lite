# Telemetry Record: [TELE-711] Batches per Hour

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-711` |
| **Name** | Batches per Hour |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many training batches are being completed per hour? Is throughput sustainable or degrading?"

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
| **Units** | batches per hour (b/h) |
| **Range** | `[0.0, inf)` — non-negative |
| **Precision** | 1 decimal place for display (e.g., "2.1b/m" when divided by 60) |
| **Default** | `0.0` |

### Semantic Meaning

> Throughput metric computed as: `batches_per_hour = (batches_completed / elapsed_seconds) * 3600`
>
> Measures the rate at which training batches are processed. Complements `epochs_per_second` for
> batch-level throughput tracking. Higher values indicate faster batch processing. Used to monitor
> system efficiency and detect performance degradation (memory issues, GPU throttling, etc.).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `batches_per_hour > 180.0` | Normal training throughput |
| **Warning** | `60.0 < batches_per_hour <= 180.0` | Slow batch processing, investigate |
| **Critical** | `batches_per_hour <= 60.0` | Severely degraded batch processing |

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
| **3. Aggregation** | Computed from runtime and batch counter, stored in `_vitals` | `aggregator.py` line 1222 |
| **4. Delivery** | Delivered via `SanctumSnapshot.vitals.batches_per_hour` | `aggregator.py` line 570 |

```
[BATCH_EPOCH_COMPLETED event] --> [_handle_batch_epoch_completed()] --> [calculate elapsed/batches] --> [_vitals.batches_per_hour] --> [SanctumSnapshot.vitals]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SystemVitals` |
| **Field** | `batches_per_hour` |
| **Path from SanctumSnapshot** | `snapshot.vitals.batches_per_hour` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1035 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Line 275: Converted to per-minute (bpm = batches_per_hour / 60) and displayed as "X.Xb/m" throughput metric in header (line 200) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed in aggregator from runtime and batch counter
- [x] **Transport works** — Updated on BATCH_EPOCH_COMPLETED, included in snapshot
- [x] **Schema field exists** — `SystemVitals.batches_per_hour: float = 0.0` (line 1035)
- [x] **Default is correct** — 0.0 is appropriate (no progress until training starts)
- [x] **Consumer reads it** — RunHeader reads `vitals.batches_per_hour` and converts to per-minute
- [x] **Display is correct** — RunHeader formats as "X.Xb/m" (batches per minute, 1 decimal place)
- [ ] **Thresholds applied** — Health thresholds exist but NOT actively used for coloring/alerts

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (computation) | `tests/karn/sanctum/test_schema.py` | `test_system_vitals_tracks_throughput` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | Line 231-237 | `[x]` |
| Integration (flow) | | | `[ ]` |
| Visual (TUI snapshot) | `tests/karn/sanctum/test_run_info_screen.py` | Line 58-63 | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader "Segment 7: Throughput" showing "X.Xb/m" metric (batches per minute)
4. Watch value update after each BATCH_EPOCH_COMPLETED event
5. Verify metric is non-zero after first batch completes (elapsed time > 0)
6. Verify calculation: batches_per_minute ≈ displayed value

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `BATCH_EPOCH_COMPLETED` event | event | Triggers throughput calculation; increments `_batches_completed` counter |
| Training start time | context | Captured at TRAINING_STARTED in `_start_time` (line 617) |
| Batch counter | context | `_batches_completed` incremented by BATCH_EPOCH_COMPLETED payload (line 1194) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader widget | display | Formats and shows as throughput metric in header bar (converted to per-minute) |
| User monitoring | system | Used to assess batch processing speed and detect stalls or GPU throttling |
| TELE-710 (epochs_per_second) | telemetry | Computed in same handler; batches_per_hour and epochs_per_second track different throughput aspects |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation, traced from schema to display |

---

## 8. Notes

> **Design Decision:** Batches per hour is computed on-demand in the aggregator rather than emitted from simic.
> This is appropriate because it's a derived metric (runtime-based) not a primary training signal.
> Computation happens at BATCH_EPOCH_COMPLETED to match the granularity of batch boundaries.
>
> **Formula:** `batches_per_hour = (batches_completed / elapsed_seconds) * 3600`
> The counter `_batches_completed` is incremented on each BATCH_EPOCH_COMPLETED event (aggregator line 1194).
>
> **Display Format Conversion:** RunHeader converts to batches per *minute* for display readability:
> `bpm = batches_per_hour / 60` (line 275), then formats as "X.Xb/m" (line 200).
> This is appropriate for typical training speeds which would show as 100+ batches/hour.
>
> **Known Issue:** Health thresholds are defined (60 critical, 180 warning) but are NOT actively applied
> for color coding or alerts. RunHeader shows the raw metric without threshold-based styling.
> This is a minor wiring gap: the metric is fully functional but missing visual feedback for degraded throughput.
>
> **Wiring Status:** This metric is fully wired end-to-end from aggregator to display. The only consumer
> is RunHeader, which displays it alongside epochs_per_second for dual throughput metrics.
> Unlike epochs_per_second, there is no secondary consumer (no EsperStatus or debug output).
>
> **Telemetry Gap:** The metric resets to 0.0 at TRAINING_STARTED (new start_time), which is correct.
> However, there is no special handling for pause/resume scenarios. If training is paused and resumed,
> elapsed time is continuous (good for accuracy but may show artificially low throughput during pauses).
>
> **Counter Accuracy:** The `_batches_completed` counter is incremented unconditionally on each
> BATCH_EPOCH_COMPLETED event, so the count is accurate. The metric is valid after the first batch
> completes (elapsed time > 0), at which point batches_per_hour becomes non-zero.
