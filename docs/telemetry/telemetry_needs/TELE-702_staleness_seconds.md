# Telemetry Record: [TELE-702] Snapshot Staleness

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-702` |
| **Name** | Snapshot Staleness |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How fresh is the Sanctum snapshot data? Is the training backend still sending updates, or has communication stalled?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging connection issues)
- [ ] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every snapshot poll, ~250ms)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | seconds |
| **Range** | `[0.0, inf)` |
| **Precision** | 1 decimal place for display |
| **Default** | `float('inf')` (unavailable/disconnected) |

### Semantic Meaning

> Staleness measures the elapsed time since the last telemetry event was received by the aggregator. Computed as: `now - _last_event_ts` in the aggregator's `get_snapshot()` method.
>
> Zero staleness = snapshot captures the live state at this moment.
> Growing staleness = training backend has stopped emitting events (potential crash, hang, or networking issue).
> Infinite staleness = no events received yet (connection not established).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy (LIVE)** | `staleness < 2.0s` | Backend actively sending events |
| **Warning (SLOW)** | `2.0s <= staleness < 5.0s` | Backend is delayed but still responsive |
| **Critical (STALE)** | `staleness >= 5.0s` | Backend communication has stalled |

**Threshold Source:** `RunHeader._get_connection_status()` lines 80-85

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Measured in aggregator when snapshot is requested |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._get_snapshot_unlocked()` |
| **Line(s)** | 400-403 |

```python
# Compute staleness (elapsed time since last event)
now = time.time()
now_dt = datetime.now(timezone.utc)
staleness = now - self._last_event_ts if self._last_event_ts else float("inf")
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Computed at snapshot generation | `aggregator.py` line 402 |
| **2. Collection** | Part of SanctumSnapshot dataclass | `schema.py` line 1332 |
| **3. Aggregation** | No transformation, direct assignment | `aggregator.py` line 553 |
| **4. Delivery** | Passed to RunHeader widget via snapshot | `app.py` polling loop |

```
[Training Thread]
    ↓ (events with timestamps)
[SanctumAggregator._last_event_ts updated]
    ↓ (every 250ms UI poll)
[aggregator._get_snapshot_unlocked() computes now - _last_event_ts]
    ↓
[SanctumSnapshot.staleness_seconds field]
    ↓
[RunHeader._get_connection_status() reads value]
    ↓
[Display: "● LIVE" / "◐ SLOW (3s)" / "○ STALE (7s)"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `staleness_seconds` |
| **Path from SanctumSnapshot** | `snapshot.staleness_seconds` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1332 |

```python
# From schema.py line 1332
staleness_seconds: float = float('inf')
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Connection status indicator + staleness display |
| n/a | n/a | Status coloring: <2s=green LIVE, 2-5s=yellow SLOW, >=5s=red STALE |

**Implementation:** `RunHeader._get_connection_status()` lines 65-85 uses thresholds to determine icon (●/◐/○) and color.

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed in `_get_snapshot_unlocked()` from `_last_event_ts`
- [x] **Transport works** — Assigned directly to snapshot field at line 553
- [x] **Schema field exists** — `SanctumSnapshot.staleness_seconds: float = float('inf')` at line 1332
- [x] **Default is correct** — `float('inf')` indicates unavailable/disconnected state
- [x] **Consumer reads it** — `RunHeader._get_connection_status()` reads `snapshot.staleness_seconds` at line 79
- [x] **Display is correct** — Shows as icon + status word (LIVE/SLOW/STALE) with optional staleness count
- [x] **Thresholds applied** — Hardcoded in `RunHeader._get_connection_status()` (2.0s, 5.0s)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_staleness_computation` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_staleness_reaches_ui` | `[ ]` |
| Visual (snapshot state) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader top-left corner showing connection status
4. Verify "● LIVE" indicator when events flowing normally
5. Force backend pause to trigger "◐ SLOW (Ns)" or "○ STALE (Ns)" display
6. Verify threshold transitions occur at 2s and 5s boundaries

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Event timestamp | infrastructure | Every telemetry event must carry `timestamp` field for `_last_event_ts` to update |
| Time source | OS | `time.time()` must be synchronized (system clock) |
| Aggregator locking | concurrency | Safe access to `_last_event_ts` requires `_lock` protection |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader status indicator | display | Primary consumer, shows connection health |
| SanctumSnapshot.is_stale property | schema | Checks if `staleness_seconds > 5.0` for data freshness assessment |
| UI polling loop | system | Trigger frequency (250ms) affects staleness detection latency |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - traced metric from aggregator to RunHeader widget |

---

## 8. Notes

### Design Decision

Staleness is computed **on-demand** during snapshot generation rather than tracked continuously. This approach:
- Avoids background threads or timers (simpler concurrency model)
- Reflects actual data freshness at the moment the snapshot is requested
- Scales naturally with snapshot poll frequency (250ms default)

### Implementation Detail

The `_last_event_ts` is updated in `_process_event_unlocked()` (line 316-320) every time an event arrives, using the event's timestamp if available, otherwise `time.time()`. When `get_snapshot()` is called (from UI poll loop), it computes elapsed time in `_get_snapshot_unlocked()` (line 402).

This design tolerates:
- Burst arrivals (multiple events → single timestamp update)
- Time skew in event timestamps (fallback to system clock)
- Long processing delays (staleness reflects receipt time, not process completion time)

### Known Issue

**None documented** — metric is fully wired and actively consumed. No gaps found in tracing.

### Threshold Justification

- **2 seconds (SLOW threshold):** Aggregator polls at ~250ms, so 2s = 8 missed polls. Indicates non-trivial backend lag.
- **5 seconds (STALE threshold):** Indicates likely backend crash/hang rather than transient network delay. Triggers visual alert (red icon).

These thresholds match Overwatch web dashboard behavior (as mentioned in schema comment at line 1396: "STALENESS THRESHOLD: 5 seconds (matches Overwatch behavior)").

### Future Improvement

Consider per-event-type staleness tracking (e.g., "PPO stale" vs "EPOCH stale") to distinguish which backend subsystems are responsive. Current metric provides only overall connection health.
