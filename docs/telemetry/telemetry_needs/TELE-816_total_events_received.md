# Telemetry Record: [TELE-816] Total Events Received

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-816` |
| **Name** | Total Events Received |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many telemetry events has the system received since training started, and at what rate are events arriving?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging event flow)
- [ ] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every snapshot update)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `int` |
| **Units** | count (cumulative) |
| **Range** | `[0, ∞)` |
| **Precision** | Exact integer |
| **Default** | `0` |

### Semantic Meaning

> Total Events Received is a monotonically increasing counter of all telemetry events processed by the aggregator since training started. This includes:
>
> - Seed lifecycle events (germination, stage changes, fossilization, pruning)
> - PPO update events
> - Batch/epoch completion events
> - Reward computation events
> - Training started/stopped events
>
> The counter is used to:
> 1. Display total event count in EventLog border title
> 2. Compute events/sec rate via rolling window derivative
>
> A healthy training run typically sees 10-100+ events/sec depending on batch size and number of environments.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `events/sec > 1` | Events flowing normally |
| **Warning** | `events/sec < 1 for 5s` | Event flow slowing |
| **Critical** | `events/sec = 0 for 10s` | No events (possible disconnect) |

**Note:** Thresholds apply to the derived `events_per_second` rate, not the raw counter.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregator event counter |
| **File** | `src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._process_event()` |
| **Line(s)** | Incremented on each event processed |

```python
# Incremented for every event processed by aggregator
self._total_events_received += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Collection** | Aggregator increments counter on each event | `aggregator.py` |
| **2. Storage** | `SanctumSnapshot.total_events_received` | `schema.py` |
| **3. Delivery** | Snapshot passed to widgets | `aggregator.get_snapshot()` |

```
[Telemetry Event Received]
  --> [aggregator._process_event()]
  --> [self._total_events_received += 1]
  --> [SanctumSnapshot.total_events_received]
  --> [EventLog widget]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `total_events_received: int = 0` |
| **Path from SanctumSnapshot** | `snapshot.total_events_received` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` (lines 262-275) | Border title displays total count and derived events/sec rate |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Aggregator increments `_total_events_received` on each event
- [x] **Transport works** — Counter included in `get_snapshot()` return value
- [x] **Schema field exists** — `SanctumSnapshot.total_events_received: int = 0`
- [x] **Default is correct** — `0` before first event received
- [x] **Consumer reads it** — EventLog accesses `snapshot.total_events_received`
- [x] **Display is correct** — Formatted with commas in border title: `EVENTS 1,234`
- [ ] **Thresholds applied** — No explicit thresholds (derived rate used informally)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | Snapshot default values | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | Event counter increment | `[ ]` |
| Widget (display) | `tests/telemetry/test_tele_event_log.py` | Border title format | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or run separately)
3. Observe EventLog panel border title
4. Verify format: `EVENTS 1,234 (45.2/s) [click for detail]`
5. Verify count increases during active training
6. Verify rate calculation appears reasonable (10-100+/s typical)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Telemetry event stream | runtime | Requires events to be emitted from training |
| Aggregator event loop | runtime | Must be processing events |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog border title | widget | Displays total count |
| Events per second rate | derived | Rolling window derivative of this counter |
| Drip-feed pacing | widget | Rate used to pace event display |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - gap identified in EventLog audit |

---

## 8. Notes

> **Design Decision:** This is a cumulative counter, not a gauge. It only increases, never decreases. This makes it suitable for computing rates via windowed derivatives.
>
> **Rate Calculation:** EventLog computes `events_per_second` using a 5-second rolling window:
> ```python
> # Simplified rate calculation
> dt = samples[-1].time - samples[0].time
> events_per_second = (samples[-1].count - samples[0].count) / dt
> ```
>
> **Display Format:** The count is formatted with commas for readability: `1,234` not `1234`. The rate is shown with one decimal place: `45.2/s`.
>
> **Drip-Feed Integration:** The computed `events_per_second` rate is used to pace drip-feed display of individual events, creating smooth scrolling rather than jarring bursts.
>
> **Wiring Status:** FULLY WIRED. Aggregator counter -> snapshot -> EventLog display all connected and working.
