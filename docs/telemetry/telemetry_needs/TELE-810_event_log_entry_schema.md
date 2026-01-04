# Telemetry Record: [TELE-810] Event Log Entry Schema

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-810` |
| **Name** | Event Log Entry Schema |
| **Category** | `event_log` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What structured telemetry events have occurred during this training run, including lifecycle transitions, PPO updates, and batch completions?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every event)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `dataclass` (EventLogEntry) |
| **Fields** | 7 fields (see below) |
| **Collection** | `list[EventLogEntry]` in SanctumSnapshot |
| **Max Size** | 100 entries (deque with maxlen) |

### Field Summary

| Field | Type | Description | TELE ID |
|-------|------|-------------|---------|
| `timestamp` | `str` | ISO timestamp (HH:MM:SS) | TELE-811 |
| `event_type` | `str` | Event type enum value | TELE-812 |
| `env_id` | `int \| None` | Environment ID (None for global) | TELE-813 |
| `message` | `str` | Generic message text | TELE-814 |
| `metadata` | `dict` | Rich inline metadata | TELE-815 |
| `episode` | `int` | Episode number for grouping | - |
| `relative_time` | `str` | "(2s ago)" relative time string | - |

### Event Types Supported

| Event Type | Domain | Description |
|------------|--------|-------------|
| `SEED_GERMINATED` | Kasmina | New seed module created |
| `SEED_STAGE_CHANGED` | Kasmina | Seed lifecycle transition |
| `SEED_GATE_EVALUATED` | Kasmina | Gate check pass/fail |
| `SEED_FOSSILIZED` | Kasmina | Seed permanently fused |
| `SEED_PRUNED` | Kasmina | Seed removed |
| `PPO_UPDATE_COMPLETED` | Simic | Policy gradient update |
| `BATCH_EPOCH_COMPLETED` | Simic | Episode/batch boundary |
| `TRAINING_STARTED` | Simic | Run initialization |
| `REWARD_COMPUTED` | Simic | Per-step reward (aggregated) |
| `EPOCH_COMPLETED` | Simic | Per-env epoch (aggregated) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Various emitters via TelemetryEvent |
| **Files** | `simic/telemetry/emitters.py`, `kasmina/slot.py` |
| **Central Emission** | `esper.leyline.telemetry.TelemetryEvent` |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TelemetryEvent with typed payload | Various emitters |
| **2. Collection** | Event queue in training thread | `simic/training/vectorized.py` |
| **3. Aggregation** | `SanctumAggregator._add_event_log()` | `karn/sanctum/aggregator.py` (line 1695-1790) |
| **4. Delivery** | `snapshot.event_log` list | `karn/sanctum/schema.py` (line 1350) |

```
[Various Emitters]
  --TelemetryEvent-->
  [Training Thread Event Queue]
  --process_event()-->
  [SanctumAggregator._add_event_log()]
  --EventLogEntry-->
  [SanctumSnapshot.event_log]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EventLogEntry` |
| **Path from SanctumSnapshot** | `snapshot.event_log[i]` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1278-1295 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` | Scrolling event log with rich metadata display |
| EventLogDetail | `widgets/event_log_detail.py` | Modal view of raw event data |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Multiple emitters via TelemetryEvent
- [x] **Transport works** - Events flow through aggregator
- [x] **Schema field exists** - `EventLogEntry` dataclass at line 1278
- [x] **Default is correct** - Empty list via `field(default_factory=list)`
- [x] **Consumer reads it** - EventLog widget processes `snapshot.event_log`
- [x] **Display is correct** - Rich inline formatting with timestamps

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Event log population | `[ ]` |
| Widget (EventLog) | `tests/karn/sanctum/widgets/test_event_log.py` | Event rendering | `[ ]` |
| Integration | `tests/karn/sanctum/test_backend.py` | End-to-end flow | `[ ]` |

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TelemetryEvent emission | system | All domain emitters |
| Typed payloads | contracts | `leyline/telemetry.py` payload dataclasses |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog widget | display | Renders scrolling log |
| EventLogDetail modal | display | Shows raw event data |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation |

---

## 8. Notes

> **Design Decision:** EventLogEntry uses a "generic message + structured metadata" pattern. The `message` field contains human-readable text suitable for log rollup (e.g., "Germinated", "Stage changed"), while specific values like slot_id, blueprint, and improvement go in the `metadata` dict. This enables proper event grouping while preserving rich detail for the modal view.
>
> **Individual vs Aggregated Events:** The EventLog widget distinguishes between "individual" events (SEED_*, PPO_*, BATCH_*) that are shown with full metadata, and "aggregated" events (EPOCH_COMPLETED, REWARD_COMPUTED) that are rolled up per-second with count indicators.
>
> **Drip-Feed Display:** Individual events are queued and drip-fed to the display at the observed events/second rate to provide smooth scrolling during high-throughput training.
