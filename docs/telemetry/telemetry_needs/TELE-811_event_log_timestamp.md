# Telemetry Record: [TELE-811] Event Log Timestamp

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-811` |
| **Name** | Event Log Timestamp |
| **Category** | `event_log` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "When did this event occur in the training timeline?"

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
| **Type** | `str` |
| **Format** | `HH:MM:SS` (24-hour time) |
| **Example** | `"14:32:05"` |
| **Default** | Current time at event creation |

### Semantic Meaning

> The timestamp captures when the telemetry event was created, formatted as a wall-clock time string for human readability. Used for:
> - Visual ordering of events in the log
> - Timestamp abbreviation (first event of each minute shows full HH:MM:SS, others show :SS)
> - Grouping aggregated events by second

### Display Behavior

| Context | Display |
|---------|---------|
| **First line** | Full format: ` HH:MM:SS ` |
| **First of new minute** | Full format: ` HH:MM:SS ` |
| **Same minute** | Abbreviated: `      :SS ` |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | `TelemetryEvent.timestamp` |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Default Factory** | `datetime.now(timezone.utc)` |
| **Conversion** | `timestamp.strftime("%H:%M:%S")` in aggregator |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `datetime` field in TelemetryEvent | `leyline/telemetry.py` (line 121) |
| **2. Aggregation** | Converted to string in `_add_event_log()` | `karn/sanctum/aggregator.py` (line 1783) |
| **3. Delivery** | `EventLogEntry.timestamp` | `karn/sanctum/schema.py` (line 1287) |

```python
# Aggregator conversion (line 1701, 1783)
timestamp = event.timestamp or datetime.now(timezone.utc)
# ...
self._event_log.append(EventLogEntry(
    timestamp=timestamp.strftime("%H:%M:%S"),
    ...
))
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EventLogEntry` |
| **Field** | `timestamp` |
| **Path from SanctumSnapshot** | `snapshot.event_log[i].timestamp` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1287 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` (lines 529-548) | Timestamp abbreviation logic for clock flow |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `TelemetryEvent.timestamp` with `_utc_now()` default
- [x] **Transport works** - Aggregator formats to string at line 1783
- [x] **Schema field exists** - `EventLogEntry.timestamp: str` at line 1287
- [x] **Default is correct** - Falls back to current UTC time
- [x] **Consumer reads it** - EventLog parses for abbreviation logic
- [x] **Display is correct** - Full/abbreviated display based on minute boundaries

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Timestamp formatting | `[ ]` |
| Widget (EventLog) | `tests/karn/sanctum/widgets/test_event_log.py` | Abbreviation logic | `[ ]` |

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TelemetryEvent.timestamp` | field | UTC datetime from emitter |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog widget | display | Timestamp column formatting |
| Event sorting | system | Clock-ordered event display |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation |

---

## 8. Notes

> **UTC Timezone:** All timestamps are captured in UTC (`timezone.utc`) at the source, then formatted as local wall-clock time strings. This ensures consistent ordering across distributed training scenarios.
>
> **Abbreviation Logic:** The EventLog widget implements "clock flow" where the first line and first event of each new minute show full `HH:MM:SS`, while subsequent events in the same minute show abbreviated `:SS` format for visual clarity and reduced clutter.
