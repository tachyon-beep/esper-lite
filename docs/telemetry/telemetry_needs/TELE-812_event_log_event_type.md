# Telemetry Record: [TELE-812] Event Log Event Type

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-812` |
| **Name** | Event Log Event Type |
| **Category** | `event_log` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What kind of telemetry event occurred?"

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
| **Source Enum** | `TelemetryEventType` |
| **Format** | SCREAMING_SNAKE_CASE |
| **Example** | `"SEED_GERMINATED"`, `"PPO_UPDATE_COMPLETED"` |

### Semantic Meaning

> The event type identifies the category of telemetry event, determining:
> - Which handler processes the event
> - What metadata fields are expected
> - How the event is displayed (individual vs aggregated)
> - What color coding to apply

### Supported Event Types

| Event Type | Color | Display Mode | Description |
|------------|-------|--------------|-------------|
| `SEED_GERMINATED` | `bright_yellow` | Individual | New seed module created |
| `SEED_STAGE_CHANGED` | `bright_white` | Individual | Lifecycle stage transition |
| `SEED_GATE_EVALUATED` | `bright_white` | Individual | Gate check result |
| `SEED_FOSSILIZED` | `bright_green` | Individual | Seed permanently fused |
| `SEED_PRUNED` | `bright_red` | Individual | Seed removed |
| `PPO_UPDATE_COMPLETED` | `bright_magenta` | Individual | Policy gradient update |
| `BATCH_EPOCH_COMPLETED` | `bright_blue` | Individual | Episode/batch boundary |
| `TRAINING_STARTED` | `bright_green` | Individual | Run initialization |
| `REWARD_COMPUTED` | `dim` | Aggregated | Per-step reward |
| `EPOCH_COMPLETED` | `bright_blue` | Aggregated | Per-env epoch |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | `TelemetryEvent.event_type` (TelemetryEventType enum) |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Enum Definition** | Lines 61-113 |
| **Conversion** | `event.event_type.name` (enum to string) |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEventType` enum | `leyline/telemetry.py` (line 120) |
| **2. Aggregation** | Converted to string in `process_event()` | `karn/sanctum/aggregator.py` (line 327) |
| **3. Delivery** | `EventLogEntry.event_type` | `karn/sanctum/schema.py` (line 1288) |

```python
# Aggregator conversion (line 327)
event_type = event.event_type.name
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EventLogEntry` |
| **Field** | `event_type` |
| **Path from SanctumSnapshot** | `snapshot.event_log[i].event_type` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1288 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` (lines 31-45, 47-57) | Color mapping, individual vs aggregated routing |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `TelemetryEventType` enum with all event types
- [x] **Transport works** - Aggregator converts enum to string
- [x] **Schema field exists** - `EventLogEntry.event_type: str` at line 1288
- [x] **Default is correct** - N/A (required field)
- [x] **Consumer reads it** - EventLog uses for color mapping and display mode
- [x] **Display is correct** - Color-coded with proper individual/aggregated handling

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Event type routing | `[ ]` |
| Widget (EventLog) | `tests/karn/sanctum/widgets/test_event_log.py` | Color mapping | `[ ]` |

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TelemetryEventType` enum | contract | Defines valid event types |
| Event emission | system | Each emitter sets the event type |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog color mapping | display | `_EVENT_COLORS` dict lookup |
| Individual/aggregated routing | display | `_INDIVIDUAL_EVENTS` set membership |
| Aggregation key | system | Groups events by (timestamp, event_type) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation |

---

## 8. Notes

> **Individual vs Aggregated Events:** The `_INDIVIDUAL_EVENTS` set in `event_log.py` (lines 48-57) determines which event types are shown with full metadata and which are aggregated per-second. Seed lifecycle events, PPO updates, and batch completions are individual; reward and epoch events are aggregated.
>
> **Color Mapping:** The `_EVENT_COLORS` dict (lines 31-45) maps event types to Rich color styles for visual differentiation in the TUI.
>
> **Future Event Types:** New event types should be added to both `TelemetryEventType` enum and the appropriate display mappings in `event_log.py`.
