# Telemetry Record: [TELE-814] Event Log Message

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-814` |
| **Name** | Event Log Message |
| **Category** | `event_log` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the human-readable summary of this event?"

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
| **Format** | Short, generic description |
| **Max Length** | ~50 characters recommended |
| **Example** | `"Germinated"`, `"Stage changed"`, `"PPO update"` |

### Semantic Meaning

> The message provides a **generic**, human-readable description of the event. It is intentionally kept generic (not including specific values like slot_id or improvement %) to enable proper event rollup and aggregation. Specific values are stored in the `metadata` dict (TELE-815).

### Message Templates by Event Type

| Event Type | Message Template |
|------------|-----------------|
| `SEED_GERMINATED` | `"Germinated"` |
| `SEED_STAGE_CHANGED` | `"Stage changed"` |
| `SEED_GATE_EVALUATED` | `"Gate evaluated"` |
| `SEED_FOSSILIZED` | `"Fossilized"` |
| `SEED_PRUNED` | `"Pruned"` |
| `PPO_UPDATE_COMPLETED` | `"PPO update"` or `"PPO skipped"` |
| `BATCH_EPOCH_COMPLETED` | `"Batch complete"` |
| `TRAINING_STARTED` | Event message or event type |
| Other | `event.message or event_type` |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Generated in aggregator based on event type |
| **Fallback** | `event.message or event_type` |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function** | `_add_event_log()` (lines 1695-1790) |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Generation** | Template selection in `_add_event_log()` | `karn/sanctum/aggregator.py` (lines 1717-1780) |
| **2. Delivery** | `EventLogEntry.message` | `karn/sanctum/schema.py` (line 1290) |

```python
# Aggregator message generation (lines 1718-1780)
if event_type == "SEED_GERMINATED" and isinstance(event.data, SeedGerminatedPayload):
    message = "Germinated"
elif event_type == "SEED_STAGE_CHANGED" and isinstance(event.data, SeedStageChangedPayload):
    message = "Stage changed"
# ... etc
else:
    message = event.message or event_type
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EventLogEntry` |
| **Field** | `message` |
| **Path from SanctumSnapshot** | `snapshot.event_log[i].message` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1290 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` | Primary text in event line (with metadata overlay) |
| EventLogDetail | `widgets/event_log_detail.py` | Modal view raw message |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Aggregator generates message templates
- [x] **Transport works** - Message set in `_add_event_log()`
- [x] **Schema field exists** - `EventLogEntry.message: str` at line 1290
- [x] **Default is correct** - Falls back to event_type name
- [x] **Consumer reads it** - EventLog displays with rich formatting overlay
- [x] **Display is correct** - Generic text with color coding

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Message templates | `[ ]` |
| Widget (EventLog) | `tests/karn/sanctum/widgets/test_event_log.py` | Message display | `[ ]` |

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Event type | field | Determines message template |
| Typed payload | object | PPO skipped detection |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog widget | display | Text content of event line |
| EventLogDetail modal | display | Raw message field |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation |

---

## 8. Notes

> **Design Decision:** Messages are intentionally generic to enable proper event rollup. The design follows the pattern: "message is GENERIC (e.g., 'Germinated', 'Stage changed'). Specific values (slot_id, reward, blueprint) go in metadata dict for display in detail modal. This allows proper rollup by event_type."
>
> **Rich Formatting Override:** The EventLog widget (`_format_individual_event()`) often bypasses the raw message and constructs rich Text objects with specific formatting based on event type and metadata. The message field serves as a fallback and for the detail modal.
>
> **PPO Skipped Case:** When `PPOUpdatePayload.skipped` is True (buffer rollback), the message becomes "PPO skipped" instead of "PPO update".
