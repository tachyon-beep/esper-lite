# Telemetry Record: [TELE-815] Event Log Metadata

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-815` |
| **Name** | Event Log Metadata |
| **Category** | `event_log` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What are the specific details of this event (slot, blueprint, improvement, etc.)?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every event)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `dict[str, str \| int \| float]` |
| **Format** | Key-value pairs of event-specific data |
| **Default** | `{"event_id": "<uuid>"}` (always present) |

### Semantic Meaning

> The metadata dict contains structured, event-specific values that complement the generic message. While the message enables rollup grouping, metadata preserves the rich detail needed for display and debugging.

### Metadata Fields by Event Type

#### Common Fields (All Events)

| Key | Type | Description |
|-----|------|-------------|
| `event_id` | `str` | Unique event identifier (UUID) |

#### SEED_GERMINATED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `slot_id` | `str` | Target slot | `"r0c1"` |
| `blueprint` | `str` | Blueprint ID | `"conv_light"` |

#### SEED_STAGE_CHANGED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `slot_id` | `str` | Target slot | `"r0c1"` |
| `from` | `str` | Previous stage | `"TRAINING"` |
| `to` | `str` | New stage | `"BLENDING"` |

#### SEED_GATE_EVALUATED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `slot_id` | `str` | Target slot | `"r0c1"` |
| `gate` | `str` | Gate name | `"G2"` |
| `target_stage` | `str` | Stage if pass | `"BLENDING"` |
| `result` | `str` | PASS or FAIL | `"PASS"` |
| `failed_checks` | `int` | Number failed (optional) | `2` |
| `detail` | `str` | Message (optional) | `"Alpha below threshold"` |

#### SEED_FOSSILIZED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `slot_id` | `str` | Target slot | `"r0c1"` |
| `improvement` | `float` | Accuracy improvement % | `2.3` |

#### SEED_PRUNED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `slot_id` | `str` | Target slot | `"r0c1"` |
| `reason` | `str` | Prune reason | `"stagnation"` |

#### PPO_UPDATE_COMPLETED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `entropy` | `float` | Policy entropy | `1.23` |
| `clip_fraction` | `float` | Clipped ratio fraction | `0.15` |
| `reason` | `str` | (skipped only) | `"buffer rollback"` |

#### BATCH_EPOCH_COMPLETED

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `batch` | `int` | Batch index | `42` |
| `episodes` | `int` | Episodes completed | `100` |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Extracted from typed payloads in aggregator |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function** | `_add_event_log()` (lines 1695-1790) |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Payload** | Typed dataclass fields | `leyline/telemetry.py` |
| **2. Extraction** | Field-by-field in `_add_event_log()` | `karn/sanctum/aggregator.py` |
| **3. Delivery** | `EventLogEntry.metadata` | `karn/sanctum/schema.py` (line 1294) |

```python
# Aggregator metadata population (lines 1714-1768)
metadata: dict[str, str | int | float] = {"event_id": event.event_id}

if event_type == "SEED_GERMINATED" and isinstance(event.data, SeedGerminatedPayload):
    message = "Germinated"
    slot_id = event.slot_id if event.slot_id else event.data.slot_id
    metadata["slot_id"] = slot_id
    metadata["blueprint"] = event.data.blueprint_id
    env_id = event.data.env_id
# ... etc
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EventLogEntry` |
| **Field** | `metadata` |
| **Path from SanctumSnapshot** | `snapshot.event_log[i].metadata` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1294 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` (lines 370-473) | Rich inline formatting |
| EventLogDetail | `widgets/event_log_detail.py` | Raw metadata display |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Aggregator extracts from typed payloads
- [x] **Transport works** - Metadata dict populated per event type
- [x] **Schema field exists** - `EventLogEntry.metadata: dict` at line 1294
- [x] **Default is correct** - `field(default_factory=dict)` with `event_id` always added
- [x] **Consumer reads it** - EventLog formats inline; EventLogDetail shows raw
- [x] **Display is correct** - Rich formatting with color-coded values

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Metadata extraction | `[ ]` |
| Widget (EventLog) | `tests/karn/sanctum/widgets/test_event_log.py` | Inline formatting | `[ ]` |

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Typed payloads | dataclass | Field extraction source |
| Event envelope | object | `event.event_id`, `event.slot_id` |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog inline display | display | Rich Text formatting |
| EventLogDetail modal | display | Raw key-value display |
| Event deduplication | system | `event_id` for processed tracking |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation |

---

## 8. Notes

> **Design Rationale:** The metadata dict enables a separation of concerns: the `message` field provides generic text for rollup grouping, while `metadata` preserves rich detail. The EventLog widget uses metadata to construct formatted inline displays like "r0c1 GERM conv_light" instead of just "Germinated".
>
> **Event ID Deduplication:** The `event_id` (UUID) is always present in metadata and used by the EventLog widget's `_mark_processed()` method to prevent duplicate display of the same event during snapshot polling.
>
> **Slot ID Priority:** For SEED_* events, the aggregator prefers `event.slot_id` (from the event envelope) over `event.data.slot_id` (from the payload) because the envelope is set at emission time and is authoritative.
>
> **Display Truncation:** The EventLog widget truncates long values for display:
> - Blueprint: 10 chars max (e.g., `"conv_light"` -> `"conv_ligh..."`)
> - Reason: 8 chars max
> - Stage names: Shortened via `_STAGE_SHORT` mapping (e.g., `"FOSSILIZED"` -> `"FOSS"`)
