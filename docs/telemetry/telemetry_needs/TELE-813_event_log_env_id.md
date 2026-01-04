# Telemetry Record: [TELE-813] Event Log Env ID

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-813` |
| **Name** | Event Log Env ID |
| **Category** | `event_log` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which training environment did this event occur in, or is this a global event?"

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
| **Type** | `int \| None` |
| **Range** | `0` to `n_envs - 1`, or `None` for global events |
| **Example** | `0`, `3`, `None` |
| **Default** | `None` (global event) |

### Semantic Meaning

> The environment ID identifies which parallel training environment produced the event. In vectorized training, multiple environments run simultaneously. The env_id enables:
> - Per-environment event filtering
> - Multi-env correlation analysis
> - Distinguishing per-env events (seed lifecycle, rewards) from global events (PPO updates, batch completions)

### Event Type to Env ID Mapping

| Event Type | Env ID | Notes |
|------------|--------|-------|
| `SEED_*` events | `int` | Per-environment seed lifecycle |
| `EPOCH_COMPLETED` | `int` | Per-environment epoch |
| `REWARD_COMPUTED` | `int` | Per-environment reward |
| `PPO_UPDATE_COMPLETED` | `None` | Global policy update |
| `BATCH_EPOCH_COMPLETED` | `None` | Global batch boundary |
| `TRAINING_STARTED` | `None` | Global initialization |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Typed payload `env_id` field |
| **Files** | Various emitters via payload dataclasses |
| **Sentinel Value** | `-1` in Kasmina slots (replaced by actual env_id in `emit_with_env_context`) |

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Payload `env_id` field | Various typed payloads |
| **2. Context Injection** | `emit_with_env_context()` replaces `-1` sentinel | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted from typed payload | `karn/sanctum/aggregator.py` (lines 1717-1758) |
| **4. Delivery** | `EventLogEntry.env_id` | `karn/sanctum/schema.py` (line 1289) |

```python
# Aggregator extraction (lines 1717-1758)
if event_type == "SEED_GERMINATED" and isinstance(event.data, SeedGerminatedPayload):
    # ...
    env_id = event.data.env_id
elif event_type == "PPO_UPDATE_COMPLETED":
    # env_id remains None (global event)
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EventLogEntry` |
| **Field** | `env_id` |
| **Path from SanctumSnapshot** | `snapshot.event_log[i].env_id` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1289 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EventLog | `widgets/event_log.py` (lines 470, 496-500, 561-562) | Right-justified env ID column |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - Typed payloads include `env_id` field
- [x] **Transport works** - Aggregator extracts from typed payloads
- [x] **Schema field exists** - `EventLogEntry.env_id: int | None` at line 1289
- [x] **Default is correct** - `None` for global events
- [x] **Consumer reads it** - EventLog formats as right-justified column
- [x] **Display is correct** - Shows env IDs or truncates with "+" for many envs

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Env ID extraction | `[ ]` |
| Widget (EventLog) | `tests/karn/sanctum/widgets/test_event_log.py` | Env ID display | `[ ]` |

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Typed payload `env_id` | field | Per-event environment context |
| `emit_with_env_context()` | function | Replaces `-1` sentinel with actual env_id |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EventLog env column | display | Right-justified env ID display |
| Aggregated event grouping | system | Groups by env_id for count display |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation |

---

## 8. Notes

> **Sentinel Value:** Kasmina slots use `-1` as a sentinel for env_id because they don't know their environment context. The `emit_with_env_context()` helper in `simic/telemetry/emitters.py` replaces this sentinel with the actual env_id before the event enters the telemetry pipeline.
>
> **Display Truncation:** When aggregated events span many environments, the EventLog widget shows up to 3 env IDs (e.g., "0 1 2") followed by "+" if more exist (`_MAX_ENVS_SHOWN = 3`).
>
> **Global Events:** PPO updates, batch completions, and training start are global events with `env_id = None`. These affect all environments simultaneously.
