# BUG-017: Nissa BlueprintAnalytics seed event field handling (not a bug)

- **Title:** Nissa BlueprintAnalytics seed event field handling (not an active bug)
- **Context:** Bug report claimed `BlueprintAnalytics.emit` expects fields that aren't guaranteed
- **Impact:** P3 – Code review observation / no production impact
- **Environment:** Main branch
- **Status:** Closed (Not a bug)
- **Resolution:** Verified that training injects `env_id`/`device`, seed events include required fields, and analytics uses defensive defaults.

## Analysis (2025-12-17)

**Not an active bug.** Investigation found the code is correct:

### 1. env_id IS Injected by Both Training Paths

**Vectorized training** (`src/esper/simic/telemetry/emitters.py`):
```python
def emit_with_env_context(hub, env_idx: int, device: str, event: TelemetryEvent) -> None:
    """Safely emit telemetry with env_id/device injected and no shared mutation.

    Creates a new event with the additional context rather than mutating the input.
    """
    data = dict(event.data) if event.data else {}
    data["env_id"] = env_idx
    data["device"] = device
    new_event = dataclasses.replace(event, data=data)
    hub.emit(new_event)
```

**Heuristic training** (`src/esper/simic/training/helpers.py`):
```python
def telemetry_callback(event):
    event.data.setdefault("env_id", 0)
    event.data.setdefault("device", device)
    hub.emit(event)
```

### 2. All Required Fields Are Provided by Emitters

| Event | Field | Emitted by slot.py |
|-------|-------|-------------------|
| SEED_GERMINATED | blueprint_id | ✓ |
| SEED_GERMINATED | seed_id | ✓ |
| SEED_GERMINATED | params | ✓ |
| SEED_FOSSILIZED | blueprint_id | ✓ |
| SEED_FOSSILIZED | seed_id | ✓ |
| SEED_FOSSILIZED | improvement | ✓ |
| SEED_FOSSILIZED | params_added | ✓ |
| SEED_FOSSILIZED | epochs_total | ✓ |
| SEED_CULLED | All required fields | ✓ |

### 3. Analytics Uses Defensive Defaults

`BlueprintAnalytics.emit()` uses `.get()` with defaults for all fields:
- `env_id` defaults to 0
- `params` defaults to 0
- `blueprint_id` defaults to "unknown"
- etc.

No `KeyError` is possible.

## Original Bug Report Claims vs Reality

| Claim | Status | Evidence |
|-------|--------|----------|
| "env_id often missing" | FALSE | Both training paths inject env_id |
| "Emitters omit required fields" | FALSE | slot.py includes all fields |
| "Can KeyError if data missing" | FALSE | All access uses .get() |
| "Analytics miscount stats" | FALSE | Field names match between emitter and consumer |

## Future Consideration

The code could be improved with:
1. A formal Leyline schema defining required fields per event type
2. Runtime validation of telemetry event payloads

These are nice-to-haves, not bugs.

## Links

- `src/esper/kasmina/slot.py::_emit_telemetry` (event emission)
- `src/esper/simic/telemetry/emitters.py::emit_with_env_context` (env_id injection)
- `src/esper/simic/training/helpers.py::telemetry_callback` (env_id injection)
- `src/esper/nissa/analytics.py::emit` (event consumption)
