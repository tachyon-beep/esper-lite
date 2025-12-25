# Bug-Hiding Pattern Remediation Plan

**Date:** 2025-12-25
**Reference:** [docs/analysis/2025-12-25-bug-hiding-patterns-audit.md](../analysis/2025-12-25-bug-hiding-patterns-audit.md)
**Total Patterns:** ~123 bug-hiding instances across 6 categories

---

## Phase 1: Typed Payload Completion (P0)

**Goal:** Eliminate all `isinstance(event.data, dict)` checks by ensuring every event type has a typed payload.

### Task 1.1: Create SeedGateEvaluatedPayload

**Files:**
- `src/esper/leyline/telemetry.py` (add payload)
- `src/esper/simic/training/vectorized.py` (update emitter)
- `src/esper/karn/collector.py` (remove dict fallback at lines 402, 439)

**Scope:**
```python
@dataclass(slots=True, frozen=True)
class SeedGateEvaluatedPayload:
    """Payload for SEED_GATE_EVALUATED events."""
    slot_id: str
    env_id: int
    gate_name: str  # e.g., "germination", "fossilization"
    passed: bool
    metrics: dict[str, float]  # Gate-specific metrics
    threshold: float | None = None
    value: float | None = None
```

**Verification:**
- [ ] Grep for `SEED_GATE_EVALUATED` — all emitters use typed payload
- [ ] No `isinstance(event.data, dict)` in collector for this event
- [ ] Tests pass

---

### Task 1.2: Create AnomalyDetectedPayload

**Files:**
- `src/esper/leyline/telemetry.py` (add payload)
- `src/esper/simic/anomaly_detector.py` (update emitter)
- `src/esper/karn/collector.py` (remove dict fallback at line 546)

**Scope:**
```python
@dataclass(slots=True, frozen=True)
class AnomalyDetectedPayload:
    """Payload for anomaly detection events."""
    anomaly_type: str  # e.g., "gradient_explosion", "loss_spike"
    severity: str  # "warning", "critical"
    env_id: int | None = None
    slot_id: str | None = None
    value: float | None = None
    threshold: float | None = None
    context: dict[str, Any] | None = None
```

**Verification:**
- [ ] Grep for anomaly emission — all use typed payload
- [ ] collector.py line 546 dict check removed
- [ ] Tests pass

---

### Task 1.3: Remove Dict Fallbacks from Overwatch Aggregator

**Files:**
- `src/esper/karn/overwatch/aggregator.py` (lines 275, 277, 358, 445, 466, 499)

**Scope:**
After 1.1 and 1.2 complete:
- Remove `isinstance(event.data, dict)` checks
- Replace with direct typed payload access
- Fail loudly if payload is wrong type

**Verification:**
- [ ] No `isinstance(..., dict)` in overwatch/aggregator.py
- [ ] All event handlers expect typed payloads
- [ ] Tests pass

---

### Task 1.4: Remove Dict Fallbacks from Sanctum Aggregator

**Files:**
- `src/esper/karn/sanctum/aggregator.py` (line 1193 and related)

**Scope:**
- Remove `_get_field_from_data()` helper that handles both dict and dataclass
- Replace with direct attribute access on typed payloads

**Verification:**
- [ ] No `isinstance(data, dict)` in sanctum/aggregator.py
- [ ] Tests pass

---

## Phase 2: Slot Initialization Fix (P0)

**Goal:** Initialize all `Slot._blend_*` attributes in `__init__` to eliminate getattr with defaults.

### Task 2.1: Initialize Blend Attributes in Slot.__init__

**Files:**
- `src/esper/kasmina/slot.py`

**Scope:**
Add to `__init__`:
```python
# Blending state (initialized to None, set when blending starts)
self._blend_algorithm_id: str | None = None
self._blend_alpha_target: float | None = None
self._blend_tempo_epochs: int | None = None
```

**Changes:**
- Line 2029: `getattr(self, "_blend_algorithm_id", "sigmoid")` → `self._blend_algorithm_id or "sigmoid"`
- Line 2065: `getattr(self, "_blend_alpha_target", None)` → `self._blend_alpha_target`
- Line 2113: `getattr(self, '_blend_tempo_epochs', None)` → `self._blend_tempo_epochs`
- Lines 2465-2467: Direct attribute access
- Lines 2477-2479: Review alpha_schedule contract

**Verification:**
- [ ] No `getattr(self, "_blend_*"` in slot.py
- [ ] Serialization works correctly
- [ ] Blending tests pass

---

### Task 2.2: Document Alpha Schedule Contract

**Files:**
- `src/esper/kasmina/slot.py`
- `src/esper/kasmina/alpha_controller.py`

**Scope:**
If `alpha_schedule` is not None, these attributes MUST exist:
- `algorithm_id`
- `total_steps`
- `_current_step`

Either:
1. Add assertions in alpha schedule initialization, OR
2. Use Protocol to define required interface

**Verification:**
- [ ] Lines 2477-2479 no longer need getattr defaults
- [ ] Alpha schedule tests verify attribute presence

---

## Phase 3: Silent Default Elimination (P1)

**Goal:** Replace `or {}` and `or 0` patterns with explicit validation.

### Task 3.1: Fix `event.data or {}` in Collector

**Files:**
- `src/esper/karn/collector.py` (line 504)

**Scope:**
```python
# Before
data = event.data or {}

# After
if event.data is None:
    _logger.warning("Event %s has no data payload", event.event_type)
    return
data = event.data
```

**Verification:**
- [ ] Missing payloads logged as warnings
- [ ] Tests verify warning is emitted

---

### Task 3.2: Fix `event.data or {}` in Sanctum Aggregator

**Files:**
- `src/esper/karn/sanctum/aggregator.py` (line 1189)

**Scope:**
Same pattern as 3.1 — log warning and return early.

---

### Task 3.3: Fix `event.data or {}` in Nissa Output

**Files:**
- `src/esper/nissa/output.py` (13 occurrences)

**Scope:**
For each occurrence, either:
1. Log warning and skip (if truly optional), OR
2. Fail loudly (if required)

Review each occurrence for intent.

---

### Task 3.4: Fix `payload.seeds or {}` in Sanctum

**Files:**
- `src/esper/karn/sanctum/aggregator.py` (line 557)

**Scope:**
```python
# Before
seeds_data = payload.seeds or {}

# After
if payload.seeds is None:
    _logger.warning("EPOCH_COMPLETED missing seeds telemetry")
    seeds_data = {}
else:
    seeds_data = payload.seeds
```

---

### Task 3.5: Fix `event.epoch or 0` Patterns

**Files:**
- `src/esper/karn/sanctum/aggregator.py` (line 716)
- `src/esper/nissa/output.py` (line 98)

**Scope:**
- If epoch is required, validate and fail loudly
- If optional, use explicit None check: `epoch = event.epoch if event.epoch is not None else 0`

---

### Task 3.6: Audit `slot_id or payload.slot_id` Patterns

**Files:**
- `src/esper/karn/collector.py` (lines 407, 510)
- `src/esper/karn/overwatch/aggregator.py` (lines 306, 330, 363, 387, 407)
- `src/esper/karn/sanctum/aggregator.py` (lines 816, 854, 882, 920)

**Scope:**
Determine authoritative source for slot_id:
1. If event.slot_id is authoritative, remove fallback
2. If payload.slot_id is authoritative, use it directly
3. If both valid, document which takes precedence

---

## Phase 4: Exception Handling Fix (P1)

**Goal:** Replace bare `except Exception: pass` with proper error handling.

### Task 4.1: Fix Vitals Collection in Sanctum

**Files:**
- `src/esper/karn/sanctum/aggregator.py` (lines 1265-1309)

**Scope:**
```python
# Before
try:
    self._vitals.cpu_percent = psutil.cpu_percent(interval=None)
except Exception:
    pass

# After
try:
    self._vitals.cpu_percent = psutil.cpu_percent(interval=None)
except Exception as e:
    _logger.warning("Failed to collect CPU vitals: %s", e)
    self._vitals.cpu_percent = None  # Explicit unavailable state
```

Apply to all 4 instances (CPU, RAM, GPU stats, GPU import).

**Verification:**
- [ ] Vitals failures logged at WARNING level
- [ ] Unavailable state explicitly set

---

### Task 4.2: Fix Blueprint Registry Cache

**Files:**
- `src/esper/kasmina/blueprints/registry.py` (lines 33-35)

**Scope:**
```python
# Before
except AttributeError:
    _logger.debug("Cache invalidation skipped: cache not initialized")
    pass

# After
except AttributeError as e:
    _logger.debug("Cache invalidation skipped: %s", e)
    # No pass needed — debug log is sufficient
```

---

### Task 4.3: Narrow Exception Types in Vectorized

**Files:**
- `src/esper/simic/training/vectorized.py` (lines 557-566)

**Scope:**
```python
# Before
except Exception:
    pass

# After
except (ImportError, AttributeError) as e:
    _logger.debug("tqdm lock configuration skipped: %s", e)
```

---

### Task 4.4: Add Logging to Network Discovery

**Files:**
- `src/esper/scripts/train.py` (lines 327-337)

**Scope:**
```python
# Before
except Exception:
    pass

# After
except OSError as e:
    _logger.debug("Network interface discovery failed: %s", e)
```

---

## Phase 5: Store Schema Cleanup (P2)

**Goal:** Reduce isinstance checks in `karn/store.py` deserialization.

### Task 5.1: Define Strict Schema Types

**Files:**
- `src/esper/karn/store.py`

**Scope:**
- Document expected types for each field
- Use TypedDict or Pydantic for stricter validation
- Remove isinstance chains where possible

### Task 5.2: Migrate to Structured Parsing

Consider using `cattrs` or similar for structured deserialization instead of manual isinstance chains.

---

## Phase 6: Config Dataclass Cleanup (P2)

**Goal:** Remove isinstance in `__post_init__` by enforcing types at construction.

### Task 6.1: TrainingConfig Type Enforcement

**Files:**
- `src/esper/simic/training/config.py`

**Scope:**
- Accept only typed values (enum, not str)
- Move coercion to factory function or `from_dict` classmethod
- Remove isinstance checks from `__post_init__`

---

## Execution Order

```
Phase 1 (P0): Typed Payloads
├── Task 1.1: SeedGateEvaluatedPayload
├── Task 1.2: AnomalyDetectedPayload
├── Task 1.3: Overwatch dict removal (depends on 1.1, 1.2)
└── Task 1.4: Sanctum dict removal (depends on 1.1, 1.2)

Phase 2 (P0): Slot Init
├── Task 2.1: _blend_* initialization
└── Task 2.2: Alpha schedule contract

Phase 3 (P1): Silent Defaults
├── Task 3.1-3.4: event.data or {} fixes
├── Task 3.5: epoch or 0 fixes
└── Task 3.6: slot_id audit

Phase 4 (P1): Exception Handling
├── Task 4.1: Vitals exceptions
├── Task 4.2: Registry cache
├── Task 4.3: Vectorized tqdm
└── Task 4.4: Network discovery

Phase 5-6 (P2): Schema/Config cleanup
└── Longer-term refactoring
```

---

## Success Metrics

After completion:

- [ ] Zero `isinstance(event.data, dict)` in aggregators/collector
- [ ] Zero `getattr(self, "_blend_*"` in slot.py
- [ ] Zero bare `except Exception: pass` without logging
- [ ] All `or {}` patterns either removed or documented as intentional
- [ ] Full test suite passes

---

## Notes

- Each task should be a single commit
- Run tests after each task
- Tasks 1.1-1.4 can be parallelized with separate agents
- Phase 2 is independent of Phase 1
- Phase 3-4 can proceed in parallel once Phase 1-2 complete
