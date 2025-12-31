# Telemetry Wiring - Remaining Tasks

**Status:** Task 1 complete (commits 2398b89, 5a3c453). Tasks 2-5 pending.

**Full plan:** docs/plans/2025-12-06-wire-telemetry-events.md

---

## Task 2: TAMIYO_INITIATED Event (P1)

**File:** `src/esper/tamiyo/tracker.py`

**What:** Emit TAMIYO_INITIATED when host training stabilizes (germination becomes allowed).

**Implementation:**
1. Add imports at top:
```python
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import get_hub
```

2. After line 120 (after `_is_stabilized = True` and print statement), add:
```python
# Emit TAMIYO_INITIATED telemetry
hub = get_hub()
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.TAMIYO_INITIATED,
    epoch=epoch,
    data={
        "env_id": self.env_id,
        "epoch": epoch,
        "stable_count": self._stable_count,
        "stabilization_epochs": self.stabilization_epochs,
        "val_loss": val_loss,
    },
    message="Host stabilized - germination now allowed",
))
```

**Test:** Create `tests/test_tamiyo_tracker.py` - test that stabilization triggers event emission.

---

## Task 3: Training Progress Events (P2)

**File:** `src/esper/simic/vectorized.py`

**What:** Emit PLATEAU_DETECTED and IMPROVEMENT_DETECTED based on accuracy delta.

**Implementation:** After line 1142 (after hub.emit(ppo_event)), add:
```python
# Emit training progress events
if len(recent_accuracies) >= 2:
    acc_delta = recent_accuracies[-1] - recent_accuracies[-2]
    if acc_delta < 0.5:  # Plateau
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.PLATEAU_DETECTED,
            data={
                "batch": batch_idx + 1,
                "accuracy_delta": acc_delta,
                "rolling_avg_accuracy": rolling_avg_acc,
                "episodes_completed": episodes_completed,
            },
        ))
    elif acc_delta > 2.0:  # Significant improvement
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
            data={
                "batch": batch_idx + 1,
                "accuracy_delta": acc_delta,
                "rolling_avg_accuracy": rolling_avg_acc,
                "episodes_completed": episodes_completed,
            },
        ))
```

---

## Task 4: Warning Events Placeholder (P3)

**File:** `src/esper/leyline/telemetry.py`

**What:** Document MEMORY_WARNING and REWARD_HACKING_SUSPECTED as TODO placeholders.

**Implementation:** Update comments around line 54:
```python
MEMORY_WARNING = auto()  # TODO: Wire up GPU memory monitoring
REWARD_HACKING_SUSPECTED = auto()  # TODO: Wire up reward divergence detection
```

---

## Task 5: Command Events Placeholder (P4)

**File:** `src/esper/leyline/telemetry.py`

**What:** Document COMMAND_* events as future work.

**Implementation:** Update comments around line 48:
```python
# Command events (future: adaptive command system)
COMMAND_ISSUED = auto()  # TODO: Implement when command system built
COMMAND_EXECUTED = auto()
COMMAND_FAILED = auto()
```

---

## Verification After All Tasks

```bash
PYTHONPATH=src pytest tests/ -v --tb=short
grep -E "TAMIYO_INITIATED|PLATEAU|IMPROVEMENT" telemetry/*/events.jsonl
```

---

## Summary of Unwired Events (for reference)

| Event | Status | Location |
|-------|--------|----------|
| RATIO_EXPLOSION_DETECTED | DONE | ppo.py |
| RATIO_COLLAPSE_DETECTED | DONE | ppo.py |
| VALUE_COLLAPSE_DETECTED | DONE | ppo.py |
| NUMERICAL_INSTABILITY_DETECTED | DONE (but won't fire - has_nan/has_inf hardcoded False) | ppo.py |
| GRADIENT_ANOMALY | DONE | ppo.py |
| TAMIYO_INITIATED | Task 2 | tracker.py |
| PLATEAU_DETECTED | Task 3 | vectorized.py |
| IMPROVEMENT_DETECTED | Task 3 | vectorized.py |
| EPOCH_COMPLETED | Deferred (too noisy) | - |
| MEMORY_WARNING | Task 4 (placeholder) | telemetry.py |
| REWARD_HACKING_SUSPECTED | Task 4 (placeholder) | telemetry.py |
| COMMAND_* | Task 5 (placeholder) | telemetry.py |
| ISOLATION_VIOLATION | Deferred | - |
| PERFORMANCE_DEGRADATION | Deferred | - |
