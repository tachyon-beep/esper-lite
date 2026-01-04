# Telemetry Record: [TELE-703] Training Thread Alive

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-703` |
| **Name** | Training Thread Alive |
| **Category** | `infrastructure` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the training thread still running, or has it crashed unexpectedly?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `bool \| None` |
| **Units** | Tri-state: True (alive), False (dead), None (unknown) |
| **Range** | `{True, False, None}` |
| **Precision** | N/A (boolean) |
| **Default** | `None` |

### Semantic Meaning

> Binary state indicator of training thread lifecycle. Sampled every poll cycle from the training thread object's `is_alive()` method.
>
> **True** = Training thread is running normally
> **False** = Training thread has exited or crashed (critical failure)
> **None** = No training thread reference available or not yet initialized

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value is True` | Training thread running, system operating normally |
| **Critical** | `value is False` | Training thread dead, training stopped, requires operator intervention |
| **Unknown** | `value is None` | Thread status unavailable (e.g., before initialization) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Python threading system via `Thread.is_alive()` method |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/app.py` |
| **Function/Method** | `SanctumApp._poll_snapshot()` |
| **Line(s)** | ~520-524 |

```python
# Source: SanctumApp._poll_snapshot()
thread_alive = self._training_thread.is_alive() if self._training_thread else None
primary.training_thread_alive = thread_alive
for snapshot in snapshots_by_group.values():
    snapshot.training_thread_alive = thread_alive
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Direct assignment in poll cycle | `sanctum/app.py` |
| **2. Collection** | Cached in `SanctumSnapshot` object | `sanctum/app.py` |
| **3. Aggregation** | No aggregation (direct UI field) | N/A |
| **4. Delivery** | Passed to widget via `update_snapshot()` | `sanctum/app.py` → `widgets/run_header.py` |

```
[Thread.is_alive()] --> [SanctumApp._poll_snapshot()] --> [SanctumSnapshot.training_thread_alive] --> [RunHeader.render()]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `training_thread_alive` |
| **Path from SanctumSnapshot** | `snapshot.training_thread_alive` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1336 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Thread health indicator (✓/✗/? symbol in Segment 2) |
| ThreadDeathModal | `widgets/thread_death_modal.py` | Trigger mechanism (displays static modal when False) |
| SanctumApp | `app.py` | Logging and modal trigger logic |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `Thread.is_alive()` called in poll cycle
- [x] **Transport works** — Value assigned to snapshot fields (primary + all groups)
- [x] **Schema field exists** — `SanctumSnapshot.training_thread_alive: bool | None = None`
- [x] **Default is correct** — `None` appropriate before thread check or when thread unavailable
- [x] **Consumer reads it** — RunHeader accesses `snapshot.training_thread_alive` at lines 234-240
- [x] **Display is correct** — Renders as ✓ (green) / ✗ (bold red) / ? (dim) symbol
- [x] **Thresholds applied** — Critical state (False) triggers ThreadDeathModal

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/karn/sanctum/test_app.py` | `test_thread_alive_checked_in_poll_cycle` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_sanctum_snapshot_thread_alive_default` | `[x]` |
| Integration (RunHeader display) | `tests/karn/sanctum/test_run_header.py` | `test_thread_status_renders_checkmark_when_alive` | `[x]` |
| Integration (ThreadDeathModal trigger) | `tests/karn/sanctum/test_app.py` | `test_thread_death_modal_shown_when_thread_dies` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (should auto-open or `uv run sanctum`)
3. Observe RunHeader "Thread" segment (should show ✓ in green)
4. In another terminal, identify training process PID and kill the thread: `pkill -f "training_thread"` or equivalent
5. Verify RunHeader switches from ✓ to ✗ (bold red) within next poll cycle (~100ms at default refresh)
6. Verify ThreadDeathModal appears with "TRAINING THREAD DIED" message
7. Return to normal state by restarting training

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `threading.Thread` | Python stdlib | Must be valid Thread object or None |
| Training process | process | Must be running for thread to be alive |
| `SanctumApp` initialization | system | Thread reference passed in constructor |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader display | widget | Uses for ✓/✗/? indicator in Segment 2 |
| ThreadDeathModal | widget | Triggers critical failure notification |
| `_thread_death_shown` flag | system | Prevents duplicate modals |
| Training shutdown logic | system | May be used for graceful cleanup |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record creation, verified full wiring |
| | | |

---

## 8. Notes

### Design Decision

This is a **lightweight debug infrastructure metric** rather than an aggregated telemetry value. It directly queries Python's threading API without any signal emission or event collection. This makes it extremely reliable but limits post-hoc analysis.

### Wiring Status

✅ **FULLY WIRED** - All components connected and tested:
- Source: `Thread.is_alive()` check in poll cycle
- Transport: Direct snapshot field assignment
- Schema: Tri-state field with proper default
- Consumer: RunHeader displays as symbol, ThreadDeathModal triggers on critical state

### Known Behavior

- **Latency:** Thread death detection latency bounded by UI refresh rate (typically 10-100ms depending on refresh_rate setting)
- **Redundancy:** Thread status is assigned to both `primary` snapshot and all per-group snapshots to ensure all widgets see consistent state
- **Modal Show-Once:** `_thread_death_shown` flag ensures ThreadDeathModal appears at most once per session, preventing notification spam
- **No Recovery:** Once thread dies (`False`), the flag remains false until training restarted. System does not attempt automatic recovery.

### Testing Notes

- RunHeader tests verify proper symbol rendering for all three states (True/False/None)
- ThreadDeathModal tests verify it only shows once and is dismissible
- Integration tests verify the full flow from thread death → snapshot update → modal display

### Future Improvements

- Could integrate with log streaming to display thread stack trace in modal
- Could track thread death timestamp for post-hoc analysis
- Could implement thread watchdog with timeout-based resurrection attempt
