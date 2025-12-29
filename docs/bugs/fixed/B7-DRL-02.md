# Finding Ticket: check_performance_degradation() Explicitly Unwired

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B7-DRL-02` |
| **Severity** | `P1` |
| **Status** | `closed` |
| **Batch** | 7 |
| **Agent** | `drl` |
| **Domain** | `simic/telemetry` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2025-12-29 |

---

## Resolution

**Status:** FIXED

**Root Cause:** The `check_performance_degradation()` function was fully implemented but never called. An explicit `TODO: [UNWIRED TELEMETRY]` comment marked where it should be wired.

**Fix Applied:**
1. Added import of `check_performance_degradation` to vectorized.py
2. Wired up call in the batch completion block (after `on_batch_completed`)
3. Removed the TODO comment from emitters.py

**Call Site (vectorized.py ~line 3385):**
```python
# B7-DRL-02: Check for performance degradation (was previously unwired)
# Detects catastrophic forgetting, reward hacking, and training decay
training_progress = (episodes_completed + envs_this_batch) / total_episodes
check_performance_degradation(
    hub,
    current_acc=avg_acc,
    rolling_avg_acc=rolling_avg_acc,
    env_id=0,  # Aggregate metric across all envs
    training_progress=training_progress,
)
```

**Verification:**
- All 40 vectorized/emitter tests pass
- Existing unit tests for the function confirm correct behavior:
  - `test_performance_degradation_emitted_on_accuracy_drop`
  - `test_no_degradation_event_when_stable`
  - `test_no_degradation_event_during_warmup`
  - `test_degradation_event_emitted_after_warmup`

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py` |
| **Line(s)** | `923-978` |
| **Function/Class** | `check_performance_degradation()` |

---

## Summary

**One-line summary:** Performance degradation detection function exists but is explicitly marked as TODO/unwired.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [x] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [x] Documentation / naming
- [ ] Defensive programming violation
- [x] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# emitters.py lines 923-924
# TODO: UNWIRED TELEMETRY
# Call check_performance_degradation() at end of each epoch
```

The function `check_performance_degradation()`:
1. Exists and is fully implemented (lines 925-978)
2. Has a warmup threshold (line 953) correctly accounting for early PPO variance
3. Has no callers anywhere in the codebase
4. Is documented as "should be called" but isn't

**DRL Impact:** Performance degradation detection is critical for RL:
- Detects when policy updates are making performance worse
- Identifies catastrophic forgetting
- Catches reward hacking (policy exploiting reward rather than solving task)

Per CLAUDE.md: "If you are being asked to deliver a telemetry component, do not defer or put it off... This pattern of behaviour is why we are several months in and have no telemetry."

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Critical RL diagnostic gap. Detects: (1) catastrophic forgetting after policy updates, (2) reward hacking (exploiting shaping without solving task), (3) hyperparameter failures causing gradual decay. The warmup threshold shows RL awareness. Must wire up per CLAUDE.md mandate.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |

**Evaluation:** Confirmed dead code with explicit TODO marker. Function operates on Python floats - pure business logic with no tensor ops or GPU concerns. Wiring it up introduces no torch.compile graph breaks. P1 appropriate per CLAUDE.md telemetry mandate.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Verified: explicit `TODO: [UNWIRED TELEMETRY]` comment. Function is fully implemented, well-designed, and ready to use. This is the exact anti-pattern CLAUDE.md warns against. The fix is trivial: add call site. Wire immediately, not as TODO.
