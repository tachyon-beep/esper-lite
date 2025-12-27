# Finding Ticket: check_performance_degradation() Explicitly Unwired

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B7-DRL-02` |
| **Severity** | `P1` |
| **Status** | `open` |
| **Batch** | 7 |
| **Agent** | `drl` |
| **Domain** | `simic/telemetry` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

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

## Recommended Fix

Wire it up in the training loop:

```python
# At end of each epoch in vectorized.py:
if telemetry_enabled:
    degradation_result = check_performance_degradation(
        current_accuracy=current_acc,
        rolling_accuracy=rolling_acc,
        episodes_completed=episode_count,
    )
    if degradation_result.is_degraded:
        _logger.warning("Performance degradation detected: %s", degradation_result)
```

---

## Verification

### How to Verify the Fix

- [ ] `grep -r "check_performance_degradation(" src/esper/simic/ | grep -v "def check_performance_degradation"` shows callers
- [ ] Run training with telemetry enabled
- [ ] Verify degradation detection works by artificially degrading performance

---

## Related Findings

- B7-DRL-01: GradientEMATracker also never used
- B7-DRL-04: check_gradient_drift() also never called

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch7-drl.md`
**Section:** "check_performance_degradation() is explicitly marked as UNWIRED"

**Report file:** `docs/temp/2712reports/batch7-codereview.md`
**Section:** "TODO: UNWIRED TELEMETRY never called"

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
