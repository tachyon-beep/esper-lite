# Finding Ticket: GradientEMATracker Instantiated but Never Used

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B7-DRL-01` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/telemetry/gradient_ema.py`, `vectorized.py` |
| **Line(s)** | `vectorized.py:1687` |
| **Function/Class** | `GradientEMATracker` |

---

## Summary

**One-line summary:** `GradientEMATracker` is instantiated but `update()` and `check_drift()` are never called - dead code.

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
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [x] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# vectorized.py line 1687
grad_ema_tracker = GradientEMATracker() if use_telemetry else None
```

The tracker is created but:
1. `grad_ema_tracker.update()` is never called
2. `grad_ema_tracker.check_drift()` is never called
3. Grep confirms no usage anywhere in the codebase

**DRL Impact:** Gradient drift detection is a valuable training diagnostic:
- Detects slow divergence from stable training dynamics
- Identifies when learning rate may need adjustment
- Catches gradual policy collapse before catastrophic failure

By instantiating but not using the tracker, the code claims drift detection capability but provides none.

**Related:** `AnomalyDetector.check_gradient_drift()` (which would consume EMA output) is also never called.

---

## Recommended Fix

Either wire it up or delete per No Legacy Code Policy:

**Option 1 - Wire it up:**
```python
# In PPO update loop:
grad_ema_tracker.update(gradient_norm, gradient_health)
drift_report = anomaly_detector.check_gradient_drift(
    norm_drift=grad_ema_tracker.norm_drift,
    health_drift=grad_ema_tracker.health_drift,
)
```

**Option 2 - Delete dead code:**
```python
# Remove from vectorized.py
# Remove GradientEMATracker class entirely
# Remove check_gradient_drift from AnomalyDetector
```

---

## Verification

### How to Verify the Fix

- [ ] `grep -r "grad_ema_tracker\.\(update\|check_drift\)" src/esper/simic/`
- [ ] Either shows calls after wiring, or no results after deletion
- [ ] Run tests to verify no breakage

---

## Related Findings

- B7-DRL-04: check_gradient_drift() also never called
- B7-DRL-02: check_performance_degradation() also unwired

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch7-drl.md`
**Section:** "GradientEMATracker is instantiated but never used"

### Dead Code Verification Commands

```bash
grep -r "grad_ema_tracker\.\(update\|check_drift\)" src/esper/simic/
# Returns empty - confirming dead code
```

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Significant RL observability gap. Gradient drift detection catches: (1) policy collapse (norms trending to zero), (2) divergence early warning (norms growing exponentially), (3) learning rate scheduling signals. Per CLAUDE.md telemetry mandate, wire it up rather than delete.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |

**Evaluation:** Confirmed dead code. Grep returns zero method calls. Clear No Legacy Code Policy violation. If wired up, gradient tracking would use torch.norm() which is well-optimized. No compile compatibility concerns with the fix.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Verified: no matches for grad_ema_tracker methods. Clear No Legacy Code Policy violation. Creates false expectations (readers assume drift detection works). Per CLAUDE.md's explicit telemetry guidance, wire it up immediately rather than delete.
