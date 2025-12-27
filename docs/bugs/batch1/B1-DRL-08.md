# Finding Ticket: loss_at_panic May Be NaN If Rollback Called Without Panic

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B1-DRL-08` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 1 |
| **Agent** | `drl` |
| **Domain** | `tolaria` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tolaria/governor.py` |
| **Line(s)** | `332` |
| **Function/Class** | `TolariaGovernor.execute_rollback()` |

---

## Summary

**One-line summary:** If `execute_rollback()` is called without a preceding panic, `loss_at_panic` in the report is `float('nan')`.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [x] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

If `execute_rollback()` is called directly (not via the normal panic path), `self._panic_loss` is None, and the report contains `float('nan')` for `loss_at_panic`.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/tolaria/governor.py:332

return GovernorReport(
    ...
    loss_at_panic=self._panic_loss if self._panic_loss is not None else float('nan'),
    ...
)
```

### Why This Matters

- Semantically correct (no panic = no panic loss)
- But NaN could be confusing vs explicit None
- Telemetry consumers need to handle NaN specially

---

## Recommended Fix

### Option A: Use None instead of NaN

```python
loss_at_panic=self._panic_loss,  # None if no panic
```

This requires updating GovernorReport to allow `loss_at_panic: float | None`.

### Option B: Document the NaN behavior

Add docstring explaining that NaN means "no panic preceded this rollback".

---

## Verification

### How to Verify the Fix

- [ ] Unit test: Verify behavior when rollback called without panic

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **DRL** | NEUTRAL | NaN vs None for loss_at_panic is a serialization/telemetry concern with no impact on RL training dynamics. Recommend Option B (document the NaN behavior) as the lowest-friction fix; this does not affect reward signals or learning stability. |
| **PyTorch** | NEUTRAL | Using NaN for missing panic loss is semantically defensible and does not affect tensor operations or torch.compile. The choice between None and NaN is a data modeling decision with no PyTorch correctness implications. |
| **CodeReview** | NEUTRAL | The finding is valid - NaN vs None for missing data is a legitimate design question. However, NaN is often the pragmatic choice for numeric fields in telemetry to avoid Optional complications in downstream aggregation. Given P4 severity and the fact that consumers can check isnan(), this is low-priority polish. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch1-drl.md`
**Section:** "T1-G-8"
