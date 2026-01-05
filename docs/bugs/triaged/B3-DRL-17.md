# Finding Ticket: isolate_gradients Default May Be Wrong for CNNs

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-17` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 3 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Line(s)** | `2537` |
| **Function/Class** | `SeedSlot.set_extra_state()` |

---

## Summary

**One-line summary:** `isolate_gradients` defaults to `False` in checkpoint loading, which is correct for transformers but not CNNs.

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

The checkpoint loading defaults `isolate_gradients=False`. CNNs typically use `isolate_gradients=True` to avoid BatchNorm stats drift. If a CNN checkpoint is loaded and this field is missing, gradient isolation will be disabled incorrectly.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:2537

self.isolate_gradients = state.get("isolate_gradients", False)
```

### Why This Is Low Risk

`set_extra_state` should only receive dicts from `get_extra_state`, which includes this field. The default is only a fallback for very old checkpoints.

---

## Resolution

**Low priority** - The field should always be present in checkpoints from current code. Consider:

1. Adding validation that the field exists
2. Failing fast on missing required fields per project policy

---

## Verification

### How to Verify the Fix

- [ ] Check that all checkpoint saves include isolate_gradients
- [ ] Consider removing default to fail fast

---

## Related Findings

- B3-CR-01: .get() silently handles malformed checkpoint

---

## Cross-Review

| Verdict | **OBJECT** |
|---------|------------|

**Evaluation:** This ticket duplicates B3-CR-01, which correctly frames this as a defensive programming violation per CLAUDE.md. The `.get(..., False)` pattern silently masks malformed checkpoints. Per project policy, replace with direct access `state["isolate_gradients"]` to fail fast. Close this as duplicate of B3-CR-01.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-SLOT-04)
