# Finding Ticket: STE Assertion Only in Debug Mode

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-16` |
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
| **Line(s)** | `1958-1963` |
| **Function/Class** | `SeedSlot.forward()` |

---

## Summary

**One-line summary:** `_DEBUG_STE` assertion checks `seed_features.requires_grad` but is env-var controlled; frozen parameters could silently produce zero gradients in production.

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

The assertion `assert seed_features.requires_grad` only runs when `_DEBUG_STE` env var is set. In production, if seed parameters are accidentally frozen (e.g., via `.eval()` misuse or checkpoint loading bug), STE would silently produce zero gradients.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:1958-1963

if _DEBUG_STE:
    assert seed_features.requires_grad, "STE requires gradient flow"
```

### Impact

- Silent training failure if seed gradients are blocked
- Hard to debug zero-gradient issues in production

---

## Recommended Fix

Consider a one-time warning instead of assertion:

```python
if not seed_features.requires_grad:
    if not self._warned_no_grad:
        logger.warning("STE: seed_features.requires_grad is False - gradients will be zero")
        self._warned_no_grad = True
```

---

## Verification

### How to Verify the Fix

- [ ] Test with accidentally frozen seed parameters
- [ ] Verify warning appears in logs

---

## Related Findings

None.

---

## Cross-Review

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Valid concern. Silent zero-gradient failures during TRAINING stage STE are hard to debug. The proposed one-time warning is appropriate - it surfaces the issue without requiring env-var configuration while avoiding log spam.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-SLOT-03)
