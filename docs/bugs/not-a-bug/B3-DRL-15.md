# Finding Ticket: Memory Format Handling Adds Overhead for channels_last

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-15` |
| **Severity** | `P2` |
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
| **Line(s)** | `1939-1946` |
| **Function/Class** | `SeedSlot.forward()` |

---

## Summary

**One-line summary:** When `isolate_gradients=True` and input is channels_last, `.contiguous().detach()` creates a contiguous copy, adding memory overhead.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [x] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

For channels_last CNN hosts with `isolate_gradients=True`, the code calls `.contiguous().detach()` which creates a contiguous copy in memory.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:1939-1946

# BUG-005 fix: ensure contiguous before detach for channels_last
seed_input = host_features.contiguous().detach()
```

### Why This Is Correct

The contiguous call is required to avoid BUG-005 (a known issue with channels_last and detach). The overhead is the cost of correctness.

---

## Resolution

**Acceptable as-is** - Consider measuring the overhead:

1. Profile memory usage with channels_last hosts
2. Document expected memory overhead in host selection guidance
3. Consider if channels_last is worth the overhead for specific use cases

---

## Verification

### How to Verify the Fix

- [ ] Profile memory with channels_last vs contiguous hosts
- [ ] Document in performance guide

---

## Related Findings

None.

---

## Cross-Review

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Ticket correctly identifies the overhead and correctly concludes it is acceptable. The actual code (lines 1940-1944) conditionally calls `.contiguous()` only when `not is_contiguous()`, minimizing overhead. The ticket's "acceptable as-is" resolution is appropriate; this is a known cost of correctness for channels_last inputs.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P2 - Performance/Safety" (B3-SLOT-02)
