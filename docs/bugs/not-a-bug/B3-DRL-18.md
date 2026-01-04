# Finding Ticket: DDP Gate Sync Uses Slow Pickle-Based Broadcast

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-18` |
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
| **Line(s)** | `2191-2256` |
| **Function/Class** | `SeedSlot._sync_gate_decision()` |

---

## Summary

**One-line summary:** `broadcast_object_list` uses pickle serialization which is slower than tensor-based broadcasts.

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

The DDP gate synchronization uses `torch.distributed.broadcast_object_list` which pickles Python objects. This is slower than tensor broadcasts.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:2229-2230

object_list: list[dict[str, Any] | None] = [sync_data]
torch.distributed.broadcast_object_list(object_list, src=0)
```

### Why This Is Currently Acceptable

- Gate checks are infrequent (once per epoch per slot)
- The latency is negligible compared to training time
- Pickle allows flexible payload structure

---

## Recommended Fix

For high-frequency sync (if gate checks become more frequent):

```python
# Encode gate result as tensor for faster broadcast
result_tensor = torch.tensor([1 if passed else 0], device=device)
torch.distributed.broadcast(result_tensor, src=0)
passed = result_tensor.item() == 1
```

Current implementation is acceptable for low-frequency gate checks.

---

## Verification

### How to Verify the Fix

- [ ] Profile gate sync latency in DDP training
- [ ] Consider optimization only if gate checks become frequent

---

## Related Findings

None.

---

## Cross-Review

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Valid finding with correct "acceptable for now" conclusion. Gate sync is once-per-epoch-per-slot; pickle overhead is negligible at this frequency. The code already minimizes the payload by serializing only essential fields (lines 2217-2224). Premature optimization would add complexity for marginal gain.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P2 - Performance/Safety" (B3-SLOT-05)
