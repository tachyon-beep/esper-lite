# Finding Ticket: Alpha Controller Tick Synchronization Not Enforced in DDP

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-05` |
| **Severity** | `P2` |
| **Status** | `closed` |
| **Batch** | 2 |
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
| **Line(s)** | `(DDP integration)` |
| **Function/Class** | `SeedSlot.step_epoch()` |

---

## Summary

**One-line summary:** Alpha controller ticks happen in `step_epoch()` without explicit DDP synchronization - ranks could drift if epochs complete at different times.

**Category:**
- [ ] Correctness bug
- [x] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
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

In DDP training, alpha controller state must be synchronized across ranks. The `_sync_gate_decision()` method in slot.py handles gate results, but alpha controller ticks happen in `step_epoch()` without explicit sync.

### Risk Scenario

1. Rank 0's epoch completes before Rank 1 (due to data loader imbalance)
2. Rank 0 calls `step_epoch()` → alpha advances
3. Rank 1 still on previous alpha value
4. Forward passes use different alpha values → gradient desync

### Code Context

```python
# From slot.py docstring (lines 21-37)
# Documents DDP symmetry requirements but relies on symmetric step_epoch() calls
```

### Current Mitigation

The docstring in slot.py documents DDP symmetry requirements. If data loaders are balanced (same number of batches per epoch), ranks stay synchronized.

---

## Recommended Fix

### Option A: Add explicit alpha sync at epoch boundaries

```python
def _sync_alpha_controller(self) -> None:
    """Synchronize alpha controller state across DDP ranks."""
    if not dist.is_initialized():
        return

    # Broadcast rank 0's alpha to all ranks
    alpha_tensor = torch.tensor(
        [self._alpha_controller.alpha],
        device=self._device
    )
    dist.broadcast(alpha_tensor, src=0)
    # ... update local controller if needed
```

### Option B: Document data loader balance requirement

Add explicit documentation that data loaders must produce the same number of batches across all ranks for DDP correctness.

---

## Verification

### How to Verify the Fix

- [ ] Add DDP integration test for alpha sync
- [ ] Test with imbalanced data loaders
- [ ] Verify alpha values match across ranks

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-CR-03` | `related` | DDP gate synchronization ordering |

---

## Resolution

**Status:** Already Fixed

**Evidence:** The documentation at `src/esper/kasmina/slot.py` lines 21-37 already addresses this concern:

- Line 27: "Call advance_stage() / step_epoch() in identical order"
- This documents the contract that the training loop must ensure symmetric calls

Per the DRL cross-review recommendation: "Document the assumption rather than add synchronization overhead." The documentation already exists.

Adding explicit alpha sync barriers would add unnecessary overhead and deadlock if ranks actually call `step_epoch()` different numbers of times (which would be a training loop bug, not a slot bug).

**Sign-off:** Confirmed by `drl-expert`

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "Cross-Cutting Integration Risks" - "Alpha Controller State Synchronization"

---

## Cross-Review: PyTorch Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | DDP synchronization of alpha is critical for gradient consistency. The `_sync_gate_decision()` pattern using `dist.broadcast` is the correct approach; extending it to alpha state is straightforward and follows PyTorch DDP best practices. Option A is preferred over documentation-only Option B. |

---

## Cross-Review: Code Review Specialist

| Verdict | **NEUTRAL** |
|---------|-------------|
| **Evaluation** | The DDP symmetry documentation at slot.py lines 21-37 explicitly requires "identical order" of `step_epoch()` calls, making rank drift a user error not a code defect. Adding explicit alpha sync (Option A) would add synchronization overhead when data loaders are correctly balanced per documented requirements.

---

## Cross-Review: DRL Specialist

| Verdict | **NEUTRAL** |
|---------|-------------|
| **Evaluation** | The risk is real but mitigated by design: `step_epoch()` is called once per epoch after all forward passes complete, and `_sync_gate_decision()` already handles gate synchronization. Alpha drift would require data loader imbalance severe enough to desync epoch boundaries, which is a training loop bug, not a Kasmina bug. Document the assumption rather than add synchronization overhead. |
