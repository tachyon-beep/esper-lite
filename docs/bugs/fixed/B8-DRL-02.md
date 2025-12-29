# Finding Ticket: Seed Optimizer Lifecycle Fragility

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-02` |
| **Severity** | `P1` |
| **Status** | `closed` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-29 |

---

## Resolution

**Status:** FIXED

**Root Cause:** The seed optimizer lifecycle had three bugs:

1. **FOSSILIZE missing pop**: When a seed was fossilized via `_advance_active_seed()`, its optimizer was never removed from `seed_optimizers` — memory leak.

2. **SET_ALPHA_TARGET incorrect pop**: When setting alpha target, the optimizer was popped despite the seed still being active — caused optimizer to be recreated with fresh momentum on next training batch.

3. **ADVANCE unconditional pop**: On any successful `advance_stage()`, the optimizer was popped regardless of whether the seed terminated — wrong for non-terminal transitions like GERMINATED→TRAINING.

**Fixes Applied:**

1. Added `seed_optimizers.pop()` after successful fossilization (line ~2801)
2. Removed incorrect `seed_optimizers.pop()` from SET_ALPHA_TARGET (line ~2850)
3. Changed ADVANCE to only pop if seed is no longer active (line ~2861)

**Verification:**
- 50 tests pass (test_vectorized.py + test_ppo.py)

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Line(s)** | `2741-2862` (action execution block) |
| **Function/Class** | `train_ppo_vectorized()` |

---

## Summary

**One-line summary:** Seed optimizer lifecycle relies on training batch execution order, creating fragile dependencies.

**Category:**
- [x] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [x] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# When seed is germinated/pruned/fossilized, optimizer is popped:
env_state.seed_optimizers.pop(slot_id, None)

# But optimizer is only created lazily in process_train_batch():
# when the slot has an active seed
```

The seed optimizer lifecycle had these issues:

1. **FOSSILIZE**: No cleanup — optimizer leaked
2. **SET_ALPHA_TARGET**: Incorrect cleanup — seed still active
3. **ADVANCE**: Unconditional cleanup — wrong for non-terminal stages

### Impact

- **Memory leaks**: Fossilized seed optimizers accumulated
- **Training instability**: SET_ALPHA_TARGET caused optimizer momentum reset
- **Stale state**: ADVANCE could leave orphaned optimizers

---

## Related Findings

- C8-07 in DRL report: _train_one_epoch doesn't handle mid-epoch pruning (mitigated by callsites)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-19 - Seed optimizer lifecycle is fragile"
