# Finding Ticket: Seed Optimizer Lifecycle Fragility

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-02` |
| **Severity** | `P1` |
| **Status** | `open` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Line(s)** | `1411-1422, 2693, 2776` |
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

The seed optimizer lifecycle has these issues:

1. **Failed actions don't clean up**: If `action_success = False` (line 2696), the optimizer is NOT popped but seed state may have changed

2. **Order dependency**: If a seed is germinated in epoch N but pruned in epoch N+1 BEFORE any training batch runs, the optimizer will never have been created. The `pop()` is a no-op but the flow is fragile.

3. **Stale optimizer state**: If dynamic seed changes happen between PPO update phases, the optimizer may reference stale param groups

### Impact

- **Silent no-ops**: Popping non-existent optimizers doesn't raise errors
- **Memory leaks**: Optimizers for removed seeds might not be cleaned up on failed actions
- **Training instability**: Stale optimizer state could cause learning rate or momentum inconsistencies

---

## Recommended Fix

Make optimizer lifecycle explicit and robust:

```python
def _ensure_seed_optimizer(env_state, slot_id, model, lr):
    """Create optimizer for seed if not exists."""
    if slot_id not in env_state.seed_optimizers:
        params = list(model.get_seed_parameters(slot_id))
        if params:
            env_state.seed_optimizers[slot_id] = torch.optim.Adam(params, lr=lr)

def _remove_seed_optimizer(env_state, slot_id):
    """Remove optimizer for seed, logging if it didn't exist."""
    opt = env_state.seed_optimizers.pop(slot_id, None)
    if opt is None:
        _logger.debug(f"Optimizer for {slot_id} already removed or never created")
```

Call `_remove_seed_optimizer` ONLY when action succeeds, and verify state consistency.

---

## Verification

### How to Verify the Fix

- [ ] Add explicit optimizer lifecycle methods
- [ ] Verify optimizer cleanup happens only on successful actions
- [ ] Add test for optimizer state after failed actions
- [ ] Add assertion that optimizer exists before stepping

---

## Related Findings

- C8-07 in DRL report: _train_one_epoch doesn't handle mid-epoch pruning (mitigated by callsites)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-19 - Seed optimizer lifecycle is fragile"
