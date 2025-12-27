# Finding Ticket: zero_tensor.clone() Allocates Inside GAE Loop

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B4-DRL-03` |
| **Severity** | `P2` |
| **Status** | `open` |
| **Batch** | 4 |
| **Agent** | `drl` |
| **Domain** | `simic` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/agent/rollout_buffer.py` |
| **Line(s)** | `388` |
| **Function/Class** | `TamiyoRolloutBuffer.compute_advantages_and_returns()` |

---

## Summary

**One-line summary:** `zero_tensor.clone()` inside loop creates a new tensor on every terminal state - up to 25 allocations per environment.

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

The GAE loop resets `last_gae` on terminal states using `zero_tensor.clone()`:

```python
# Line 388
if dones[t] and not truncated[t]:
    last_gae = zero_tensor.clone()  # New allocation
```

For a 25-step episode with multiple terminal states, this creates unnecessary tensor allocations.

### Why This Matters

- Each `.clone()` allocates a new tensor
- In a tight loop, this causes memory fragmentation
- The subsequent line `last_gae = delta + ...` creates a new tensor anyway, so the clone is wasteful

---

## Recommended Fix

Use in-place zeroing or direct assignment:

```python
# Option 1: Direct assignment (creates new tensor, but cleaner)
if dones[t] and not truncated[t]:
    last_gae = torch.zeros_like(last_gae)

# Option 2: Multiply by zero (reuses computation on next line)
# The delta + gamma * gae_lambda * (1 - done) * last_gae handles this naturally
# when done=True, the (1 - done) term zeros out last_gae contribution

# Option 3: Pre-computed mask (most efficient)
done_mask = (dones & ~truncated).float()  # Pre-compute outside loop
# Then in loop:
last_gae = delta + gamma * gae_lambda * (1 - done_mask[t]) * last_gae
```

---

## Verification

### How to Verify the Fix

- [ ] Profile allocation count with current vs fixed code
- [ ] Verify GAE values are unchanged
- [ ] Run `pytest tests/simic/test_ppo.py -v`

---

## Related Findings

- B4-CR-03: GAE loop vectorization (broader optimization)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch4-drl.md`
**Section:** "P2 (Performance/Correctness Concerns)" (R-1)

---

## Cross-Review

| Reviewer | Verdict | Date |
|----------|---------|------|
| Code Review Specialist | **ENDORSE** | 2024-12-27 |
| PyTorch Specialist | **ENDORSE** | 2024-12-27 |

**Code Review Evaluation:** Valid finding - `zero_tensor.clone()` at lines 388 and 409 allocates inside the inner loop on terminal states. The suggested Option 3 (pre-computed mask) would eliminate all clone() calls and is worth implementing for cleaner semantics even if perf gain is modest.

**PyTorch Evaluation:** Option 2 is the cleanest fix from a PyTorch perspective - the GAE formula `last_gae = delta + gamma*lambda*(1-done)*last_gae` already handles terminal reset via the `(1-done)` term when done=True. The explicit `zero_tensor.clone()` conditional is redundant and can be removed entirely. However, note the code distinguishes `dones[t] and not truncated[t]` which requires `(1 - float(dones[t] and not truncated[t]))` not just `(1-done)`. With this correction, Option 2 remains correct and eliminates all clone() calls.

| DRL Specialist | **ENDORSE** | 2024-12-27 |

**DRL Evaluation:** Valid minor optimization, but ticket overstates impact. "Up to 25 allocations per env" is incorrect - clone() only executes on TRUE terminal states (not truncation), which occur at most once per env per rollout. The `next_non_terminal` term at lines 401/405 already handles GAE reset mathematically via `(1 - float(dones[t] and not truncated[t]))`. The explicit clone() is semantically redundant but harmless. Correct GAE treatment of truncation (bootstrap, not reset) is preserved.
