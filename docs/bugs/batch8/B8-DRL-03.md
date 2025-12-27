# Finding Ticket: Sequential A/B Training Introduces Bias

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-03` |
| **Severity** | `P2` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/dual_ab.py` |
| **Line(s)** | `178-225` |
| **Function/Class** | `train_dual_ab()` |

---

## Summary

**One-line summary:** Sequential group training means later groups benefit from CUDA warmup and torch.compile caching.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [x] Performance bottleneck
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
# Groups are trained sequentially:
for group_id, group in groups.items():
    # First group: cold GPU, compilation overhead
    # Second group: warm GPU, cached compiled kernels
    train_ppo_vectorized(...)
```

The docstring acknowledges this is "Phase 1" with limitations, but the impact on results is non-trivial:

1. **CUDA warmup**: First group incurs JIT compilation, memory allocation
2. **torch.compile caching**: Compiled kernels are cached, benefiting later groups
3. **GPU memory fragmentation**: First group may have different allocation patterns

For rigorous A/B testing, these confounds can mask or amplify real reward mode differences.

### Impact

- **Confounded results**: Can't distinguish reward mode effects from execution order effects
- **Statistical invalidity**: A/B results may not generalize to production
- **Misleading conclusions**: Wrong reward mode might be selected

---

## Recommended Fix

**Option 1 - Randomize order:**
```python
import random
group_order = list(groups.keys())
random.shuffle(group_order)
for group_id in group_order:
    group = groups[group_id]
    ...
```

**Option 2 - Alternating lockstep (better):**
Train one episode from each group alternating to equalize warmup effects.

**Option 3 - True parallel (best):**
Use multiprocessing to train groups simultaneously on separate GPUs.

---

## Verification

### How to Verify the Fix

- [ ] Implement randomized group ordering
- [ ] Add metadata field for group training order
- [ ] Compare A/B results with shuffled vs fixed order
- [ ] Consider Phase 2 parallel implementation

---

## Related Findings

- All three reports noted this issue (CR, DRL, PT)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-04 - Sequential training confounds A/B comparison"
