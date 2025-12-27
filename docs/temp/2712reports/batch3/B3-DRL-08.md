# Finding Ticket: FlexAttention Block Mask Cache May Accumulate

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-08` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blueprints/transformer.py` |
| **Line(s)** | `238` |
| **Function/Class** | `FlexAttentionSeed._block_mask_cache` |

---

## Summary

**One-line summary:** FlexAttention block mask cache uses LRU bound of 8, but block masks can be large and may accumulate GPU memory with varying sequence lengths.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [x] Memory leak / resource issue
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

`_MAX_CACHE_SIZE = 8` bounds the LRU cache, but:
- Block masks are proportional to sequence length
- Training with varying sequence lengths accumulates up to 8 different masks
- The `_apply()` hook clears cache on device transfer but not on dtype changes via autocast

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blueprints/transformer.py:238

_MAX_CACHE_SIZE = 8  # LRU cache bound
```

### Why This Matters

For training runs with highly variable sequence lengths, this could hold onto 8 large block masks unnecessarily.

---

## Recommended Fix

Consider:
1. Clearing caches at epoch boundaries
2. Adding dtype to cache key to prevent stale entries under autocast
3. Reducing cache size if memory pressure is observed

```python
def _apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
    # Clear cache on any _apply call (device AND dtype changes)
    self._block_mask_cache.clear()
    return super()._apply(fn)
```

---

## Verification

### How to Verify the Fix

- [ ] Profile block mask memory with varying sequence lengths
- [ ] Test cache behavior under autocast dtype changes
- [ ] Monitor cache size during training

---

## Related Findings

None.

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| **DRL Specialist** | **ENDORSE** | Legitimate memory concern for variable-length RL sequences. Adding dtype to cache key is the right fix; 8-mask LRU is already conservative but worth monitoring under sequence length variance. |
| **PyTorch Specialist** | **ENDORSE** | Valid P2: block masks store O(seq_len^2/block_size^2) metadata. Cache key already includes dtype (line 255); the real issue is `_apply()` not clearing on `autocast()` dtype changes. Fix is straightforward. |
| **Code Review Specialist** | **ENDORSE** | Confirmed P2. The `_apply()` override (line 240-243) clears cache but autocast bypasses `_apply()`. Recommend epoch-boundary cache clear or reducing `_MAX_CACHE_SIZE`. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P2 - Performance/Safety" (B3-TFM-01)
