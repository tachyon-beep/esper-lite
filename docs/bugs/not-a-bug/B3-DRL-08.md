# Finding Ticket: FlexAttention Block Mask Cache May Accumulate

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-08` |
| **Severity** | `P2` |
| **Status** | `invalid` |
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

---

## Resolution

### Status: NOT-A-BUG

**Closed via Systematic Debugging investigation.**

#### Evidence Table

| Claim | Status | Evidence |
|-------|--------|----------|
| "Block masks can be large and accumulate GPU memory" | ❌ FALSE | Block masks use int32 indices: ~1KB for seq_len=1024, ~8KB total with 8 cache entries |
| "Block masks are proportional to sequence length" | ⚠️ MISLEADING | O(seq²/block_size²), not O(seq²); seq=1024 → only 256 elements |
| "autocast bypasses _apply()" | ✅ TRUE | But irrelevant - block masks are dtype-agnostic |
| "dtype changes cause stale entries" | ❌ FALSE | Tested: same block mask works with float32/float16/bfloat16 |
| "_MAX_CACHE_SIZE = 8 is a memory concern" | ❌ FALSE | 8KB total is negligible vs. model weights/activations |

#### Why This Is Not A Bug

1. **Block masks are NOT attention matrices:** They store sparse block indices (int32), not O(seq²) floating-point tensors. Memory is O((seq/block_size)²) ≈ 64 elements for seq=1024.

2. **Block masks are dtype-agnostic:** `create_block_mask()` has no dtype parameter. Empirically verified: same mask works across float32/float16/bfloat16.

3. **Cache is already well-bounded:** `_MAX_CACHE_SIZE = 8` limits total memory to ~8KB regardless of sequence length variation.

4. **Including dtype in cache key is conservative but harmless:** Creates at most 2-3 redundant entries (float32/float16/bfloat16) but doesn't cause correctness issues.

5. **The `_apply()` method correctly clears cache on `.to()` calls:** Autocast doesn't call `_apply()` but doesn't need to - block masks don't change with dtype.

#### Severity Downgrade

- Original: P2 (memory leak / resource issue)
- Revised: N/A (not a bug)
- Resolution: NOT-A-BUG - claims based on misunderstanding of BlockMask structure

#### Sources

- [PyTorch FlexAttention Documentation](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [FlexAttention Blog Post](https://pytorch.org/blog/flexattention/)
