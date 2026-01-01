# GPU Gather Iterator: Buffer Reuse Implementation Plan

**Date:** 2026-01-01
**Status:** ❌ ABANDONED (2026-01-01) - Race condition identified
**Prerequisites:** None (builds on existing `SharedGPUGatherBatchIterator`)
**Expected Speedup:** 10-20%
**Estimated Time:** 2-3 hours

---

## Reason for Abandonment

**Critical race condition identified during implementation:**

The training loop (`train_ppo_vectorized`) pipelines batches and only syncs once per epoch, not per batch. This means `__next__()` can overwrite a reused buffer while CUDA kernels from the previous step are still reading it - reintroducing the same class of corruption this gather work was designed to avoid.

**Safe alternatives (not implemented):**

- **Ring buffers + CUDA event fencing:** Per-env ring buffer (≥2 slots) with producer/consumer events
- **Move index_select to env compute stream:** Loses producer/consumer overlap

The CUDA caching allocator already provides stream-aware memory reuse when tensors are freed, so the marginal benefit doesn't justify the complexity.

---

## Original Overview

Eliminate per-batch allocation overhead by pre-allocating output buffers and reusing them via `torch.index_select(..., out=buffer)`.

**Current behavior:** 16 allocations per batch (8 envs × 2 tensors)
**Target behavior:** 0 allocations per batch (pre-allocated buffers reused)

---

## Task 1: Add Buffer State to `_GatherDeviceState`

**File:** `src/esper/utils/data.py`

### 1.1 Extend dataclass with buffer references

Add new fields to hold pre-allocated buffers:

```python
@dataclass
class _GatherDeviceState:
    device: str
    env_indices: list[int]
    dataset_x: torch.Tensor
    dataset_y: torch.Tensor
    total_batch: int
    drop_last: bool
    shuffle: bool
    cpu_gen: torch.Generator
    perm: torch.Tensor | None = None
    cursor: int = 0
    # NEW: Pre-allocated output buffers (one per env on this device)
    input_buffers: list[torch.Tensor] | None = None   # Shape: (batch_size_per_env, C, H, W)
    target_buffers: list[torch.Tensor] | None = None  # Shape: (batch_size_per_env,)
```

### 1.2 Location in file

Line ~371-382 (modify existing `_GatherDeviceState` dataclass)

---

## Task 2: Allocate Buffers in `__init__`

**File:** `src/esper/utils/data.py`

### 2.1 Add buffer allocation after device state creation

After creating `_GatherDeviceState` objects (~line 460-469), allocate buffers:

```python
# After self._device_states[device] = _GatherDeviceState(...)

# Allocate output buffers for each env on this device
n_envs_on_device = len(env_indices)
input_shape = (batch_size_per_env,) + dataset_x.shape[1:]  # (B, C, H, W)
target_shape = (batch_size_per_env,)

input_buffers = [
    torch.empty(input_shape, dtype=dataset_x.dtype, device=device)
    for _ in range(n_envs_on_device)
]
target_buffers = [
    torch.empty(target_shape, dtype=dataset_y.dtype, device=device)
    for _ in range(n_envs_on_device)
]

self._device_states[device].input_buffers = input_buffers
self._device_states[device].target_buffers = target_buffers
```

### 2.2 Handle drop_last=False case

For variable batch sizes (test set with drop_last=False), buffers may be larger than needed:

```python
# Use max possible batch size for buffer allocation
if self.drop_last:
    # Fixed size: batch_size_per_env
    buffer_size = batch_size_per_env
else:
    # Variable size: allocate for max, slice when smaller
    buffer_size = batch_size_per_env
    # NOTE: Last batch may be smaller - we'll slice the buffer
```

---

## Task 3: Modify `__next__` to Use Buffers

**File:** `src/esper/utils/data.py`

### 3.1 Replace allocation with buffer reuse

Current code (~lines 510-516):
```python
for local_idx, idx_chunk in enumerate(index_chunks):
    if idx_chunk.numel() == 0:
        continue
    env_idx = state.env_indices[local_idx]
    inputs = torch.index_select(state.dataset_x, 0, idx_chunk)
    targets = torch.index_select(state.dataset_y, 0, idx_chunk)
    result[env_idx] = (inputs, targets)
```

New code:
```python
for local_idx, idx_chunk in enumerate(index_chunks):
    if idx_chunk.numel() == 0:
        continue
    env_idx = state.env_indices[local_idx]

    # Get pre-allocated buffers for this env (within this device)
    input_buffer = state.input_buffers[local_idx]
    target_buffer = state.target_buffers[local_idx]

    batch_size = idx_chunk.numel()

    if batch_size == input_buffer.size(0):
        # Full batch: use buffer directly
        torch.index_select(state.dataset_x, 0, idx_chunk, out=input_buffer)
        torch.index_select(state.dataset_y, 0, idx_chunk, out=target_buffer)
        result[env_idx] = (input_buffer, target_buffer)
    else:
        # Partial batch (drop_last=False): slice buffer to actual size
        input_view = input_buffer[:batch_size]
        target_view = target_buffer[:batch_size]
        torch.index_select(state.dataset_x, 0, idx_chunk, out=input_view)
        torch.index_select(state.dataset_y, 0, idx_chunk, out=target_view)
        result[env_idx] = (input_view, target_view)
```

### 3.2 Why this is safe

**No aliasing across environments:**
- Each env has its own buffer (separate allocations)
- `input_buffers[0]` and `input_buffers[1]` have different underlying storage

**No aliasing across batches:**
- Training loop consumes batch, computes gradients, then requests next batch
- Buffer is overwritten on next `__next__` call
- No reference to old data survives (verified in `vectorized.py` training loop)

**Slice safety for partial batches:**
- `input_buffer[:batch_size]` creates a view of the first `batch_size` elements
- `out=` parameter writes to this view correctly
- Training loop receives correctly-sized tensor

---

## Task 4: Add Safety Check (Optional but Recommended)

**File:** `src/esper/utils/data.py`

### 4.1 Add buffer ownership assertion

In `__next__`, before overwriting buffers, optionally check that previous batch was consumed:

```python
# Optional: Verify training loop consumed previous batch
# Uncomment for debugging buffer lifecycle issues
# if __debug__:
#     for local_idx in range(n_envs_on_device):
#         buf = state.input_buffers[local_idx]
#         # If refcount > 2 (buffer + state dict), something is holding a reference
#         # This would indicate a bug in the training loop
#         import sys
#         assert sys.getrefcount(buf) <= 3, f"Buffer {local_idx} still referenced"
```

**Note:** This is expensive in debug mode. Use only for diagnosing issues.

---

## Task 5: Update Tests

**File:** `tests/simic/test_record_stream_fix.py`

### 5.1 Verify buffer reuse doesn't break aliasing guarantees

Add test that verifies consecutive batches don't alias:

```python
def test_buffer_reuse_no_aliasing_across_batches():
    """Verify that buffer reuse doesn't cause batch-to-batch aliasing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=10,
        n_envs=4,
        env_devices=["cuda:0"] * 4,
        shuffle=True,
        seed=42,
    )

    it = iter(iterator)
    batch1 = next(it)

    # Store references to first batch tensors
    batch1_inputs = [b[0].clone() for b in batch1]
    batch1_data_ptrs = [b[0].data_ptr() for b in batch1]

    batch2 = next(it)
    batch2_data_ptrs = [b[0].data_ptr() for b in batch2]

    # With buffer reuse, data_ptrs SHOULD be the same (same buffers reused)
    # But the DATA should be different (new indices gathered)
    assert batch1_data_ptrs == batch2_data_ptrs, "Buffers should be reused"

    for i, (b1, b2) in enumerate(zip(batch1_inputs, batch2)):
        # Original data (cloned before overwrite) should differ from current buffer
        assert not torch.equal(b1, b2[0]), f"Batch data should differ for env {i}"
```

### 5.2 Verify partial batch handling

Add test for drop_last=False case:

```python
def test_buffer_reuse_partial_batch():
    """Verify buffer reuse handles partial batches correctly (drop_last=False)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Use small batch size to force partial final batch
    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=1000,  # Large batch relative to test set
        n_envs=2,
        env_devices=["cuda:0"] * 2,
        shuffle=False,
        is_train=False,  # Uses test set with drop_last=False
        seed=42,
    )

    batches = list(iterator)

    # Last batch may be smaller
    last_batch = batches[-1]
    for inputs, targets in last_batch:
        # Verify tensor is correctly sized (may be smaller than buffer)
        assert inputs.size(0) <= 1000
        assert targets.size(0) <= 1000
        assert inputs.size(0) == targets.size(0)
```

---

## Task 6: Verify Training Loop Compatibility

**File:** `src/esper/simic/training/vectorized.py`

### 6.1 Check batch consumption pattern

Verify that the training loop doesn't hold references to old batches:

```python
# Current pattern in vectorized.py (simplified):
for batch_data in shared_iterator:
    for env_idx, (inputs, targets) in enumerate(batch_data):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # After this point, inputs/targets are no longer needed
        # Python GC + CUDA allocator will handle cleanup

    # PPO update
    agent.update()

    # Next iteration: shared_iterator.__next__() overwrites buffers
    # This is safe because we're done with previous batch
```

**Verification:** The training loop doesn't store batch tensors in lists or dicts that persist across iterations. Each batch is consumed immediately.

---

## Implementation Order

1. **Modify `_GatherDeviceState`** (Task 1) - Add buffer fields
2. **Allocate buffers in `__init__`** (Task 2) - Create buffers once
3. **Modify `__next__`** (Task 3) - Use `out=` parameter
4. **Run existing tests** - Verify no regressions
5. **Add new tests** (Task 5) - Buffer reuse + partial batch
6. **Profile before/after** - Measure allocation overhead reduction

---

## Rollback Plan

If issues are discovered:
1. Revert to original `index_select` without `out=` parameter
2. Remove buffer allocation from `__init__`
3. Remove buffer fields from `_GatherDeviceState`

Changes are isolated to `data.py` and easily reversible.

---

## Success Criteria

1. **All existing tests pass**
2. **New buffer reuse tests pass**
3. **Profiler shows no `aten::empty` calls in `__next__` hot path**
4. **End-to-end throughput improvement measurable (10-20%)**
