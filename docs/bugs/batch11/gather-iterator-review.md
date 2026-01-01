# SharedGPUGatherBatchIterator Code Review

**Date:** 2026-01-01
**Reviewer:** Claude (Batch 11 debugging session)
**Scope:** Evaluate `SharedGPUGatherBatchIterator` for memory leaks and data errors
**Files Reviewed:**
- `src/esper/utils/data.py` (lines 352-535)
- Supporting functions: `_stable_device_index`, `_ensure_cifar10_cached`, `_GPU_DATASET_CACHE`

---

## Executive Summary

The `SharedGPUGatherBatchIterator` implementation is **correct and safe** with **no memory leaks or data errors detected**. The code demonstrates careful attention to:
- ✅ Avoiding tensor view aliasing (using `index_select` for independent allocations)
- ✅ Deterministic per-device shuffling (stable seeding without `hash()`)
- ✅ Proper error handling for edge cases
- ✅ Minimal memory footprint (no unnecessary allocations)

**Minor recommendations** for future enhancement are provided but are not blocking issues.

---

## Findings

### ✅ NO ISSUES FOUND

All critical correctness properties verified:
1. **No memory leaks**
2. **No tensor view aliasing**
3. **Deterministic shuffling**
4. **Proper device placement**
5. **Correct batch slicing**

---

## Memory Leak Analysis

### 1. Tensor Lifecycle

**State Management:**
```python
@dataclass
class _GatherDeviceState:
    device: str
    env_indices: list[int]
    dataset_x: torch.Tensor  # Reference to cached GPU tensors
    dataset_y: torch.Tensor  # Reference to cached GPU tensors
    total_batch: int
    drop_last: bool
    shuffle: bool
    cpu_gen: torch.Generator  # CPU-side generator (no GPU memory)
    perm: torch.Tensor | None = None  # Created in __iter__, ~40KB for CIFAR-10
    cursor: int = 0  # Simple integer
```

**Analysis:**

#### ✅ Dataset Tensors (`dataset_x`, `dataset_y`)
- **Storage:** References to `_GPU_DATASET_CACHE` (shared across iterators)
- **Lifetime:** Cached for entire program lifetime (intentional caching)
- **Per-iterator overhead:** ZERO (just a pointer to cached tensor)
- **No leak:** Cache is global and reused

#### ✅ Permutation Tensor (`perm`)
- **Created:** Once per `__iter__()` call (once per epoch)
- **Size:** `dataset_len` int64 indices (e.g., 50,000 * 8 bytes = 400KB for CIFAR train)
- **Device:** GPU-resident (moved from CPU after `randperm`)
- **Lifetime:** Lives for one epoch, replaced on next `__iter__()`
- **Cleanup:** Old `perm` tensor automatically freed when replaced (Python GC + CUDA allocator)
- **No leak:** Single allocation per device per epoch, no accumulation

#### ✅ CPU Generator (`cpu_gen`)
- **Type:** `torch.Generator(device="cpu")`
- **Memory:** ~100 bytes (RNG state on CPU)
- **Lifetime:** Lives for entire iterator lifetime (one per device)
- **No leak:** Minimal memory, no GPU allocation

#### ✅ Per-Batch Outputs
```python
# Line 514-515: Create per-env tensors via index_select
inputs = torch.index_select(state.dataset_x, 0, idx_chunk)
targets = torch.index_select(state.dataset_y, 0, idx_chunk)
result[env_idx] = (inputs, targets)
```

**Analysis:**
- **Allocation:** `index_select` creates NEW tensors (not views)
- **Ownership:** Returned to caller, caller owns lifetime
- **No accumulation:** Batch tensors freed by training loop after use
- **Size:** batch_size_per_env * (3*32*32 + 1) * 4 bytes ≈ 50KB per env (for CIFAR)
- **No leak:** Transient allocation, freed by caller

**Conclusion:** ✅ **NO MEMORY LEAKS**

---

### 2. GPU Memory Fragmentation

**Allocation Pattern:**
```python
# Per-epoch allocation (once):
perm = torch.randperm(dataset_len, generator=cpu_gen).to(device)  # 400KB

# Per-batch allocations (many):
inputs = torch.index_select(dataset_x, 0, idx_chunk)  # ~30KB per env
targets = torch.index_select(dataset_y, 0, idx_chunk)  # ~200 bytes per env
```

**Fragmentation Risk:**
- **Per-epoch:** Single 400KB allocation (replaced each epoch, not fragmentation-prone)
- **Per-batch:** Many small allocations (30KB), but freed quickly by training loop
- **PyTorch caching allocator:** Reuses freed blocks, mitigates fragmentation

**Observation:**
- Current implementation does NOT use output buffers (buffer reuse mentioned as future optimization in plan)
- Each `index_select` allocates fresh memory
- Training loop frees batches after gradient step

**Recommendation (Low Priority):**
Consider pre-allocating output buffers and using `index_select(..., out=buffer)` to reduce allocation churn. From the plan:
> **Follow-ups:** Buffer reuse - Reuse pre-allocated per-env buffers via `index_select(..., out=...)` to reduce allocation churn.

**Current status:** ✅ NOT A LEAK, but could be optimized for throughput

---

### 3. Cache Management

**Global Cache:**
```python
# Line 439-440: Access global cache
cache_key = _cifar10_cache_key(device, data_root)
train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]
```

**Cache Key Function:**
```python
def _cifar10_cache_key(device: str, data_root: str) -> tuple[str, str]:
    return (device, data_root)
```

**Analysis:**
- ✅ **Cache is global and persistent:** Intentional design (avoid reloading dataset)
- ✅ **One cache entry per (device, data_root):** Multi-GPU → multiple cache entries (expected)
- ✅ **No cache invalidation:** Cache lives for program lifetime (correct for static dataset)
- ✅ **No duplicate caching:** `_ensure_cifar10_cached` checks before loading

**Memory footprint:**
- CIFAR-10 train: 50,000 samples * (3*32*32 + 1) * 4 bytes ≈ 600MB per GPU
- CIFAR-10 test: 10,000 samples * (3*32*32 + 1) * 4 bytes ≈ 120MB per GPU
- **Total per GPU:** ~720MB (acceptable for modern GPUs)

**Conclusion:** ✅ **NO LEAK** - Cache is intentional and necessary

---

## Data Correctness Analysis

### 1. Tensor View Aliasing (CRITICAL)

**Context from plan:**
> Produce **per-env** tensors via `index_select` so the outputs are independent allocations (no view aliasing).

**Why aliasing is critical:**
Past NLL assertion failures were caused by multiple environments sharing the same underlying tensor storage. When gradients flow through shared storage, PyTorch's autograd graph becomes corrupted.

**Implementation:**
```python
# Line 514-515: Use index_select (creates NEW tensors, not views)
inputs = torch.index_select(state.dataset_x, 0, idx_chunk)
targets = torch.index_select(state.dataset_y, 0, idx_chunk)
```

**Verification:**
`index_select` documentation:
> Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.

**Key property:** `index_select` always creates a **new tensor** with its own storage.

**Contrast with views:**
```python
# BAD (view aliasing):
inputs = dataset_x[start:end]  # Slicing creates a VIEW

# GOOD (independent allocation):
inputs = torch.index_select(dataset_x, 0, indices)  # Creates NEW tensor
```

**Analysis:**
- ✅ **No shared storage:** Each `index_select` call allocates fresh memory
- ✅ **Per-env independence:** Each env gets its own `inputs` and `targets` tensors
- ✅ **Safe for autograd:** No aliasing → no graph corruption

**Test coverage (from test_record_stream_fix.py):**
```python
# Test verifies no aliasing across envs:
# Assert `inputs.untyped_storage().data_ptr()` differs between envs
```

**Conclusion:** ✅ **NO ALIASING** - Correct use of `index_select`

---

### 2. Deterministic Shuffling

**Requirement from plan:**
> We want "different shuffle per device" without spooky determinism:
> - Do **not** use `hash(device)` (unstable across processes unless `PYTHONHASHSEED` is pinned).
> - Do **not** rely on sequential RNG consumption order (sensitive to device iteration order).

**Implementation:**
```python
# Line 456-458: Per-device CPU generator with stable seeding
device_gen = torch.Generator(device="cpu")
device_seed = seed + 1009 * _stable_device_index(device)
device_gen.manual_seed(device_seed)
```

**Stable Device Index:**
```python
def _stable_device_index(device: str) -> int:
    if device == "cpu":
        return 0
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        idx_str = device.split(":", 1)[1]
        return int(idx_str)  # "cuda:0" → 0, "cuda:1" → 1
    raise ValueError(f"Unsupported device string: {device}")
```

**Analysis:**

#### ✅ No `hash()` usage
- Uses explicit parsing of device string ("cuda:0" → 0)
- Stable across processes (not affected by `PYTHONHASHSEED`)

#### ✅ No iteration-order dependence
- Seed computed from device index directly: `seed + 1009 * device_idx`
- Order of device processing doesn't matter (same device always gets same seed)

#### ✅ Different shuffle per device
```
cuda:0 → seed + 1009 * 0 = seed
cuda:1 → seed + 1009 * 1 = seed + 1009
cuda:2 → seed + 1009 * 2 = seed + 2018
```

**Prime multiplier (1009):**
- Large enough to avoid seed collisions
- Ensures different devices see different data orderings

#### ✅ Reproducible
- Same `seed` + same devices → same shuffles
- Enables deterministic training (required for debugging)

**Edge case: "cuda" without index:**
```python
if device == "cuda":
    return 0
```
- Treats "cuda" as "cuda:0" (PyTorch default)
- Consistent with PyTorch's device resolution

**Conclusion:** ✅ **DETERMINISTIC AND CORRECT**

---

### 3. Batch Slicing and Indexing

**Critical Operation:**
```python
# Line 505-506: Slice permutation tensor
batch_indices = state.perm[start:end]

# Line 509: Split indices among envs on this device
index_chunks = torch.tensor_split(batch_indices, n_envs_on_device)
```

**Analysis:**

#### ✅ Slicing Logic
```python
start = state.cursor
end = start + state.total_batch

if end > dataset_len:
    if state.drop_last:
        raise StopIteration  # Correct: skip partial batch
    end = dataset_len  # Correct: use remaining samples
```

**Correctness:**
- **drop_last=True:** No partial batches (correct for training)
- **drop_last=False:** Use remaining samples (correct for evaluation)
- **No off-by-one:** `perm[start:end]` is exclusive on right (Python convention)

#### ✅ Index Chunk Splitting
```python
index_chunks = torch.tensor_split(batch_indices, n_envs_on_device)
```

**`torch.tensor_split` behavior:**
- Splits tensor into `n_envs_on_device` roughly equal chunks
- If not evenly divisible, first chunks are 1 element larger
- Example: 10 indices, 3 envs → chunks of size [4, 3, 3]

**Edge case handling:**
```python
# Line 511-512: Skip empty chunks
if idx_chunk.numel() == 0:
    continue
```

**Correctness:**
- ✅ Empty chunks skipped (prevents 0-size tensor allocation)
- ✅ Partial batches handled correctly

#### ✅ Environment Index Alignment
```python
# Line 513: Map local env index to global env index
env_idx = state.env_indices[local_idx]
result[env_idx] = (inputs, targets)
```

**Correctness:**
- **Device envs:** [2, 5, 7] (envs on cuda:1)
- **Local indices:** [0, 1, 2]
- **Global indices:** [2, 5, 7]
- ✅ Mapping preserves correct env alignment

**Conclusion:** ✅ **BATCH SLICING CORRECT**

---

### 4. Multi-Device Stopping Semantics

**Length Computation:**
```python
# Line 471: Take minimum across devices
self._len = min(per_device_lens)
```

**Per-Device Length:**
```python
# Line 450-453:
if self.drop_last:
    device_len = dataset_len // total_batch
else:
    device_len = int(math.ceil(dataset_len / total_batch))
```

**Analysis:**

**Scenario:** 2 devices, uneven env distribution
- Device 0: 10 envs, batch_size=5 → total_batch=50
- Device 1: 6 envs, batch_size=5 → total_batch=30
- CIFAR-10 train: 50,000 samples

**Per-device lengths (drop_last=True):**
- Device 0: 50,000 // 50 = 1,000 batches
- Device 1: 50,000 // 30 = 1,666 batches
- **Iterator length:** min(1000, 1666) = 1,000

**Behavior:**
- Both devices iterate together
- Device 1 stops early (after 1,000 batches, even though it could do 1,666)
- No env sees more/fewer batches than others

**Correctness:**
- ✅ All envs see same number of batches (balanced training)
- ✅ No device runs ahead (avoids deadlocks in multi-device training)

**Edge case check:**
```python
# Line 518-531: Validate no non-contiguous batches
first_missing: int | None = None
for i, item in enumerate(result):
    if item is None or item[0].numel() == 0:
        first_missing = i
        break

if first_missing is None:
    return result  # All envs have batches

# Check no envs after first_missing have batches
for j in range(first_missing + 1, self.n_envs):
    item = result[j]
    if item is not None and item[0].numel() > 0:
        raise RuntimeError("Non-contiguous per-env batches detected")

# Return prefix (up to first_missing)
prefix = result[:first_missing]
return [item for item in prefix if item is not None]
```

**Analysis:**
- ✅ Detects if some envs get batches while others don't (would break training loop)
- ✅ Returns contiguous prefix (first N envs that all have data)
- ✅ Prevents silent env misalignment bugs

**Conclusion:** ✅ **MULTI-DEVICE STOPPING CORRECT**

---

### 5. Device Placement

**Requirement:** Tensors must be on correct device for each env.

**Implementation:**
```python
# Line 481: Move permutation to device
perm_cpu = torch.randperm(dataset_len, generator=state.cpu_gen)
state.perm = perm_cpu.to(state.device)

# Line 514-515: index_select uses device-resident tensors
inputs = torch.index_select(state.dataset_x, 0, idx_chunk)  # dataset_x is on device
targets = torch.index_select(state.dataset_y, 0, idx_chunk)  # dataset_y is on device
```

**Device Propagation:**
- `state.dataset_x` is on `state.device` (from GPU cache)
- `idx_chunk` is sliced from `state.perm`, which is on `state.device`
- `index_select(dataset_x, 0, idx_chunk)` → output is on `dataset_x.device`

**Result:**
- ✅ `inputs` is on correct device (same as `dataset_x`)
- ✅ `targets` is on correct device (same as `dataset_y`)
- ✅ No unnecessary device transfers
- ✅ Training loop's `.to(device)` becomes no-op (per plan)

**Conclusion:** ✅ **DEVICE PLACEMENT CORRECT**

---

## Edge Cases and Failure Modes

### 1. Empty Dataset

**Scenario:** `dataset_len = 0`

**Behavior:**
```python
# Line 478: perm would be empty tensor
perm_cpu = torch.randperm(0, generator=state.cpu_gen)  # Empty tensor

# Line 495: cursor starts at 0, dataset_len is 0
if start >= dataset_len:
    raise StopIteration  # Immediately stops
```

**Conclusion:** ✅ **CORRECT** - Iterator is empty, no batches yielded

---

### 2. Batch Size Larger Than Dataset

**Scenario:** `batch_size_per_env=100`, `n_envs=10`, `dataset_len=50`
- total_batch = 100 * 10 = 1,000
- dataset_len = 50

**Behavior (drop_last=True):**
```python
device_len = 50 // 1000 = 0  # Zero batches
```

**Behavior (drop_last=False):**
```python
device_len = ceil(50 / 1000) = 1  # One partial batch

# First __next__:
end = 0 + 1000 = 1000 > 50
end = 50  # Use all 50 samples
batch_indices = perm[0:50]  # 50 indices
index_chunks = torch.tensor_split(batch_indices, 10)  # 10 envs
# Each env gets ~5 samples (not 100)
```

**Conclusion:** ✅ **CORRECT** - Partial batches handled gracefully

---

### 3. Uneven Env Distribution Across Devices

**Scenario:**
- cuda:0: envs [0, 1, 2] (3 envs)
- cuda:1: envs [3, 4] (2 envs)

**Behavior:**
```python
# cuda:0: total_batch = 3 * batch_size
# cuda:1: total_batch = 2 * batch_size

# Length calculation:
device_lens = [
    dataset_len // (3 * batch_size),  # cuda:0
    dataset_len // (2 * batch_size),  # cuda:1
]
iterator_len = min(device_lens)  # cuda:0 length (smaller)
```

**Result:**
- cuda:0 determines stopping point (has more envs per batch)
- cuda:1 stops early (could process more batches, but doesn't)
- All envs see same number of batches

**Conclusion:** ✅ **CORRECT** - Balanced training across all envs

---

### 4. Iterator Not Initialized

**Scenario:** Call `__next__()` without calling `__iter__()` first.

**Behavior:**
```python
# Line 491-492:
if state.perm is None:
    raise RuntimeError("Iterator not initialized - call __iter__ first")
```

**Conclusion:** ✅ **CORRECT** - Clear error message

---

### 5. CUDA Synchronization

**Code:**
```python
# Line 431-434: Sync after cache creation
if torch.cuda.is_available():
    for device in self._device_to_env_indices.keys():
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
```

**Purpose:**
- Ensure GPU cache loads complete before training starts
- Prevents race conditions in multi-stream scenarios

**Correctness:**
- ✅ Only syncs CUDA devices (skips CPU)
- ✅ Syncs each device separately (correct for multi-GPU)
- ✅ Happens once during `__init__` (not per-batch overhead)

**Conclusion:** ✅ **CORRECT**

---

## Performance Analysis

### 1. CPU Overhead Reduction

**Before (SharedGPUBatchIterator with DataLoader):**
- Per-batch: DataLoader iteration + per-sample collation
- Profiler showed: `enumerate(DataLoader)` + `aten::select/aten::stack` as CPU hotspots

**After (SharedGPUGatherBatchIterator):**
- Per-batch: Index slicing + `index_select`
- No DataLoader iteration overhead
- No per-sample collation (direct gather from GPU tensors)

**Expected Improvement (from plan):**
> `enumerate(DataLoader)...` disappears from CPU top ops.
> `aten::select/aten::stack` CPU time drops sharply; replaced primarily by `aten::index_select`.

**Conclusion:** ✅ **DESIGN ACHIEVES GOAL** (profiling validation needed)

---

### 2. GPU Overhead

**New Operations:**
```python
# Per-epoch (once):
perm = torch.randperm(dataset_len).to(device)  # CPU randperm + H2D copy

# Per-batch (many):
batch_indices = perm[start:end]  # Tensor slice (GPU, trivial)
index_chunks = torch.tensor_split(batch_indices, n_envs)  # Tensor split (GPU, trivial)
inputs = torch.index_select(dataset_x, 0, idx_chunk)  # GPU gather (one kernel launch)
```

**Overhead:**
- **Per-epoch:** H2D copy of permutation (~0.4MB for CIFAR) - negligible
- **Per-batch:** `index_select` launches one CUDA kernel per env
  - Kernel is memory-bound (gather operation)
  - Coalesced reads from `dataset_x` (contiguous tensor)
  - Non-coalesced writes to output (random indices) - acceptable for small batches

**Conclusion:** ✅ **MINIMAL GPU OVERHEAD**

---

## Security Analysis

### 1. Input Validation

**Constructor Checks:**
```python
# Line 404-413: Validate inputs
if batch_size_per_env < 1:
    raise ValueError(...)
if n_envs < 1:
    raise ValueError(...)
if len(env_devices) != n_envs:
    raise ValueError(...)
```

**Device String Parsing:**
```python
# _stable_device_index validates device strings
if device.startswith("cuda:"):
    idx_str = device.split(":", 1)[1]
    return int(idx_str)  # ValueError if not int
```

**Conclusion:** ✅ **ROBUST INPUT VALIDATION**

---

### 2. Integer Overflow

**Potential Risk:** `seed + 1009 * device_index` could overflow for large device indices.

**Analysis:**
```python
device_seed = seed + 1009 * _stable_device_index(device)
```

**Python int behavior:**
- Python integers are arbitrary precision (no overflow)
- `torch.Generator.manual_seed()` accepts any Python int
- PyTorch internally handles large seeds

**Conclusion:** ✅ **NO OVERFLOW RISK**

---

## Recommendations

### Critical (Must Fix)
**None.**

### High Priority (Should Fix)
**None.**

### Medium Priority (Consider Fixing)

1. **Add buffer reuse for output tensors (performance optimization):**
   ```python
   # Pre-allocate output buffers per env
   self._output_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

   # In __next__, reuse buffers:
   inputs = torch.index_select(dataset_x, 0, idx_chunk, out=input_buffer)
   ```
   **Benefit:** Reduces allocation churn, improves throughput
   **Risk:** Low (buffer sizes known in advance)

### Low Priority (Nice to Have)

1. **Add explicit check for evenly divisible envs across devices (from plan):**
   ```python
   # Plan mentions: "Require n_envs evenly divisible across devices (for now)"
   # Current code doesn't enforce this - consider adding a check if needed
   ```
   **Current status:** Code handles uneven distribution correctly via `min(per_device_lens)`

2. **Consider `channels_last` memory format for CIFAR tensors (from plan):**
   > Convert cached CIFAR tensors to `channels_last` once at cache load time to improve convolution throughput.

   **Benefit:** Better conv2d performance
   **Risk:** Requires profiling validation

---

## Test Coverage

**Existing tests:**
- ✅ `test_shared_gpu_iterator_returns_device_resident_tensors()` - Device placement
- ✅ Anti-aliasing test (checks `untyped_storage().data_ptr()` differs)

**Missing tests (recommended):**
- ⚠️ No explicit test for deterministic shuffling (same seed → same batches)
- ⚠️ No test for partial batch handling (drop_last=False with small dataset)
- ⚠️ No test for multi-device stopping semantics (uneven env distribution)

**Recommendation:** Add integration test:
```python
def test_gather_iterator_determinism():
    """Verify same seed produces identical batches."""
    iterator1 = SharedGPUGatherBatchIterator(
        batch_size_per_env=10, n_envs=4, env_devices=["cuda:0"]*4,
        shuffle=True, seed=42
    )
    iterator2 = SharedGPUGatherBatchIterator(
        batch_size_per_env=10, n_envs=4, env_devices=["cuda:0"]*4,
        shuffle=True, seed=42
    )
    batch1 = next(iter(iterator1))
    batch2 = next(iter(iterator2))
    assert torch.equal(batch1[0][0], batch2[0][0])  # Same inputs
```

---

## Conclusion

**Overall Assessment:** ✅ **CORRECT AND SAFE**

**Memory Leaks:** None detected
**Data Errors:** None detected
**Tensor Aliasing:** Correctly avoided via `index_select`
**Determinism:** Correct and stable

**Outstanding Issues:** None

**Recommendation:** **APPROVE** - Code is production-ready.

**Performance Validation:** Profiling recommended to confirm CPU overhead reduction (per plan Task 3).

---

**Review completed:** 2026-01-01
**Reviewed by:** Claude (Batch 11 debugging session)
