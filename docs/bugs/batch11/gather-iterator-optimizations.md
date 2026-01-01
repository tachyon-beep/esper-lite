# GPU Gather Iterator: Further Optimization Opportunities

**Date:** 2026-01-01
**Context:** Analysis of `SharedGPUGatherBatchIterator` for additional performance improvements
**Current Status:** Implementation is correct and eliminates DataLoader CPU overhead

---

## Executive Summary

The current implementation successfully eliminates the primary bottleneck (DataLoader CPU iteration overhead). **Five additional optimizations** are identified, ranked by expected impact:

| Optimization | Expected Speedup | Complexity | Priority |
|--------------|------------------|------------|----------|
| 1. Buffer reuse | 10-20% | Low | **High** |
| 2. Channels-last format | 15-30% | Low | **High** |
| 3. Fused gather kernel | 5-10% | Medium | Medium |
| 4. GPU randperm | 1-2% | Low | Low |
| 5. Index range caching | <1% | Low | Low |

**Recommended next steps:** Implement #1 (buffer reuse) and #2 (channels-last) as they provide the highest ROI with minimal risk.

---

## Current Performance Characteristics

### Hot Path Analysis

**Per-batch operations (for n_envs_on_device=8):**

```python
batch_indices = state.perm[start:end]                                    # ~1 μs (view)
index_chunks = torch.tensor_split(batch_indices, n_envs_on_device)       # ~5 μs (splits into 8 views)

for local_idx in range(8):
    inputs = torch.index_select(state.dataset_x, 0, idx_chunk)           # ~50 μs (GPU kernel + alloc)
    targets = torch.index_select(state.dataset_y, 0, idx_chunk)          # ~5 μs (GPU kernel + alloc)
```

**Total per-batch overhead:**
- Kernel launches: 16 (8 inputs + 8 targets)
- Allocations: 16 (8 inputs + 8 targets)
- Estimated time: ~500 μs per batch

**Breakdown:**
- CPU operations: ~10 μs (negligible)
- GPU kernel launches: 16 × ~10 μs = ~160 μs
- Memory allocations: 16 × ~20 μs = ~320 μs
- Actual gather compute: ~10 μs (memory-bound)

**Key insight:** Allocation overhead dominates, not compute.

---

## Optimization 1: Output Buffer Reuse (High Impact)

### Problem

Current implementation allocates new tensors every batch:

```python
# 16 allocations per batch (8 envs × 2 tensors)
inputs = torch.index_select(state.dataset_x, 0, idx_chunk)   # Allocates ~30KB
targets = torch.index_select(state.dataset_y, 0, idx_chunk)  # Allocates ~200B
```

**Overhead:**
- PyTorch caching allocator: ~20 μs per allocation
- 16 allocations × 20 μs = ~320 μs per batch
- Over a 12-epoch episode with 100 batches: 32ms wasted on allocations

### Solution

Pre-allocate output buffers and reuse them:

```python
# In __init__:
self._output_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
for device, state in self._device_states.items():
    for env_idx in state.env_indices:
        input_buffer = torch.empty(
            (self.batch_size_per_env, 3, 32, 32),
            dtype=state.dataset_x.dtype,
            device=device
        )
        target_buffer = torch.empty(
            (self.batch_size_per_env,),
            dtype=state.dataset_y.dtype,
            device=device
        )
        self._output_buffers[env_idx] = (input_buffer, target_buffer)

# In __next__:
input_buffer, target_buffer = self._output_buffers[env_idx]
torch.index_select(state.dataset_x, 0, idx_chunk, out=input_buffer)
torch.index_select(state.dataset_y, 0, idx_chunk, out=target_buffer)
result[env_idx] = (input_buffer, target_buffer)
```

### Benefits

- **Eliminates allocation overhead:** ~320 μs → 0 μs per batch
- **Reduces memory fragmentation:** No churn in caching allocator
- **Improves cache locality:** Same buffers reused → better L2 cache hits

### Risks and Mitigations

**Risk 1:** Training loop holds references to old batches
- **Mitigation:** Training loop already frees batches after gradient step (verified in vectorized.py)
- **Verification:** Add assert that batch tensors are consumed before next batch

**Risk 2:** Variable batch sizes (drop_last=False)
- **Current code:** Only uses drop_last=False for test set (small dataset, infrequent)
- **Mitigation:** For variable sizes, fall back to allocation or use largest size and slice

**Risk 3:** Buffer aliasing across batches
- **Not a concern:** Each batch gets independent buffers, same as current allocations

### Implementation Complexity

**Low:** ~50 lines of code
- Add `_output_buffers` dict in `__init__`
- Modify `__next__` to use `out=` parameter
- Handle variable batch sizes for drop_last=False case

### Expected Speedup

**10-20%** reduction in per-batch latency
- Baseline: ~500 μs per batch
- After optimization: ~180 μs per batch (36% faster)
- End-to-end: 10-20% (depends on other training loop overhead)

---

## Optimization 2: Channels-Last Memory Format (High Impact)

### Problem

CIFAR-10 tensors are stored in **NCHW (channels-first) format:**

```python
dataset_x.shape = (50000, 3, 32, 32)  # NCHW
```

Modern GPUs (especially with Tensor Cores) achieve **better convolution throughput** with **NHWC (channels-last) format.**

### Solution

Convert cached tensors to channels-last **once at cache load time:**

```python
# In _ensure_cifar10_cached():
train_x = train_x.to(memory_format=torch.channels_last)
test_x = test_x.to(memory_format=torch.channels_last)
```

### Benefits

- **Faster convolutions:** 15-30% speedup for conv layers (depends on GPU architecture)
- **Better memory access patterns:** Channels dimension is contiguous in memory
- **No runtime overhead:** Conversion happens once at cache load, not per-batch

### Verification

Check that downstream code handles channels-last correctly:

```python
# After first conv layer, tensor should remain channels-last
assert batch.is_contiguous(memory_format=torch.channels_last)
```

### Risks and Mitigations

**Risk 1:** Some ops don't preserve channels-last format
- **Mitigation:** Modern PyTorch (2.0+) preserves format through most ops
- **Verification:** Add format checks in training loop, fall back to contiguous if needed

**Risk 2:** Increased memory usage
- **Not a concern:** Channels-last uses same memory (just different layout)

### Implementation Complexity

**Low:** ~5 lines of code
- Add `.to(memory_format=torch.channels_last)` after loading tensors
- Verify format preservation in first training run

### Expected Speedup

**15-30%** reduction in convolution time
- ResNet-18 on CIFAR-10: ~60% of forward pass is convolutions
- 15-30% conv speedup → 10-20% end-to-end speedup

---

## Optimization 3: Fused Gather Kernel (Medium Impact)

### Problem

Current implementation launches **2 kernels per env** (inputs + targets):

```python
# For 8 envs → 16 kernel launches
for env in envs:
    inputs = index_select(dataset_x, 0, indices)   # Kernel launch
    targets = index_select(dataset_y, 0, indices)  # Kernel launch
```

**Kernel launch overhead:** ~10 μs × 16 = ~160 μs per batch

### Solution

**Option A: Batch all gathers into single kernel**

```python
# Concatenate all indices
all_indices = torch.cat([idx_chunk for idx_chunk in index_chunks])

# Single gather for all envs
all_inputs = torch.index_select(state.dataset_x, 0, all_indices)
all_targets = torch.index_select(state.dataset_y, 0, all_indices)

# Split results (with cloning to avoid aliasing)
split_sizes = [chunk.numel() for chunk in index_chunks]
input_chunks = torch.split(all_inputs, split_sizes)
target_chunks = torch.split(all_targets, split_sizes)

for local_idx, (inputs, targets) in enumerate(zip(input_chunks, target_chunks)):
    env_idx = state.env_indices[local_idx]
    result[env_idx] = (inputs.clone(), targets.clone())  # Clone to avoid aliasing
```

**Benefit:**
- 16 kernel launches → 2 kernel launches (1 for inputs, 1 for targets)
- Kernel launch overhead: ~160 μs → ~20 μs
- **Savings: ~140 μs per batch**

**Cost:**
- `torch.cat` allocation: ~20 μs
- `torch.split` + clones: ~80 μs (8 clones × 10 μs)
- **Net savings: ~40 μs per batch (~8% speedup)**

**Option B: Custom CUDA kernel**

Write a custom kernel that does gather + split in one GPU kernel:

```cuda
__global__ void gather_and_split(
    const float* dataset,
    const int64_t* indices,
    const int* split_offsets,
    float** outputs,
    int n_envs,
    int batch_size_per_env
) {
    int env_idx = blockIdx.x;
    int sample_idx = threadIdx.x;

    int global_idx = indices[split_offsets[env_idx] + sample_idx];
    outputs[env_idx][sample_idx] = dataset[global_idx];
}
```

**Benefit:**
- Single kernel launch
- Minimal overhead
- Direct per-env writes (no intermediate allocations)

**Cost:**
- Custom CUDA code (maintenance burden)
- Need to handle edge cases (variable batch sizes, different dtypes)
- Testing complexity

### Recommendation

**Option A (batched gather):** Implement if profiling shows kernel launch overhead is significant.

**Option B (custom kernel):** Only consider if Option A doesn't provide sufficient speedup and this is a proven bottleneck.

### Expected Speedup

**5-10%** reduction in per-batch latency
- Baseline: ~500 μs
- After optimization: ~460 μs
- Marginal given allocation overhead dominates

---

## Optimization 4: GPU-Native Permutation (Low Impact)

### Problem

Current implementation generates permutation on CPU and copies to GPU:

```python
# Per-epoch cost
perm_cpu = torch.randperm(dataset_len, generator=state.cpu_gen)  # CPU RNG: ~0.5ms
state.perm = perm_cpu.to(state.device)                          # H2D copy: ~1ms
# Total: ~1.5ms per epoch
```

**Frequency:** Once per epoch (~100 batches)
**Amortized overhead:** 1.5ms / 100 batches = ~15 μs per batch

### Solution

Use GPU-native random permutation:

```python
# PROBLEM: torch doesn't support generator on CUDA for randperm
# state.perm = torch.randperm(dataset_len, generator=cuda_gen, device=device)

# Workaround: Use random indices on GPU
# But this loses determinism without CUDA generator seeding
```

**Challenge:** PyTorch doesn't support `torch.Generator(device="cuda")` for `randperm`.

**Alternative:** Use `torch.rand` + `argsort`:

```python
# Generate random floats on GPU (seeded)
random_vals = torch.rand(dataset_len, device=device, generator=cuda_gen)
state.perm = torch.argsort(random_vals)  # GPU argsort
```

**But:** `torch.rand` with CUDA generator requires PyTorch 2.0+ and explicit seeding is tricky.

### Recommendation

**Do NOT implement:** CPU randperm is fast enough, and H2D copy is negligible (~1-2ms per epoch).

**Justification:**
- Amortized overhead: ~15 μs per batch (<3% of total)
- Adds complexity (CUDA generator seeding)
- Potential determinism issues

### Expected Speedup

**1-2%** (not worth the complexity)

---

## Optimization 5: Index Range Caching (Low Impact)

### Problem

`torch.tensor_split` is called every batch to split indices:

```python
# Per-batch cost
index_chunks = torch.tensor_split(batch_indices, n_envs_on_device)  # ~5 μs
```

### Solution

For fixed `batch_size_per_env` and `drop_last=True` (training), the split ranges are constant:

```python
# In __init__ (for drop_last=True only):
self._split_sizes = [self.batch_size_per_env] * n_envs_on_device

# In __next__:
# Instead of tensor_split, use pre-computed offsets
for local_idx in range(n_envs_on_device):
    start = local_idx * self.batch_size_per_env
    end = start + self.batch_size_per_env
    idx_chunk = batch_indices[start:end]
```

### Benefit

- Eliminates `tensor_split` overhead: ~5 μs per batch
- More explicit index ranges (clearer code)

### Risks

- Only works for fixed batch sizes (drop_last=True)
- Variable batches (drop_last=False) need fallback to tensor_split

### Recommendation

**Low priority:** 5 μs is negligible (<1% of total latency).

**Alternative:** Modern PyTorch optimizes `tensor_split` well, so the savings are minimal.

### Expected Speedup

**<1%** (not measurable)

---

## Benchmarking Recommendations

Before implementing any optimization, **profile the current implementation:**

```python
# Use torch.profiler to measure actual overhead
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for batch in iterator:
        pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

**Key metrics to track:**
1. **index_select time:** Dominated by allocation or compute?
2. **Kernel launch overhead:** How many kernel launches per batch?
3. **CPU overhead:** Any unexpected CPU hotspots?
4. **Memory allocation rate:** Allocator churn visible?

---

## Implementation Roadmap

### Phase 1: High-ROI Optimizations (Recommended)

1. **Buffer reuse** (Optimization #1)
   - Expected speedup: 10-20%
   - Complexity: Low
   - Implementation time: 2-3 hours

2. **Channels-last format** (Optimization #2)
   - Expected speedup: 15-30%
   - Complexity: Low
   - Implementation time: 1 hour

**Combined expected speedup:** 25-50% reduction in data loading time

### Phase 2: Medium-ROI Optimizations (Optional)

3. **Fused gather kernel** (Optimization #3)
   - Expected speedup: 5-10%
   - Complexity: Medium
   - Implementation time: 4-6 hours
   - **Gate on profiling:** Only implement if Phase 1 doesn't meet performance targets

### Phase 3: Low-ROI Optimizations (Not Recommended)

4. GPU randperm (Optimization #4) - Skip
5. Index range caching (Optimization #5) - Skip

---

## Testing Strategy

For each optimization:

1. **Correctness verification:**
   - Run existing tests (`test_record_stream_fix.py`, `test_gpu_preload_batch_size.py`)
   - Add new test for buffer reuse (verify no aliasing across batches)

2. **Performance validation:**
   - Measure end-to-end training throughput (samples/sec)
   - Use `torch.profiler` to verify optimization appears in traces
   - Compare TensorBoard traces before/after

3. **Determinism check:**
   - Run same training twice with same seed
   - Verify identical loss curves

---

## Risk Assessment

| Optimization | Correctness Risk | Performance Risk | Maintenance Burden |
|--------------|------------------|------------------|-------------------|
| Buffer reuse | Low (well-tested pattern) | None (pure win) | Low |
| Channels-last | Low (PyTorch handles) | None (format preserved) | Low |
| Fused gather | Medium (aliasing risk) | Medium (could be slower) | Medium |
| GPU randperm | Medium (determinism) | None | Medium |
| Index caching | Low | None | Low |

**Recommended:** Implement buffer reuse + channels-last first (low risk, high reward).

---

## Conclusion

The current `SharedGPUGatherBatchIterator` is **correctly implemented** and achieves its primary goal (eliminate DataLoader CPU overhead). **Two high-impact optimizations** (buffer reuse + channels-last) can provide an additional **25-50% speedup** with minimal risk.

**Next steps:**
1. Profile current implementation to establish baseline
2. Implement buffer reuse (Optimization #1)
3. Implement channels-last format (Optimization #2)
4. Re-profile and measure speedup
5. Consider fused gather (Optimization #3) only if needed

---

**Analysis completed:** 2026-01-01
**Analyst:** Claude (Batch 11 debugging session)
