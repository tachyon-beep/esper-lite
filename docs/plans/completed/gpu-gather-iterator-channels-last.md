# GPU Gather Iterator: Channels-Last Memory Format Implementation Plan

**Date:** 2026-01-01
**Status:** ✅ IMPLEMENTED (2026-01-01)
**Prerequisites:** None (independent of buffer reuse optimization)
**Expected Speedup:** 15-30% on convolution-heavy models
**Estimated Time:** 1 hour

---

## Implementation Notes

**Implemented with corrections from code review:**

1. **One-shot transfer**: Used `.to(device, memory_format=torch.channels_last)` to avoid 2x VRAM spike
2. **Advanced indexing**: Switched from `index_select` to `dataset_x[idx_chunk]` to preserve format (original plan incorrectly claimed `index_select` preserves format)
3. **Host model conversion**: Added `model.to(memory_format=torch.channels_last)` in `create_env_state` for the CNN host
4. **LSTM policy NOT converted**: Channels-last only applies to CNNs with conv layers, not LSTM networks
5. **Verification tests**: Added `tests/simic/test_data_opt.py` with model format tests

**Buffer reuse NOT implemented** - requires ring buffers + CUDA event fencing due to pipelined batch consumption. See `docs/plans/abandoned/gpu-gather-iterator-buffer-reuse.md`.

---

## Overview

Convert GPU-cached CIFAR-10 tensors to channels-last memory format (`torch.channels_last`) once at load time. This improves convolution throughput on modern GPUs without any runtime overhead.

**Current behavior:** Tensors in NCHW (channels-first) format
**Target behavior:** Tensors in NHWC (channels-last) format

---

## Background

### Memory Format Comparison

**Channels-first (NCHW):** Default PyTorch format
```
Tensor shape: (N, C, H, W) = (batch, 3, 32, 32)
Memory layout: [[[R00, R01, ...], [R10, ...]], [[G00, ...], ...], [[B00, ...], ...]]
               ^--- All red pixels, then all green, then all blue
```

**Channels-last (NHWC):** Optimized for modern GPUs
```
Tensor shape: (N, C, H, W) = (batch, 3, 32, 32)  # Shape unchanged!
Memory layout: [(R00, G00, B00), (R01, G01, B01), ...]
               ^--- RGB values contiguous per pixel
```

### Why Channels-Last is Faster

1. **Tensor Core utilization:** NVIDIA Tensor Cores (Volta+) prefer NHWC layout
2. **cuDNN optimization:** cuDNN conv kernels are optimized for channels-last
3. **Memory coalescing:** Better cache utilization for conv kernels
4. **PyTorch 2.0+:** Format is preserved through most operations automatically

**Expected speedup for convolutions:** 15-30% (varies by GPU architecture)

---

## Task 1: Modify Cache Loading Function

**File:** `src/esper/utils/data.py`

### 1.1 Find `_ensure_cifar10_cached` function

Locate the function that loads and caches CIFAR-10 tensors (~line 280-350).

### 1.2 Add channels-last conversion after GPU transfer

Current code (approximate):
```python
def _ensure_cifar10_cached(device: str, data_root: str, refresh: bool = False) -> None:
    cache_key = _cifar10_cache_key(device, data_root)
    if not refresh and cache_key in _GPU_DATASET_CACHE:
        return

    # Load dataset
    train_x, train_y = load_cifar10_tensors(data_root, train=True)
    test_x, test_y = load_cifar10_tensors(data_root, train=False)

    # Move to GPU
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    _GPU_DATASET_CACHE[cache_key] = (train_x, train_y, test_x, test_y)
```

New code (add channels-last conversion):
```python
def _ensure_cifar10_cached(device: str, data_root: str, refresh: bool = False) -> None:
    cache_key = _cifar10_cache_key(device, data_root)
    if not refresh and cache_key in _GPU_DATASET_CACHE:
        return

    # Load dataset
    train_x, train_y = load_cifar10_tensors(data_root, train=True)
    test_x, test_y = load_cifar10_tensors(data_root, train=False)

    # Move to GPU
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    # Convert to channels-last format for better conv performance
    # This is a one-time cost at cache load, amortized across all batches
    if train_x.dim() == 4:  # Only for 4D tensors (N, C, H, W)
        train_x = train_x.to(memory_format=torch.channels_last)
        test_x = test_x.to(memory_format=torch.channels_last)

    _GPU_DATASET_CACHE[cache_key] = (train_x, train_y, test_x, test_y)
```

### 1.3 Why only image tensors

- `train_x`, `test_x`: 4D tensors (N, C, H, W) - convert to channels-last
- `train_y`, `test_y`: 1D tensors (N,) - no memory format (labels don't have channels)

---

## Task 2: Verify Format Preservation Through Iterator

**File:** `src/esper/utils/data.py`

### 2.1 ~~Check `index_select` preserves format~~ CORRECTED

> **⚠️ ORIGINAL PLAN WAS INCORRECT:** `torch.index_select` does NOT preserve channels-last format!

```python
# Source is channels-last
dataset_x.is_contiguous(memory_format=torch.channels_last)  # True

# index_select DOES NOT preserve format - returns NCHW-contiguous
output = torch.index_select(dataset_x, 0, indices)
output.is_contiguous(memory_format=torch.channels_last)  # FALSE!
output.is_contiguous()  # True (NCHW)
```

**ACTUAL FIX:** Use advanced indexing instead, which DOES preserve format:

```python
# Advanced indexing preserves memory format
output = dataset_x[indices]
output.is_contiguous(memory_format=torch.channels_last)  # True
```

**Changes required in `__next__`:** Replace `torch.index_select(state.dataset_x, 0, idx_chunk)` with `state.dataset_x[idx_chunk]`.

### 2.2 Verify with buffer reuse (if combined with Optimization 1)

If implementing both optimizations, ensure buffers are created with channels-last format:

```python
# In __init__ when allocating buffers:
input_buffers = [
    torch.empty(
        input_shape,
        dtype=dataset_x.dtype,
        device=device,
        memory_format=torch.channels_last  # ADD THIS
    )
    for _ in range(n_envs_on_device)
]
```

This ensures `index_select(..., out=buffer)` writes to channels-last memory.

---

## Task 3: Verify Model Compatibility

**File:** `src/esper/kasmina/blueprints/*.py` (model definitions)

### 3.1 Check model input handling

Modern PyTorch models (ResNet, etc.) automatically handle channels-last inputs:

```python
# Model created normally (channels-first)
model = resnet18()

# Input in channels-last format
x = batch.to(memory_format=torch.channels_last)

# Forward pass works correctly
# Internal conv layers detect channels-last and use optimized kernels
output = model(x)
```

**No changes needed to model definitions** - PyTorch handles format internally.

### 3.2 Optional: Convert model to channels-last

For maximum performance, convert model weights to channels-last:

```python
# In training setup (one-time):
model = model.to(memory_format=torch.channels_last)
```

This ensures both input data AND model weights are channels-last, avoiding any format conversion during forward pass.

**Location:** `src/esper/simic/training/vectorized.py`, after model creation (~line 1600)

---

## Task 4: Add Format Verification Test

**File:** `tests/simic/test_gpu_preload_batch_size.py` or new file

### 4.1 Test that cached tensors are channels-last

```python
def test_cached_tensors_channels_last():
    """Verify GPU-cached CIFAR tensors are in channels-last format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Ensure cache is populated
    _ensure_cifar10_cached("cuda:0", "./data")

    cache_key = _cifar10_cache_key("cuda:0", "./data")
    train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

    # Image tensors should be channels-last
    assert train_x.is_contiguous(memory_format=torch.channels_last), \
        "train_x should be channels-last"
    assert test_x.is_contiguous(memory_format=torch.channels_last), \
        "test_x should be channels-last"

    # Labels are 1D, no memory format concept
    assert train_y.dim() == 1
    assert test_y.dim() == 1
```

### 4.2 Test that iterator outputs preserve format

```python
def test_iterator_preserves_channels_last():
    """Verify iterator outputs maintain channels-last format."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=32,
        n_envs=4,
        env_devices=["cuda:0"] * 4,
        shuffle=True,
        seed=42,
    )

    batch = next(iter(iterator))

    for env_idx, (inputs, targets) in enumerate(batch):
        assert inputs.is_contiguous(memory_format=torch.channels_last), \
            f"Env {env_idx} inputs should be channels-last"
```

---

## Task 5: Benchmark Performance

### 5.1 Profile convolution time before/after

Use torch.profiler to measure conv kernel time:

```bash
# Before (channels-first)
PYTHONPATH=src uv run python -c "
import torch
from esper.utils.data import _ensure_cifar10_cached, _GPU_DATASET_CACHE, _cifar10_cache_key

_ensure_cifar10_cached('cuda:0', './data')
train_x, _, _, _ = _GPU_DATASET_CACHE[_cifar10_cache_key('cuda:0', './data')]
print(f'Format: {train_x.is_contiguous(memory_format=torch.channels_last)}')

# Simple conv benchmark
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
batch = train_x[:128]

# Warmup
for _ in range(10):
    _ = conv(batch)
torch.cuda.synchronize()

# Time
import time
start = time.perf_counter()
for _ in range(100):
    _ = conv(batch)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f'100 conv passes: {elapsed*1000:.2f}ms')
"
```

### 5.2 Run before/after with full training

```bash
# Baseline (channels-first)
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 --config-json config.json --gpu-preload \
  --rounds 1 --envs 8 --episode-length 12 --no-tui \
  --torch-profiler --torch-profiler-dir ./profiler_traces/channels_first \
  --torch-profiler-summary

# After (channels-last)
# ... same command, compare CUDA time in aten::convolution
```

---

## Implementation Order

1. **Modify `_ensure_cifar10_cached`** (Task 1) - Add format conversion
2. **Verify buffer allocation** (Task 2.2) - If combined with buffer reuse
3. **Run existing tests** - Verify no regressions
4. **Add format verification tests** (Task 4) - New tests
5. **Profile before/after** (Task 5) - Measure conv speedup
6. **Optional: Convert model to channels-last** (Task 3.2)

---

## Risks and Mitigations

### Risk 1: Some operations don't preserve channels-last

**Symptoms:** Format warnings, unexpected format conversions in profiler

**Mitigation:** Modern PyTorch (2.0+) preserves format through:
- All conv layers
- All pooling layers
- BatchNorm
- ReLU and most activations
- Linear (treats as (N, C) with C=H*W*channels)

**Operations that may convert:** Some custom ops, certain reshape/view patterns

**Detection:** Add format assertion after model forward:
```python
assert output.is_contiguous(memory_format=torch.channels_last)
```

### Risk 2: Increased memory during conversion

**Concern:** `.to(memory_format=...)` may allocate temporary memory

**Reality:** For pre-allocated tensors, conversion is in-place conceptually (PyTorch may use temp buffer internally)

**Mitigation:** Conversion happens once at cache load, not per-batch. Any temp memory is freed immediately.

### Risk 3: Format mismatch with model

**Concern:** If model is channels-first but input is channels-last, PyTorch converts

**Impact:** Small overhead (~1-2%) for format conversion in first conv layer

**Optimal solution:** Convert model to channels-last (Task 3.2):
```python
model = model.to(memory_format=torch.channels_last)
```

---

## Rollback Plan

If issues are discovered:
1. Remove `.to(memory_format=torch.channels_last)` from `_ensure_cifar10_cached`
2. Clear GPU cache to force reload: `_GPU_DATASET_CACHE.clear()`
3. Restart training

Changes are minimal and easily reversible.

---

## Success Criteria

1. **All existing tests pass**
2. **New format verification tests pass**
3. **Profiler shows `aten::convolution` using channels-last kernels**
4. **Convolution time reduced by 15-30%**
5. **No unexpected format conversion warnings in logs**

---

## Compatibility Notes

- **PyTorch version:** Channels-last fully supported in PyTorch 1.5+, optimized in 2.0+
- **GPU architecture:** Best gains on Volta (V100) and later (Tensor Cores)
- **cuDNN version:** Ensure cuDNN 8.0+ for optimal channels-last support
- **torch.compile:** Channels-last is fully compatible with torch.compile
