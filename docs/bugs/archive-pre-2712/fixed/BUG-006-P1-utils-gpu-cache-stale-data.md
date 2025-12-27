# BUG-006: GPU dataset cache can serve stale/mismatched data across roots

- **Title:** GPU CIFAR-10 cache keyed only by device, ignoring `data_root`
- **Context:** `_GPU_DATASET_CACHE` backing both `SharedGPUBatchIterator` and `load_cifar10_gpu` (`src/esper/utils/data.py`)
- **Impact:** P1 – in long-lived processes, reusing `device="cuda:0"` with a different `data_root` silently returns tensors from the first root (wrong training/eval data). Cache entries also pinned GPU memory with no invalidation API.
- **Environment:** Main branch; any run that uses GPU-preload CIFAR-10 in-process across different roots.
- **Status:** FIXED (2025-12-20)

## Root Cause

Cache key was `cifar10_{device}`, so `data_root` (and any future cache-relevant config) was ignored.

## Fix

- Key cache entries by `(dataset, device, normalized_data_root, version)` so changing roots cannot hit stale tensors.
- Add `clear_gpu_dataset_cache(...)` for explicit invalidation/freeing memory and a `refresh` flag to reload even if cached.

## Validation

- Unit test: `tests/utils/test_data.py::TestData::test_load_cifar10_gpu_cache_key_includes_data_root`
- Commands run: `uv run pytest -q tests/utils/test_data.py`

## Links

- Fix: `src/esper/utils/data.py` (`_cifar10_cache_key`, `clear_gpu_dataset_cache`, `load_cifar10_gpu(refresh=...)`)
