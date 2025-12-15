# BUG Template

- **Title:** GPU dataset cache can serve stale/mismatched data across runs/devices
- **Context:** Utils / `_GPU_DATASET_CACHE` in `src/esper/utils/data.py::load_cifar10_gpu`
- **Impact:** P1 – cache is keyed only by device string; changing transforms, `data_root`, or dataset contents reuses an old GPU tensor snapshot, producing incorrect training data without warning. Also grows unbounded per device within a long-lived process.
- **Environment:** Main branch; any process that calls `load_cifar10_gpu` multiple times with different roots/transforms or runs many devices in one interpreter.
- **Reproduction Steps:**
  1. Call `load_cifar10_gpu(device="cuda:0")` with default transforms; note the returned tensor values.
  2. Modify transforms (e.g., no normalization) or point to a different `data_root`, then call again in the same process.
  3. Observe the cache key `cifar10_cuda:0` hits and returns the original tensors (stale transforms/root), ignoring the new arguments.
- **Expected Behavior:** Cache should be invalidated when inputs change (transforms/root), or at minimum expose a way to clear/refresh; OOM risk should be bounded.
- **Observed Behavior:** Cache is global keyed only by device; transforms and data_root are ignored, and entries never evict.
- **Hypotheses:** Minimal cache key and lack of eviction were chosen for fast reuse, but they break correctness when transforms/roots vary and can accumulate per-device tensors in long-lived services.
- **Fix Plan:** Expand cache key to include `data_root` and a transform/version hash; add an explicit `clear_gpu_cache()` or `cache_max_entries` guard; document behavior.
- **Validation Plan:** Unit test that two calls with different transforms or roots do not reuse stale tensors; ensure cache eviction keeps memory bounded in a multi-device loop.
- **Status:** Open
- **Links:** `src/esper/utils/data.py` (`_GPU_DATASET_CACHE`, `load_cifar10_gpu`)
