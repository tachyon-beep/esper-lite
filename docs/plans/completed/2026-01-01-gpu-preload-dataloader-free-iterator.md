# CIFAR-10 GPU-Preload: DataLoader-Free Shared Iterator (Experimental)

**Goal:** Remove `torch.utils.data.DataLoader` overhead from the `--gpu-preload` path by iterating the GPU-cached CIFAR tensors directly, using batched gathers instead of per-sample collation.

**Motivation (from torch.profiler):**
- With `--gpu-preload`, the dominant CPU hotspot is `enumerate(DataLoader)#_SingleProcessDataLoaderIter...` plus `aten::select/aten::stack/aten::as_strided` from per-sample collation.
- This stalls the PPO inner-epoch loop and underutilizes both GPUs.
- The very large `ProfilerStep*` CUDA totals in summaries are largely step-annotation aggregation noise across streams; focus on the real op hotspots above.

**Non-goals:**
- No behavior changes to PPO, rewards, Sanctum, or telemetry semantics.
- No changes to the non-preload path (`SharedBatchIterator`).
- No augmentation changes (keep current CIFAR preprocessing).
- Keep existing `SharedGPUBatchIterator` as the default until this is validated.

---

## Design Summary

Add a new DataLoader-free iterator (e.g. `SharedGPUGatherBatchIterator`) in `src/esper/utils/data.py`, enabled only via an experimental CLI flag:

- New CLI flag: `--experimental-gpu-preload-gather`
- Behavior:
  - Default (`--gpu-preload` only): keep current `SharedGPUBatchIterator` (DataLoader-based)
  - With `--gpu-preload --experimental-gpu-preload-gather`: use the new gather iterator

1. **Keep the existing GPU cache** (`_GPU_DATASET_CACHE`) and per-device dataset replication (one full CIFAR copy per GPU).
2. For each device:
   - Create a **shuffled index order** once per `__iter__()` (like an epoch), using a CPU-side `randperm` and copy indices to the GPU.
   - Maintain a cursor into that index tensor.
3. For each `__next__()`:
   - Slice the next `total_batch = batch_size_per_env * n_envs_on_device` indices.
   - Split indices per env and gather `inputs/targets` via GPU `index_select` (one allocation per env, no view aliasing).
4. Return `list[(inputs, targets)]` sized `n_envs`, already on each env’s device (so `.to(device)` is a no-op and the training loop’s existing stream sync remains correct).

**Why CPU-side `randperm` + per-device generator?**
- Current `SharedGPUBatchIterator` uses DataLoader’s CPU-side sampling/collation even though tensors are GPU-resident.
- We want “different shuffle per device” without spooky determinism:
  - Do **not** use `hash(device)` (unstable across processes unless `PYTHONHASHSEED` is pinned).
  - Do **not** rely on sequential RNG consumption order (sensitive to device iteration order).
- Approach: build a **per-device CPU generator** seeded from:
  - `base_seed` (training seed) + `PRIME * device_index`, where `cuda:0 -> 0`, `cuda:1 -> 1`, etc.
  - This is stable, order-independent, and reproducible.

---

## Task 0: Baseline Repro (before changes)

Run a short capture and save the output for comparison:

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 cuda:1 \
  --config-json config.json --gpu-preload \
  --rounds 1 --envs 16 --episode-length 12 --no-tui \
  --telemetry-level normal --telemetry-dir ./telemetry \
  --torch-profiler --torch-profiler-dir ./profiler_traces/baseline_gpu_preload \
  --torch-profiler-wait 5 --torch-profiler-warmup 5 --torch-profiler-active 1 \
  --torch-profiler-summary
```

**Baseline expectation:** CPU top ops dominated by `enumerate(DataLoader)` and `aten::select/aten::stack`.

---

## Task 1: Implement DataLoader-Free Iterator (New Class)

**Files:**
- Modify: `src/esper/utils/data.py`
- Validate: `src/esper/simic/training/vectorized.py` (no functional changes expected)
- Modify: `src/esper/scripts/train.py` (add experimental CLI flag)

**Steps:**
1. Add a new iterator class (keep `SharedGPUBatchIterator` unchanged) that uses a per-device state object:
   - Dataset tensors: `(train_x, train_y)` or `(test_x, test_y)` on that device
   - `perm_cuda: torch.Tensor` (int64 indices on device)
   - `cursor: int`
   - `drop_last: bool`
   - `cpu_gen: torch.Generator` seeded deterministically (see above)
2. Implement `__len__` without `DataLoader`:
   - `drop_last=True`: `len = dataset_len // total_batch`
   - `drop_last=False`: `len = ceil(dataset_len / total_batch)`
   - Keep `self._len = min(per_device_len)` to preserve multi-device stopping semantics.
3. Implement `__iter__`:
   - For `shuffle=True`, generate `perm_cpu = torch.randperm(dataset_len, generator=cpu_gen)` (CPU), then `perm_cuda = perm_cpu.to(device)`.
   - For `shuffle=False`, generate `perm_cuda = torch.arange(dataset_len, device=device)`.
   - Reset cursor to `0`.
4. Implement `__next__`:
   - For each device, slice `perm_cuda[cursor:cursor+total_batch]` (or remaining if `drop_last=False`).
   - Split indices per env on that device.
   - Produce **per-env** tensors via `index_select` so the outputs are independent allocations (no view aliasing).
   - Return global `result` list aligned to original env indices.

**Correctness constraints:**
- Returned tensors must be device-resident and match the environment’s device exactly.
- Avoid view aliasing between environments (this was the root cause of past NLL assertion failures).
- Preserve `shuffle` and `drop_last` behavior.
- Cache keying must reuse `_cifar10_cache_key()` / `_GPU_DATASET_CACHE` helpers (no regressions).
- (Experimental constraint) Require `n_envs` evenly divisible across devices (for now) to avoid tricky partial-batch/env-alignment corner cases.

---

## Task 2: Update/Extend Tests

**Files:**
- Update: `tests/simic/test_record_stream_fix.py`
- Keep passing: `tests/simic/test_gpu_preload_batch_size.py`

**Steps:**
1. Keep `test_shared_gpu_iterator_returns_device_resident_tensors()` but extend it to assert:
   - `inputs.is_cuda` and `targets.is_cuda`
   - `.to(device, non_blocking=True)` is identity (existing)
2. Add a new test for “no aliasing across envs” (CUDA only):
   - Fetch `env_batches = next(iter(iterator))`
   - Assert `inputs.untyped_storage().data_ptr()` differs between envs (and same for targets)
   - (Rationale) `data_ptr()` can differ for views with different offsets; `untyped_storage()` detects shared backing allocations.

---

## Task 3: Profiling + Acceptance Criteria

Re-run the same profiler capture as Task 0 with the updated iterator:

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 cuda:1 \
  --config-json config.json --gpu-preload \
  --experimental-gpu-preload-gather \
  --rounds 1 --envs 16 --episode-length 12 --no-tui \
  --telemetry-level normal --telemetry-dir ./telemetry \
  --torch-profiler --torch-profiler-dir ./profiler_traces/after_gpu_preload_gather \
  --torch-profiler-wait 5 --torch-profiler-warmup 5 --torch-profiler-active 1 \
  --torch-profiler-summary
```

**Acceptance criteria:**
- `enumerate(DataLoader)...` disappears from CPU top ops.
- `aten::select/aten::stack` CPU time drops sharply; replaced primarily by `aten::index_select` (or equivalent gather op).
- End-to-end inner-epoch wall time decreases for the same `episode-length` (sanity check: compare timestamps or throughput telemetry).

---

## Risks + Mitigations

- **Identical shuffle across GPUs (sample duplication):** Use a per-device CPU `torch.Generator` seeded from `base_seed + PRIME * device_index` (no `hash(device)`, no iteration-order dependence).
- **Stream safety / premature reuse:** Return per-env allocations (no shared base storage), and keep the training loop’s existing `default_stream` → `env_state.stream.wait_stream()` sync intact.
- **Memory growth:** Avoid storing per-step batches; return tensors and let the training loop own their lifetime.

---

## Follow-ups (Optional, Separate Plan)

1. **Buffer reuse:** Reuse pre-allocated per-env buffers via `index_select(..., out=...)` to reduce allocation churn.
2. **Memory format:** Convert cached CIFAR tensors to `channels_last` once at cache load time to improve convolution throughput (requires profiling confirmation).
