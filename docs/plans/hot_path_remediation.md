# Hot Path Performance Remediation Plan

**Status:** Draft
**Created:** 2025-12-18
**Source:** PyTorch specialist analysis of `src/esper/simic/training/vectorized.py`

## Overview

Analysis of the vectorized PPO training hot path identified 5 performance issues affecting GPU utilization, `torch.compile` compatibility, and overall training throughput. This plan prioritizes fixes by impact and implementation complexity.

## Current State

- **torch.compile Readiness:** Partial — multiple graph break patterns present
- **Primary Bottlenecks:** GPU synchronization from `.item()` calls, Python overhead in batch loop
- **Affected File:** `src/esper/simic/training/vectorized.py`

---

## Issue 1: `.item()` Calls in Per-Environment Loop

**Priority:** P0 (Critical)
**Lines:** 1729-1732, 2089
**Impact:** Graph breaks + implicit GPU synchronization per environment per epoch

### Problem

```python
# Line 1729-1732: Inside per-environment loop
actions = [
    {key: actions_dict[key][i].item() for key in actions_dict}
    for i in range(len(env_states))
]
values = values_tensor.tolist()  # Line 1732 - synchronizes
```

Each `.item()` call forces a GPU→CPU sync, stalling the CUDA pipeline. With `n_envs=4` and typical epoch counts, this creates thousands of sync points per training run.

### Solution

Keep action indices as tensors until environment execution truly needs scalar values:

1. Store tensor indices directly in the rollout buffer
2. Only materialize to Python integers at the environment step boundary
3. Use batched indexing operations instead of per-element `.item()`

### Verification

- Run training with `CUDA_LAUNCH_BLOCKING=1` and profile sync points before/after
- Measure epoch time improvement (expect 5-15% speedup)

---

## Issue 2: Dictionary Comprehensions with Dynamic Keys

**Priority:** P1 (High)
**Lines:** 1646-1651, 1728-1731
**Impact:** Graph breaks — Dynamo cannot trace `.keys()` iteration

### Problem

```python
# Line 1646-1651: Creating masks_batch
masks_batch = {
    key: torch.stack([m[key] for m in all_masks]).to(device)
    for key in all_masks[0].keys()  # Dynamic iteration!
}
```

The `all_masks[0].keys()` creates data-dependent iteration that Dynamo cannot trace statically.

### Solution

Replace dynamic key iteration with the static `HEAD_NAMES` constant:

```python
from esper.leyline import HEAD_NAMES

masks_batch = {
    key: torch.stack([m[key] for m in all_masks]).to(device)
    for key in HEAD_NAMES  # Static tuple, Dynamo-friendly
}
```

### Verification

- Attempt `torch.compile(fullgraph=True)` on policy forward pass
- Confirm no graph breaks from dict iteration in Dynamo logs

---

## Issue 3: Autocast Context Manager Recreated Per-Batch

**Priority:** P1 (High)
**Lines:** 1028-1032
**Impact:** Python overhead in hot path, potential graph breaks

### Problem

```python
# Inside process_train_batch() - called every batch
autocast_ctx = (
    torch_amp.autocast(device_type="cuda", dtype=torch.float16)
    if use_amp and env_dev.startswith("cuda")
    else nullcontext()
)
with autocast_ctx:
    outputs = model(inputs)
```

Context manager object created anew for every batch. String comparison `env_dev.startswith("cuda")` also happens every batch.

### Solution

Move autocast decision outside the batch loop:

```python
# In create_env_state or at episode start
env_state.autocast_enabled = use_amp and env_device.startswith("cuda")

# Option A: Use torch.set_autocast_enabled() around entire training loop
# Option B: Create autocast context once per episode, reuse
# Option C: Apply @torch.amp.autocast decorator to model forward method
```

### Verification

- Profile Python overhead in batch loop before/after
- Confirm autocast dtype propagation unchanged

---

## Issue 4: Redundant Stream Synchronization

**Priority:** P2 (Medium)
**Lines:** 1217-1222, 1244-1245
**Impact:** Over-serialized GPU execution, reduced compute/transfer overlap

### Problem

```python
# Lines 1217-1222: Before training loop
for i, env_state in enumerate(env_states):
    if env_state.stream:
        env_state.stream.wait_stream(torch.cuda.default_stream(...))

# Lines 1244-1245: ALSO called per-batch inside the loop
if env_state.stream:
    env_state.stream.wait_stream(torch.cuda.default_stream(...))
```

`wait_stream()` called both before batch loop AND inside each iteration. For `gpu_preload=True`, the inner sync is completely unnecessary since data is already on GPU.

### Solution

Make inner `wait_stream()` conditional on actual data transfer:

```python
# Only sync if SharedBatchIterator performed async transfer this batch
if env_state.stream and not gpu_preload:
    env_state.stream.wait_stream(torch.cuda.default_stream(...))
```

### Verification

- Profile with Nsight Systems to visualize stream overlap
- Confirm no race conditions with `CUDA_LAUNCH_BLOCKING=1`

---

## Issue 5: Per-Slot Python Loops with Data-Dependent Conditionals

**Priority:** P2 (Medium)
**Lines:** 988-1026, 1044-1070
**Impact:** Non-vectorized operations, graph breaks from conditionals

### Problem

```python
# Lines 988-997: Per-slot stage advancement
for slot_id in slots:
    if not model.has_active_seed_in_slot(slot_id):
        continue
    slot_state = model.seed_slots[slot_id].state
    if slot_state and slot_state.stage == SeedStage.GERMINATED:
        gate_result = model.seed_slots[slot_id].advance_stage(SeedStage.TRAINING)
```

Multiple Python loops with `has_active_seed_in_slot()` called repeatedly. Gradient telemetry collection iterates twice over slots.

### Solution

1. Cache `has_active_seed_in_slot()` results at batch start
2. Precompute which slots need gradient telemetry once per epoch
3. Isolate telemetry collection behind `@torch.compiler.disable`:

```python
@torch.compiler.disable
def collect_all_gradient_telemetry(model, slots, env_dev, use_telemetry):
    """Isolated from torch.compile to prevent graph breaks."""
    if not use_telemetry:
        return None
    ...
```

### Verification

- Confirm telemetry still collected correctly
- Profile to ensure no regression from function call overhead

---

## Additional Observations (Lower Priority)

### Issue 6: AMP Autocast Scope Too Narrow

**Priority:** P3 (Low)
**Lines:** 1033-1035
**Impact:** Suboptimal kernel selection for backward pass

The autocast context in `process_train_batch` wraps only the forward pass. Modern PyTorch AMP documentation suggests wrapping both forward AND backward under autocast for optimal kernel selection. The backward pass currently runs outside autocast, relying solely on GradScaler.

**Solution:** Extend autocast scope to include backward pass, or use `@torch.amp.autocast` decorator on model forward.

---

### Issue 7: Buffer Allocation Strategy

**Priority:** P3 (Low)
**Lines:** rollout_buffer.py:340-387
**Impact:** ~20 `.to(device, non_blocking=nb)` calls per PPO update

The `get_batched_sequences()` method performs many device transfers per update. While `non_blocking=True` enables overlap, this could be optimized by keeping the buffer on GPU from the start when all environments share the same device.

**Solution:** For single-device training, allocate buffer tensors directly on GPU. Add `buffer_device` parameter to `RolloutBuffer.__init__()`.

---

### Issue 8: CUDATimer Synchronization

**Priority:** P3 (Low)
**Lines:** 1177, 2204
**Impact:** Blocking synchronize() call at end of each step

The `CUDATimer.stop()` method calls `synchronize()` which blocks CPU until all GPU work completes. While necessary for accurate timing, this adds latency in production.

**Solution:** Make timing optional via config flag, or use sampling-based timing (every Nth step).

---

## Implementation Order

| Phase | Issues | Effort | Expected Impact |
|-------|--------|--------|-----------------|
| **Phase 1** | #2 (dict keys) | 30 min | Enables torch.compile analysis |
| **Phase 2** | #3 (autocast) | 1 hour | Reduces Python overhead |
| **Phase 3** | #1 (.item() calls) | 2-3 hours | Major sync reduction |
| **Phase 4** | #4, #5 (streams, loops) | 2-3 hours | Fine-tuning |

## Success Criteria

1. **Epoch time reduction:** 10-20% faster training epochs
2. **torch.compile compatibility:** Policy forward pass compiles with `fullgraph=True`
3. **No correctness regression:** All existing tests pass, training curves unchanged

## Related Files

- `src/esper/simic/training/vectorized.py` — Main hot path
- `src/esper/simic/agent/ppo.py` — Policy forward pass
- `src/esper/simic/training/rollout_buffer.py` — Buffer storage
- `src/esper/leyline/__init__.py` — `HEAD_NAMES` constant

## References

- PyTorch 2.x Compile Documentation: https://pytorch.org/docs/stable/torch.compiler.html
- Dynamo Graph Break Debugging: https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html
