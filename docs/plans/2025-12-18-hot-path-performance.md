# Hot Path Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the vectorized PPO training hot path for 10-20% faster epochs and torch.compile compatibility.

**Architecture:** Replace dynamic Python patterns with static alternatives. Eliminate GPU synchronization from `.item()` calls in the action loop. Move autocast decisions outside the batch loop. Guard telemetry collection with `@torch.compiler.disable`.

**Tech Stack:** PyTorch 2.x, torch.compile/Dynamo, CUDA streams, AMP

---

## Task 1: Replace Dynamic Dict Key Iteration with Static HEAD_NAMES

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1648-1651`
- Test: `tests/simic/test_vectorized.py`

**Step 1: Verify HEAD_NAMES is exported from leyline**

Run: `python -c "from esper.leyline import HEAD_NAMES; print(HEAD_NAMES)"`
Expected: `('slot', 'blueprint', 'blend', 'op')`

**Step 2: Find all `.keys()` patterns in vectorized.py**

Run: `grep -n "\.keys()" src/esper/simic/training/vectorized.py`
Expected: Line 1650 and potentially others

**Step 3: Update masks_batch creation to use HEAD_NAMES**

Replace lines 1648-1651:

```python
# OLD (line 1648-1651):
masks_batch = {
    key: torch.stack([m[key] for m in all_masks]).to(device)
    for key in all_masks[0].keys()
}
```

With:

```python
# NEW - use static HEAD_NAMES for torch.compile compatibility
from esper.leyline import HEAD_NAMES

masks_batch = {
    key: torch.stack([m[key] for m in all_masks]).to(device)
    for key in HEAD_NAMES
}
```

**Note:** The import should go at the top of the file with other leyline imports (around line 30-50).

**Step 4: Run existing tests to verify no regression**

Run: `PYTHONPATH=src pytest tests/simic/test_vectorized.py -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): use static HEAD_NAMES for torch.compile compatibility

Replace dynamic .keys() iteration with static HEAD_NAMES tuple.
Enables Dynamo to trace the dict comprehension without graph breaks."
```

---

## Task 2: Update Action Dict Iteration to Use HEAD_NAMES

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1728-1731`
- Test: `tests/simic/test_vectorized.py`

**Step 1: Locate the action conversion pattern**

Run: `grep -n "for key in actions_dict" src/esper/simic/training/vectorized.py`
Expected: Line 1729

**Step 2: Update action dict iteration to use HEAD_NAMES**

Replace lines 1728-1731:

```python
# OLD (line 1728-1731):
actions = [
    {key: actions_dict[key][i].item() for key in actions_dict}
    for i in range(len(env_states))
]
```

With:

```python
# NEW - use static HEAD_NAMES (still has .item() - addressed in Task 5)
actions = [
    {key: actions_dict[key][i].item() for key in HEAD_NAMES}
    for i in range(len(env_states))
]
```

**Step 3: Run tests**

Run: `PYTHONPATH=src pytest tests/simic/test_vectorized.py -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): use static HEAD_NAMES in action dict iteration

Second instance of dynamic key iteration replaced with static tuple."
```

---

## Task 3: Cache Autocast Decision in EnvState

**Files:**
- Modify: `src/esper/simic/training/vectorized.py` (EnvState creation ~line 860, process_train_batch ~line 1028)
- Test: `tests/simic/test_vectorized.py`

**Step 1: Locate EnvState creation**

Run: `grep -n "EnvState(" src/esper/simic/training/vectorized.py | head -5`
Expected: Line around 860

**Step 2: Add autocast_enabled field to env_state initialization**

Find the EnvState creation (around line 860-880) and add after scaler assignment:

```python
# After: env_scaler = torch.amp.GradScaler(...)
# Add:
autocast_enabled = amp_enabled and env_device_obj.type == "cuda"
```

Then include in the EnvState creation:

```python
env_state = EnvState(
    # ... existing fields ...
    autocast_enabled=autocast_enabled,  # Add this field
)
```

**Step 3: Update EnvState dataclass to include autocast_enabled**

Find the EnvState dataclass definition and add:

```python
@dataclass
class EnvState:
    # ... existing fields ...
    autocast_enabled: bool = False  # Pre-computed for hot path
```

**Step 4: Update process_train_batch to use cached autocast decision**

Replace lines 1028-1032:

```python
# OLD (line 1028-1032):
autocast_ctx = (
    torch_amp.autocast(device_type="cuda", dtype=torch.float16)
    if use_amp and env_dev.startswith("cuda")
    else nullcontext()
)
```

With:

```python
# NEW - use pre-computed autocast decision from env_state
autocast_ctx = (
    torch_amp.autocast(device_type="cuda", dtype=torch.float16)
    if env_state.autocast_enabled
    else nullcontext()
)
```

**Note:** The function signature may need `env_state` to access `.autocast_enabled`. Check if it's already passed.

**Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/simic/test_vectorized.py -v --tb=short`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): cache autocast decision in EnvState

Move device type check and amp flag evaluation outside the batch loop.
Reduces Python overhead per batch."
```

---

## Task 4: Skip Inner wait_stream When gpu_preload=True

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1244-1245`
- Test: `tests/simic/test_vectorized.py`

**Step 1: Locate the inner wait_stream call**

Run: `grep -n "wait_stream" src/esper/simic/training/vectorized.py`
Expected: Lines 1222 and 1245

**Step 2: Make inner wait_stream conditional on data transfer**

Replace lines 1244-1245:

```python
# OLD (line 1244-1245):
if env_state.stream:
    env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))
```

With:

```python
# NEW - only sync if SharedBatchIterator did async transfer (not gpu_preload)
# When gpu_preload=True, data is already on GPU, no transfer to wait for
if env_state.stream and shared_train_iter is not None:
    env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))
```

**Note:** `shared_train_iter` is `None` when `gpu_preload=True`, so this condition correctly skips the sync.

**Step 3: Run tests**

Run: `PYTHONPATH=src pytest tests/simic/test_vectorized.py -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): skip redundant wait_stream when gpu_preload=True

Inner wait_stream only needed when SharedBatchIterator performs async
transfers on default stream. When gpu_preload=True, data is already on
GPU, so no sync needed."
```

---

## Task 5: Isolate Gradient Telemetry with @torch.compiler.disable

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1044-1070`
- Test: `tests/simic/test_vectorized.py`

**Step 1: Create isolated telemetry collection function**

Add new function before `process_train_batch` (around line 980):

```python
@torch.compiler.disable
def _collect_gradient_telemetry_for_batch(
    model: "HostWithSeeds",
    slots: list[str],
    env_dev: str,
) -> dict[str, dict[str, Any]] | None:
    """Collect gradient telemetry for all active slots.

    Isolated from torch.compile to prevent graph breaks from
    data-dependent slot iteration and conditional logic.
    """
    from esper.leyline import SeedStage
    from esper.simic.telemetry.gradient_collector import (
        collect_host_gradients_async,
        collect_seed_gradients_only_async,
    )

    slots_needing_grad_telemetry = []
    for slot_id in slots:
        if not model.has_active_seed_in_slot(slot_id):
            continue
        seed_state = model.seed_slots[slot_id].state
        if seed_state and seed_state.stage in (SeedStage.TRAINING, SeedStage.BLENDING):
            slots_needing_grad_telemetry.append(slot_id)

    if not slots_needing_grad_telemetry:
        return None

    # Compute host gradient stats ONCE (expensive), then reuse for each seed
    host_stats = collect_host_gradients_async(
        model.get_host_parameters(),
        device=env_dev,
    )

    grad_stats_by_slot = {}
    for slot_id in slots_needing_grad_telemetry:
        seed_stats = collect_seed_gradients_only_async(
            model.get_seed_parameters(slot_id),
            device=env_dev,
        )
        grad_stats_by_slot[slot_id] = {
            **host_stats,
            **seed_stats,
        }

    return grad_stats_by_slot
```

**Step 2: Update process_train_batch to use the isolated function**

Replace lines 1044-1070 (the telemetry collection block):

```python
# OLD: inline telemetry collection with loops
if use_telemetry:
    slots_needing_grad_telemetry = []
    for slot_id in slots:
        # ... loop body ...
```

With:

```python
# NEW: call isolated function
grad_stats_by_slot = None
if use_telemetry:
    grad_stats_by_slot = _collect_gradient_telemetry_for_batch(model, slots, env_dev)
```

**Step 3: Run tests**

Run: `PYTHONPATH=src pytest tests/simic/test_vectorized.py -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): isolate gradient telemetry from torch.compile

Extract telemetry collection to @torch.compiler.disable function.
Prevents graph breaks from data-dependent slot iteration."
```

---

## Task 6: Cache Active Slot Results Per Batch

**Files:**
- Modify: `src/esper/simic/training/vectorized.py` (process_train_batch)
- Test: `tests/simic/test_vectorized.py`

**Step 1: Locate repeated has_active_seed_in_slot calls**

Run: `grep -n "has_active_seed_in_slot" src/esper/simic/training/vectorized.py | wc -l`
Expected: Multiple calls

**Step 2: Add slot activity cache at batch start**

At the start of `process_train_batch` (after getting model), add:

```python
# Cache slot activity to avoid repeated dict lookups in hot path
active_slots = {
    slot_id: model.has_active_seed_in_slot(slot_id)
    for slot_id in slots
}
slots_to_step = [slot_id for slot_id, active in active_slots.items() if active]
```

**Step 3: Update slot loops to use cached results**

Replace patterns like:

```python
for slot_id in slots:
    if not model.has_active_seed_in_slot(slot_id):
        continue
```

With:

```python
for slot_id in slots_to_step:
    # Already filtered to active slots
```

**Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/simic/test_vectorized.py -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): cache active slot results at batch start

Avoid repeated has_active_seed_in_slot() calls in hot path loops."
```

---

## Task 7: Final Integration Test

**Files:**
- Test: `tests/integration/test_vectorized_factored.py`

**Step 1: Run full integration test suite**

Run: `PYTHONPATH=src pytest tests/integration/test_vectorized_factored.py -v --tb=short`
Expected: All tests PASS

**Step 2: Run smoke test with actual training**

Run: `PYTHONPATH=src timeout 60 python -m esper.scripts.train ppo --preset cifar10 --no-tui 2>&1 | head -50`
Expected: Training starts without errors, shows epoch progress

**Step 3: Commit final verification**

```bash
git add -A
git commit -m "test(simic): verify hot path optimizations

All vectorized and integration tests pass after performance optimizations."
```

---

## Success Criteria

1. **All tests pass:** `pytest tests/simic/ tests/integration/test_vectorized_factored.py`
2. **No dynamic key iteration:** `grep "\.keys()" src/esper/simic/training/vectorized.py` returns no hot path hits
3. **Autocast cached:** `grep "autocast_enabled" src/esper/simic/training/vectorized.py` shows field usage
4. **Telemetry isolated:** `grep "@torch.compiler.disable" src/esper/simic/training/vectorized.py` shows decorator

---

## Deferred (P3 - Future Work)

The following optimizations are documented but deferred to a future iteration:

1. **Issue 6:** AMP autocast scope (wrap backward pass)
2. **Issue 7:** Buffer allocation on GPU
3. **Issue 8:** Optional CUDATimer sampling

See `docs/plans/hot_path_remediation.md` for details.
