# Simic Vectorized Training Audit Report

**File**: `/home/john/esper-lite/src/esper/simic/vectorized.py`
**Date**: 2025-12-16
**Auditor**: Claude Opus 4.5 (PyTorch Engineering Specialist)
**Lines**: ~2550

---

## Executive Summary

The `vectorized.py` module implements high-performance vectorized PPO training with CUDA streams for multi-GPU parallelism. The code demonstrates sophisticated understanding of CUDA async execution, proper stream synchronization, and memory management. However, several issues exist that could impact performance, correctness, and maintainability.

**Overall Assessment**: The module is well-architected but has notable gaps in torch.compile integration, some device placement inconsistencies, and telemetry overhead that could be reduced.

---

## 1. torch.compile Integration

### 1.1 Missing torch.compile on Hot Paths

**Severity**: MEDIUM
**Location**: Lines 1144-1338 (`process_train_batch`, `process_val_batch`)

The core training functions `process_train_batch` and `process_val_batch` are not compiled despite being called thousands of times per episode. The simic module uses torch.compile elsewhere (ppo.py line 253, training.py line 68), creating an inconsistency.

```python
# Current: No compilation
def process_train_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    ...
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict] | None]:
```

**Analysis**:
- These functions contain dynamic control flow based on `slots`, `use_telemetry`, and `use_amp` flags
- The model forward pass and backward pass would benefit from compilation
- However, the CUDA stream context managers and conditional AMP contexts create graph break risks

**Recommendation**: Extract the inner forward+backward logic into a separate function decorated with `@torch.compiler.disable` for the outer stream/context handling, allowing the core computation to be compiled.

### 1.2 Graph Break Risk: Dynamic Control Flow

**Severity**: LOW
**Location**: Lines 1196-1205, 1252-1290

```python
# Graph break risk: Python loop with dynamic slot iteration
for slot_id in slots:
    if not model.has_active_seed_in_slot(slot_id):
        continue
    slot_state = model.seed_slots[slot_id].state
    if slot_state and slot_state.stage == SeedStage.GERMINATED:
        gate_result = model.seed_slots[slot_id].advance_stage(SeedStage.TRAINING)
```

The dynamic `slots` iteration and attribute access patterns would cause graph breaks if these functions were compiled. This is acceptable given the current uncompiled state but should be addressed if compilation is added.

### 1.3 torch._foreach_norm Usage

**Severity**: INFO
**Location**: Lines 1265, 1278

```python
host_norms = torch._foreach_norm(host_grads, ord=2)
seed_norms = torch._foreach_norm(seed_grads, ord=2)
```

Good practice: Using `torch._foreach_norm` for efficient batched norm computation. This is the correct pattern for multi-tensor gradient norm calculation without synchronization.

---

## 2. Device Placement

### 2.1 Device String Comparison Anti-Pattern

**Severity**: MEDIUM
**Location**: Lines 1238, 1245, 1292

```python
if use_amp and env_dev.startswith("cuda"):
    ...
```

String-based device checking is fragile. If `env_dev` is a `torch.device` object (which it could become after refactoring), this will fail silently.

**Recommendation**: Use `torch.device(env_dev).type == "cuda"` or ensure `env_device` is always a string via type annotation enforcement.

### 2.2 Inconsistent Device Type for env_device

**Severity**: LOW
**Location**: Lines 380-396 (`ParallelEnvState`)

```python
@dataclass
class ParallelEnvState:
    ...
    env_device: str = "cuda:0"  # Device this env runs on
    stream: torch.cuda.Stream | None = None  # CUDA stream for async execution
```

The `env_device` is typed as `str` but `stream` uses `torch.cuda.Stream`. The stream creation at line 1110 properly uses `torch.device(env_device)`, but this inconsistency could lead to bugs if someone passes a `torch.device` object.

### 2.3 Cross-Device Tensor Movement

**Severity**: LOW
**Location**: Lines 1004-1006

```python
obs_normalizer.mean = torch.tensor(metadata['obs_normalizer_mean'], device=device)
obs_normalizer.var = torch.tensor(metadata['obs_normalizer_var'], device=device)
obs_normalizer._device = device
```

Creating tensors directly with `torch.tensor(..., device=device)` is correct. However, the direct assignment to `obs_normalizer._device` bypasses any device migration logic in the normalizer class.

---

## 3. Gradient Flow

### 3.1 Gradient Isolation Pattern

**Severity**: INFO
**Location**: Lines 1094-1098

```python
# Incubator mode gradient isolation: detach host input into the seed path so
# host gradients remain identical to the host-only model while the seed
# trickle-learns via STE in TRAINING.
slot.isolate_gradients = True
```

This is a core architectural decision. The gradient isolation ensures the host model's training is not affected by seeds during their training phase. This is correct for the morphogenetic learning approach.

### 3.2 AMP Scaler Sharing Across Environments

**Severity**: MEDIUM
**Location**: Lines 991-992, 1293-1296

```python
scaler = torch_amp.GradScaler(enabled=amp_enabled)
...
scaler.step(env_state.host_optimizer)
for slot_id in slots_to_step:
    scaler.step(env_state.seed_optimizers[slot_id])
scaler.update()
```

A single `GradScaler` is shared across all environments. This is problematic because:
1. Each environment runs in its own CUDA stream
2. `scaler.step()` and `scaler.update()` are called sequentially for each optimizer
3. The scaler's internal state (inf/nan detection, scale factor) is updated globally

In multi-GPU scenarios where environments run on different devices, this could cause:
- Scale factor updates from one device affecting another
- Incorrect inf/nan detection across devices

**Recommendation**: Create one `GradScaler` per device or per environment when using multi-GPU.

### 3.3 Gradient Accumulation Opportunity

**Severity**: LOW
**Location**: Lines 1232-1300

Currently, each training batch performs a full backward pass and optimizer step. For very small batch sizes, gradient accumulation could improve training stability. The current implementation does not support this.

---

## 4. Memory Management

### 4.1 Pre-allocated Accumulators

**Severity**: INFO (Positive)
**Location**: Lines 436-462

```python
def init_accumulators(self, slots: list[str]) -> None:
    """Initialize pre-allocated accumulators on the environment's device."""
    self.train_loss_accum = torch.zeros(1, device=self.env_device)
    self.train_correct_accum = torch.zeros(1, device=self.env_device)
    ...

def zero_accumulators(self) -> None:
    """Zero accumulators at the start of each epoch (faster than reallocating)."""
    self.train_loss_accum.zero_()
```

Excellent pattern: Pre-allocating accumulators and using `zero_()` instead of reallocation avoids GPU memory fragmentation and allocation overhead.

### 4.2 record_stream Usage

**Severity**: INFO (Positive)
**Location**: Lines 1188-1190, 1327-1329

```python
if env_state.stream and inputs.is_cuda:
    inputs.record_stream(env_state.stream)
    targets.record_stream(env_state.stream)
```

Correct usage of `record_stream` to prevent premature deallocation when tensors are created on the default stream but consumed by a non-default stream.

### 4.3 Best State CPU Offload

**Severity**: INFO (Positive)
**Location**: Lines 2506-2507

```python
# Store on CPU to save GPU memory (checkpoint is rarely loaded)
best_state = {k: v.cpu().clone() for k, v in agent.network.state_dict().items()}
```

Correct pattern for checkpoint storage: moving to CPU saves GPU memory for the active training workload.

### 4.4 Hidden State Cloning for Buffer Storage

**Severity**: LOW
**Location**: Lines 1853, 1863-1865

```python
pre_step_hiddens = [(h.clone(), c.clone()) for h, c in zip(h_list, c_list)]
...
env_h = init_h[:, env_idx:env_idx+1, :].clone()
env_c = init_c[:, env_idx:env_idx+1, :].clone()
```

The hidden state cloning is necessary for correct BPTT but represents memory overhead. For very long episodes, this could accumulate. Consider whether the hidden states need to be stored with each transition or only at episode boundaries.

### 4.5 Potential Memory Leak in raw_states_for_normalizer_update

**Severity**: MEDIUM
**Location**: Lines 1388, 1828, 2238

```python
raw_states_for_normalizer_update = []
...
raw_states_for_normalizer_update.append(states_batch.detach())
...
all_raw_states = torch.cat(raw_states_for_normalizer_update, dim=0)
obs_normalizer.update(all_raw_states)
```

Over a long episode (`max_epochs` steps), this list accumulates tensors. While they are detached, the concatenation at the end creates a large temporary tensor. For `max_epochs=200` with `n_envs=4` and `state_dim=80`, this is approximately:
- 200 * 4 * 80 * 4 bytes = 256 KB per episode

This is acceptable but worth documenting. The list is cleared after each batch, so no true leak exists.

---

## 5. Integration Risks

### 5.1 Governor Rollback State Consistency

**Severity**: HIGH
**Location**: Lines 1929-1931, 2217-2232

```python
if env_idx in governor_panic_envs:
    env_state.governor.execute_rollback(env_id=env_id)
    batch_rollback_occurred = True  # Mark batch as having stale transitions
...
if batch_rollback_occurred:
    agent.buffer.reset()
```

The Governor rollback mechanism invalidates the entire batch buffer when any environment experiences a panic. This is correct for safety but has significant sample efficiency implications:

1. All environments' data is discarded, not just the panicking one
2. No partial recovery is attempted
3. The rollback counter is not visible in the telemetry summary

**Recommendation**: Consider per-environment buffer segments that can be selectively invalidated.

### 5.2 Circular Import Risk

**Severity**: LOW
**Location**: Lines 725-726

```python
from esper.tolaria import create_model
from esper.tamiyo import SignalTracker
```

Late imports inside the function body avoid circular import issues at module load time. However, this pattern has a small per-call overhead. For a function called once per batch, this is negligible.

### 5.3 BlueprintAnalytics Side Effects

**Severity**: LOW
**Location**: Lines 835-836, 1101-1102

```python
analytics = BlueprintAnalytics(quiet=quiet_analytics)
hub.add_backend(analytics)
...
analytics.set_host_params(env_idx, host_params)
```

The `BlueprintAnalytics` is added to a global hub and mutated throughout training. This introduces:
1. Global state that persists across function calls
2. Potential thread-safety issues if `get_hub()` is ever called from multiple threads

### 5.4 SharedBatchIterator Integration

**Severity**: INFO
**Location**: Lines 946-956

```python
shared_train_iter = SharedBatchIterator(
    dataset=train_dataset,
    batch_size_per_env=batch_size_per_env,
    n_envs=n_envs,
    env_devices=env_device_map,
    num_workers=effective_workers,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    generator=gen,
)
```

The `SharedBatchIterator` is a custom abstraction that replaces N independent DataLoaders with one shared iterator. This is correctly implemented, but the integration assumes:
1. All environments process the same number of batches per epoch
2. The iterator yields exactly `n_envs` batches per iteration

If these assumptions are violated, silent data misalignment could occur.

---

## 6. Code Quality

### 6.1 Function Length

**Severity**: MEDIUM
**Location**: Lines 652-2544 (`train_ppo_vectorized`)

The main function is approximately 1900 lines long. This violates typical code quality guidelines and makes the function difficult to:
- Test in isolation
- Reason about
- Refactor safely

**Recommendation**: Extract the following into separate functions/classes:
1. Environment creation and initialization (lines 1078-1142)
2. Epoch training loop (lines 1396-2198)
3. PPO update and metrics aggregation (lines 2199-2276)
4. Telemetry emission (various locations)
5. Checkpoint save/load (lines 996-1027, 2520-2538)

### 6.2 Magic Numbers

**Severity**: LOW
**Location**: Various

```python
# Line 1084
torch.manual_seed(base_seed + env_idx * 1000)

# Line 1360
base_seed = seed + batch_idx * 10000

# Line 1670
if epoch % 5 == 0:
    env_state.governor.snapshot()
```

These magic numbers control reproducibility and checkpoint frequency. They should be named constants or configuration parameters.

### 6.3 Exception Handling Gaps

**Severity**: MEDIUM
**Location**: Lines 1429-1434, 1543-1548

```python
try:
    fetch_start = time.perf_counter()
    env_batches = next(train_iter)
    dataloader_wait_ms_epoch += (time.perf_counter() - fetch_start) * 1000.0
except StopIteration:
    break
```

The `StopIteration` handling is correct, but there's no handling for:
1. DataLoader worker crashes
2. CUDA OOM during data transfer
3. Corrupted data batches

These would propagate as unhandled exceptions and terminate training without cleanup.

### 6.4 Type Annotation Gaps

**Severity**: LOW
**Location**: Lines 389, 391, 400

```python
signal_tracker: any  # SignalTracker from tamiyo
...
action_enum: type | None = None
...
telemetry_cb: any = None
```

Using `any` type annotations defeats the purpose of type checking. These should use proper type imports with `TYPE_CHECKING` guards if needed.

### 6.5 Docstring Completeness

**Severity**: LOW
**Location**: Line 691

The main function docstring at lines 691-724 documents many but not all parameters. Missing documentation for:
- `telemetry_config`
- `telemetry_lifecycle_only`
- `slots`
- `max_seeds`
- `param_budget`
- `param_penalty_weight`
- `sparse_reward_scale`
- `reward_family`
- `quiet_analytics`

---

## 7. Performance Considerations

### 7.1 Stream Synchronization Pattern

**Severity**: INFO (Positive)
**Location**: Lines 1489-1492, 1624-1627

```python
# Sync all streams ONCE at epoch end
for env_state in env_states:
    if env_state.stream:
        env_state.stream.synchronize()
```

Correct pattern: Synchronizing once at epoch end rather than per-batch maximizes async execution benefits.

### 7.2 Fused Validation + Counterfactual

**Severity**: INFO (Positive)
**Location**: Lines 1498-1503

```python
# VALIDATION + COUNTERFACTUAL (FUSED): Single pass over test data
# Instead of iterating test data twice (once for main validation, once for
# counterfactual), we fuse both into a single loop.
```

Excellent optimization: Fusing the validation and counterfactual passes eliminates DataLoader overhead and improves throughput.

### 7.3 Telemetry Overhead

**Severity**: MEDIUM
**Location**: Lines 2095-2127

```python
if hub and use_telemetry and (
    telemetry_config is None or telemetry_config.should_collect("ops_normal")
):
    masked_flags = {
        "slot": not bool(masks_batch["slot"][env_idx].all().item()),
        ...
    }
    ...
    _emit_last_action(...)
```

The `.item()` calls inside the telemetry block cause GPU synchronization. This happens per-environment, per-epoch, negating some of the benefits of async stream execution.

**Recommendation**: Batch telemetry collection and emit after stream synchronization.

### 7.4 Dictionary Creation in Hot Path

**Severity**: LOW
**Location**: Lines 1888-1891

```python
actions = [
    {key: actions_dict[key][i].item() for key in actions_dict}
    for i in range(len(env_states))
]
```

Creating dictionaries in a list comprehension with `.item()` calls causes synchronization. Consider using tensor operations to batch this conversion.

---

## 8. Severity Summary

| Severity | Count | Categories |
|----------|-------|------------|
| HIGH | 1 | Governor rollback invalidates all env data |
| MEDIUM | 6 | AMP scaler sharing, function length, exception handling, device string comparison, telemetry overhead, missing torch.compile |
| LOW | 9 | Magic numbers, type annotations, docstrings, device typing, gradient accumulation, etc. |
| INFO | 7 | Positive patterns and observations |

---

## 9. Recommended Actions

### Immediate (Before Next Release)
1. **Create per-device GradScalers** for multi-GPU AMP correctness
2. **Add exception handling** for DataLoader failures with graceful cleanup
3. **Replace device string comparisons** with `torch.device.type` checks

### Short-term (Next Sprint)
4. **Extract epoch training loop** into a separate class/function
5. **Batch telemetry emission** to avoid per-step synchronization
6. **Add torch.compile** to the extracted forward/backward logic

### Long-term (Technical Debt)
7. **Implement per-environment buffer segments** for Governor rollback granularity
8. **Add proper type annotations** using TYPE_CHECKING guards
9. **Document all function parameters** in docstrings
10. **Add constants** for magic numbers

---

## 10. Testing Recommendations

1. **Multi-GPU stress test**: Run with `--devices cuda:0 cuda:1` and verify AMP scaling is correct per-device
2. **Governor rollback test**: Inject artificial panics and verify buffer clearing and telemetry
3. **Memory profiling**: Run with `torch.cuda.memory._record_memory_history()` to detect leaks
4. **Compilation test**: Enable torch.compile on extracted hot paths and verify no graph breaks in steady-state
5. **Edge cases**: Test with `n_envs=1`, `max_epochs=1`, and `slots=['mid']` to verify boundary conditions
