# PyTorch Deep Review: simic/training/ Subfolder

**Reviewer**: PyTorch Specialist Agent
**Date**: 2025-12-17
**Files Reviewed**:
- `vectorized.py` (2458 lines) - Main vectorized PPO training loop
- `helpers.py` (684 lines) - Heuristic training and compiled train step
- `config.py` (318 lines) - TrainingConfig dataclass
- `parallel_env_state.py` (111 lines) - Per-environment state management
- `__init__.py` (30 lines) - Package exports

**PyTorch Version Target**: 2.0+ (torch.compile compatible)

---

## Executive Summary

The simic/training subfolder implements a sophisticated vectorized PPO training system with CUDA stream-based parallelization. Overall code quality is **high**, with careful attention to GPU memory management, CUDA synchronization, and numerical stability. The architecture follows the "inverted control flow" pattern documented in ROADMAP.md.

**Key Findings**:
- 1 Critical correctness issue (potential memory leak in Governor snapshots)
- 3 High-priority issues (torch.compile compatibility, AMP edge cases)
- 5 Medium-priority issues (best practices, defensive programming)
- 7 Low-priority suggestions (minor optimizations, style)

---

## Critical Issues

### CRIT-1: Governor Snapshot Memory Leak in Long Training Runs

**Location**: `vectorized.py` lines 867-876 (Governor creation) + external `tolaria/governor.py`

**Issue**: The TolariaGovernor snapshots model state every 5 epochs (line 1414-1415):
```python
if epoch % 5 == 0:
    env_state.governor.snapshot()
```

The `snapshot()` method in `governor.py` (lines 83-127) creates a deep copy of filtered state dict. While it explicitly frees the old snapshot (lines 95-97), the memory is stored on CPU via `.cpu().clone()`.

**The Bug**: When training with many environments over hundreds of episodes, each Governor maintains its own snapshot. With 4 envs and a 500K parameter model, this is ~8MB per env constantly held in RAM. More critically, if `last_good_state` contains references to CUDA tensors (e.g., through non-tensor objects in the state dict), those tensors won't be freed.

**Impact**: Potential OOM in long runs (1000+ episodes) or slow memory leak.

**Recommendation**:
1. Add explicit `torch.cuda.empty_cache()` after snapshot deletion when GPU tensors detected
2. Consider a single shared snapshot per batch instead of per-env
3. Add memory profiling hooks for training runs > 100 episodes

---

## High-Priority Issues

### HIGH-1: torch.compile Graph Breaks in helpers.py

**Location**: `helpers.py` lines 64-89 (`_get_compiled_train_step`)

**Issue**: The compiled train step uses `functools.cache` which is thread-safe but prevents recompilation when input shapes change dynamically. The comment correctly notes CUDA graphs aren't used, but there's a subtle issue:

```python
@functools.cache
def _get_compiled_train_step(use_compile: bool = True) -> Callable:
    if use_compile:
        try:
            return torch.compile(_train_step_impl, mode="default")
        except Exception:
            return _train_step_impl
```

**Problems**:
1. `functools.cache` caches the compiled function globally - if different training runs use different batch sizes or model architectures, they share the same compiled kernel (which may have been specialized for different shapes)
2. The `except Exception` catches compilation errors silently without logging

**Recommendation**:
```python
@functools.cache
def _get_compiled_train_step(use_compile: bool = True) -> Callable:
    if use_compile:
        try:
            # Use dynamic=True for shape flexibility in vectorized training
            return torch.compile(_train_step_impl, mode="default", dynamic=True)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"torch.compile failed: {e}, using eager mode")
            return _train_step_impl
    return _train_step_impl
```

### HIGH-2: AMP GradScaler Per-Env Race Condition Window

**Location**: `vectorized.py` lines 852-853, 996-999, 1031-1035

**Issue**: The code correctly creates per-env GradScalers to avoid stream race conditions:
```python
env_scaler = torch_amp.GradScaler(enabled=amp_enabled) if env_device_obj.type == "cuda" else None
```

However, the GradScaler's internal `_scale` tensor is updated during `scaler.update()` (line 1035). If two streams call `scaler.update()` before the previous update's CUDA work completes, the scale factor can become inconsistent.

**Current Code**:
```python
if use_amp and env_state.scaler is not None and env_dev.startswith("cuda"):
    env_state.scaler.scale(loss).backward()
# ... later ...
    env_state.scaler.step(env_state.host_optimizer)
    for slot_id in slots_to_step:
        env_state.scaler.step(env_state.seed_optimizers[slot_id])
    env_state.scaler.update()
```

**The Risk**: Each env has its own scaler, but all scalers for the same device share CUDA memory. The `update()` call modifies internal state based on the previous `step()` results. If the stream hasn't synced before another iteration starts, the scale factor may be stale.

**Mitigation**: The code does sync streams at epoch end (lines 1233-1236), but this happens AFTER the batch processing. Within a single batch (multiple train batches), there's no sync between scalers.

**Recommendation**: Add a comment documenting this is safe because scalers are per-env and streams are per-env, so no cross-env scaler access occurs. Alternatively, sync after each `scaler.update()` call for maximum safety:
```python
env_state.scaler.update()
if env_state.stream:
    env_state.stream.synchronize()  # Ensure scale factor is visible
```

### HIGH-3: Observation Normalizer Statistics Update Race

**Location**: `vectorized.py` lines 549-555, 1573-1582, 2287-2293

**Issue**: The `RunningMeanStd` normalizer is updated with raw states AFTER PPO update (line 2291-2293 in `_run_ppo_updates`):
```python
if raw_states_for_normalizer_update and update_metrics:
    all_raw_states = torch.cat(raw_states_for_normalizer_update, dim=0)
    obs_normalizer.update(all_raw_states)
```

This is correct for frozen normalization during rollout collection. However, `RunningMeanStd.update()` (in `normalization.py` lines 48-63) operates on `self.mean` and `self.var` tensors directly without any locking.

**Risk**: If multiple training batches are run in quick succession (e.g., in a multi-threaded future extension), the mean/var tensors could be updated concurrently, leading to corrupted statistics.

**Current Safety**: The code is single-threaded, so this is not currently a bug. But the normalizer's docstring claims "GPU-native" which could mislead future developers into using it in multi-threaded contexts.

**Recommendation**: Add thread-safety documentation to `RunningMeanStd`:
```python
class RunningMeanStd:
    """Running mean and std for observation normalization.

    WARNING: Not thread-safe. Use only from a single thread.
    GPU-native refers to tensor operations, not thread safety.
    """
```

---

## Medium-Priority Issues

### MED-1: Unnecessary Tensor Creation in Validation Loop

**Location**: `vectorized.py` lines 1301-1325 (counterfactual validation)

**Issue**: Inside the validation loop, for each slot needing counterfactual, a new `stream_ctx` context manager is created:
```python
for slot_id in slots_needing_counterfactual[i]:
    with stream_ctx:
        with env_state.model.seed_slots[slot_id].force_alpha(0.0):
            _, cf_correct_tensor, cf_total = process_val_batch(...)
        env_state.cf_correct_accums[slot_id].add_(cf_correct_tensor)
    env_state.cf_totals[slot_id] += cf_total
```

The `cf_totals[slot_id] += cf_total` happens OUTSIDE the stream context but uses a value (`cf_total`) that was returned from GPU computation. This is safe because `total` is a Python int computed on CPU from `targets.numel()`, but it's confusing.

**Recommendation**: Move the `cf_totals` update inside the stream context for clarity, or add a comment explaining why it's safe.

### MED-2: Missing Input Validation for slots Parameter

**Location**: `vectorized.py` line 485-486, `helpers.py` line 305-309

**Issue**: Both files validate slots but with slightly different error messages:
```python
# vectorized.py
if not slots:
    raise ValueError("slots parameter is required and cannot be empty")

# helpers.py
if not slots:
    raise ValueError("slots parameter is required and cannot be empty")
if len(slots) != len(set(slots)):
    raise ValueError(f"slots contains duplicates: {slots}")
```

`vectorized.py` doesn't check for duplicates, while `helpers.py` does. The `validate_slot_ids` function is called later, but duplicate detection happens earlier in helpers.py.

**Recommendation**: Unify validation logic. Consider moving duplicate detection to `validate_slot_ids` in leyline/slot_id.py for single source of truth.

### MED-3: Hardcoded Batch Size Override

**Location**: `vectorized.py` lines 538-540

**Issue**: CIFAR-10 batch size is hardcoded to 512 regardless of task_spec settings:
```python
batch_size_per_env = task_spec.dataloader_defaults.get("batch_size", 128)
if task_spec.name == "cifar10":
    batch_size_per_env = 512  # High-throughput setting for CIFAR
```

This silently overrides any user-configured batch size and could cause OOM on smaller GPUs.

**Recommendation**: Make this configurable or at least log when overriding:
```python
if task_spec.name == "cifar10" and batch_size_per_env < 512:
    logging.info(f"Overriding batch_size {batch_size_per_env} -> 512 for CIFAR-10 throughput")
    batch_size_per_env = 512
```

### MED-4: Potential Division by Zero in Gradient Health Calculation

**Location**: `gradient_collector.py` line 178 (called from vectorized.py)

**Issue**: The gradient norm is computed as:
```python
gradient_norm = (total_squared_norm ** 0.5) / n_grads
```

If `n_grads` is somehow 0 (e.g., all parameters frozen), this would raise a ZeroDivisionError. The code checks for empty grads earlier, but edge cases with `requires_grad=False` could slip through.

**Recommendation**: Add defensive check:
```python
gradient_norm = (total_squared_norm ** 0.5) / max(n_grads, 1)
```

### MED-5: Bootstrap Value Computation Uses Wrong Hidden State

**Location**: `vectorized.py` lines 1962-1973

**Issue**: When computing bootstrap value for truncated episodes, the code uses `env_state.lstm_hidden`:
```python
with torch.inference_mode():
    _, _, bootstrap_tensor, _ = agent.network.get_action(
        post_action_normalized,
        hidden=env_state.lstm_hidden,  # This is POST-step hidden
        ...
    )
```

The `env_state.lstm_hidden` was just updated (lines 1627-1631) to the POST-action hidden state. For V(s_{t+1}), this is correct - we want the hidden state after the action. However, the comment on line 1907-1910 says "use V(s_{t+1})", which is ambiguous about whether this means pre- or post-action.

**Observation**: After careful analysis, this is actually correct - V(s_{t+1}) should use the hidden state that would be passed to the next step, which is the post-action hidden. No change needed, but the comment could be clearer.

---

## Low-Priority Suggestions

### LOW-1: Unused `quiet_analytics` Propagation

**Location**: `vectorized.py` line 411, 568

The `quiet_analytics` parameter is passed to `train_ppo_vectorized` and used to create `BlueprintAnalytics(quiet=quiet_analytics)`. However, it's not exposed in `TrainingConfig.to_train_kwargs()` in config.py, so it can only be set programmatically.

**Suggestion**: Either add to TrainingConfig or remove from function signature if it's internal-only.

### LOW-2: Magic Number in Rolling Average Window

**Location**: `vectorized.py` lines 2175-2177

```python
if len(recent_accuracies) > 10:
    recent_accuracies.pop(0)
```

The window size 10 is hardcoded. Consider making this a constant or config parameter.

### LOW-3: Inconsistent Device String Handling

**Location**: Throughout vectorized.py

The code mixes `str` device strings and `torch.device` objects:
- Line 450: `def _parse_device(device_str: str) -> torch.device`
- Line 989: `if use_amp and env_dev.startswith("cuda")`
- Line 849: `env_device_obj = torch.device(env_device)`

**Suggestion**: Standardize on `torch.device` objects internally, converting strings only at API boundaries.

### LOW-4: Redundant Empty Check

**Location**: `helpers.py` line 207

```python
if grad_stats is not None and not grad_stats.get('_empty', False):
    grad_stats = materialize_grad_stats(grad_stats)
```

The `_empty` check is redundant because `materialize_grad_stats` handles the empty case internally (line 162-169 in gradient_collector.py).

### LOW-5: Missing Type Hints for Nested Dict Returns

**Location**: `vectorized.py` line 904, `helpers.py` line 132

Return type hints for gradient stats are inconsistent:
```python
# vectorized.py
def process_train_batch(...) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict] | None]:
# Should be dict[str, dict[str, Any]] for clarity
```

### LOW-6: Comment-Code Drift in CUDATimer

**Location**: `vectorized.py` lines 121-157

The CUDATimer class is well-implemented but the docstring mentions "falls back to CPU timing when CUDA unavailable" while the implementation checks `device.startswith("cuda")`. A device like `"mps"` would get CPU timing, which is fine but not explicitly documented.

### LOW-7: Pre-allocated Accumulator Efficiency

**Location**: `parallel_env_state.py` lines 82-97

The `init_accumulators` method creates per-slot counterfactual accumulators:
```python
self.cf_correct_accums: dict[str, torch.Tensor] = {
    slot_id: torch.zeros(1, device=self.env_device) for slot_id in slots
}
```

With 3 slots, this creates 3 single-element tensors. Consider using a single tensor with 3 elements indexed by slot position for better memory locality.

---

## Cross-File Architectural Observations

### OBS-1: Clean Separation of Concerns

The architecture properly separates:
- **config.py**: Hyperparameter validation and serialization
- **parallel_env_state.py**: Per-environment state management
- **helpers.py**: Single-environment training (heuristic mode)
- **vectorized.py**: Multi-environment vectorized training

This enables testing each component in isolation.

### OBS-2: Consistent Use of Context Managers

CUDA stream contexts are consistently used via `torch.cuda.stream()` with `nullcontext()` fallback for CPU. This pattern is applied correctly throughout the codebase.

### OBS-3: Telemetry Integration Well-Designed

The telemetry callbacks use a factory pattern (`make_telemetry_callback`) that injects env_id context. This allows the Nissa hub to correlate events across parallel environments without tight coupling.

### OBS-4: Reward Normalization Architecture

The two-level normalization (observation + reward) is a DRL best practice:
- `RunningMeanStd` for observations (EMA mode for long runs)
- `RewardNormalizer` for rewards (Welford's algorithm, std-only normalization)

The reward normalizer correctly avoids mean subtraction to preserve reward semantics.

### OBS-5: torch.compile Compatibility

The codebase is largely torch.compile compatible:
- `TamiyoRolloutStep` uses `NamedTuple` (compile-safe)
- `TamiyoRolloutBuffer` uses pre-allocated tensors (compile-safe)
- `_train_step_impl` is separated from control flow (compile-safe)

However, the main training loop (`train_ppo_vectorized`) cannot be compiled due to Python-level control flow (conditionals, loops over environments). This is expected and correct.

---

## Numerical Stability Assessment

### STABLE: Loss Accumulation
Lines 1197-1201 use tensor accumulation with deferred `.item()` - correct pattern.

### STABLE: Gradient Norm Computation
Uses `torch._foreach_norm` (PyTorch 2.0+ stable internal API) for fused kernel efficiency.

### STABLE: Action Masking
Uses `-1e4` instead of `-inf` for mask values (line 40 in network.py) - prevents FP16 overflow.

### STABLE: Advantage Normalization
Happens in `TamiyoRolloutBuffer.normalize_advantages()` before PPO update - correct location.

### POTENTIAL ISSUE: Gradient Health Calculation
Line 2137 in vectorized.py:
```python
grad_health = 1.0 if 0.01 <= ppo_grad_norm <= 100.0 else max(0.0, 1.0 - abs(ppo_grad_norm - 50) / 100)
```
This formula has a discontinuity at boundaries (0.01, 100.0). Consider using a smooth function or documenting the rationale.

---

## Memory Management Assessment

### GOOD: Pre-allocated Accumulators
`ParallelEnvState.init_accumulators()` and `zero_accumulators()` pattern avoids per-epoch allocation churn.

### GOOD: Best State Storage
Line 2410-2411 stores best state on CPU:
```python
best_state = {k: v.cpu().clone() for k, v in agent.network.state_dict().items()}
```

### GOOD: record_stream Usage
Lines 939-941 use `record_stream` to prevent premature tensor deallocation in stream contexts.

### CONCERN: Global GPU Cache
`utils/data.py` line 114 uses a global cache for GPU-resident datasets:
```python
_GPU_DATASET_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
```
This persists across training runs and isn't cleared. Consider adding a `clear_gpu_cache()` function.

---

## Recommendations Summary

| Priority | ID | Action |
|----------|-----|--------|
| Critical | CRIT-1 | Add memory profiling for Governor snapshots in long runs |
| High | HIGH-1 | Add `dynamic=True` to torch.compile and log compilation failures |
| High | HIGH-2 | Document GradScaler stream safety or add sync after update |
| High | HIGH-3 | Add thread-safety warning to RunningMeanStd docstring |
| Medium | MED-1 | Clarify stream context usage in counterfactual loop |
| Medium | MED-2 | Unify slot duplicate validation |
| Medium | MED-3 | Make CIFAR-10 batch size override configurable |
| Medium | MED-4 | Add defensive division check in gradient norm calculation |
| Medium | MED-5 | Clarify bootstrap value hidden state comment |

---

## Conclusion

The simic/training subfolder demonstrates professional-grade PyTorch engineering with careful attention to:
- CUDA stream synchronization
- Memory-efficient accumulator patterns
- torch.compile boundaries
- Numerical stability

The identified issues are primarily edge cases and documentation improvements rather than fundamental design flaws. The critical Governor memory issue should be addressed before production use with very long training runs.

The codebase is ready for PyTorch 2.0+ deployment with the recommended fixes applied.
