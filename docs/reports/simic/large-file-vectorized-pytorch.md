# PyTorch Code Review: `vectorized.py`

**File:** `/home/john/esper-lite/src/esper/simic/training/vectorized.py`
**Lines:** 2458
**Reviewer:** PyTorch Engineering Specialist
**Date:** 2025-12-17

---

## Executive Summary

The `vectorized.py` file implements high-performance vectorized PPO training with multi-GPU support using CUDA streams for async execution. The implementation demonstrates **strong PyTorch engineering practices** overall, with careful attention to:

- GPU-accurate timing via CUDA events
- Async-safe gradient collection avoiding `.item()` inside stream contexts
- Proper `record_stream()` usage to prevent premature tensor deallocation
- Pre-allocated accumulators to avoid per-epoch allocation churn
- Correct device handling with explicit validation

However, the review identified several issues ranging from potential correctness problems to performance optimization opportunities. The most critical issues relate to:

1. **LSTM hidden state device mismatch risk** during checkpoint resume
2. **GradScaler misuse with multiple optimizer steps** per iteration
3. **Potential memory fragmentation** from repeated environment creation/destruction
4. **Missing gradient clipping** for seed optimizers

---

## Critical Issues (Bugs / Correctness)

### C1: GradScaler Used Incorrectly with Multiple Optimizers

**Location:** Lines 1031-1035

```python
if use_amp and env_state.scaler is not None and env_dev.startswith("cuda"):
    env_state.scaler.step(env_state.host_optimizer)
    for slot_id in slots_to_step:
        env_state.scaler.step(env_state.seed_optimizers[slot_id])
    env_state.scaler.update()
```

**Problem:** When using AMP with multiple optimizers, calling `scaler.step()` multiple times followed by a single `scaler.update()` is the correct pattern. However, the scale factor is shared across all optimizer steps within an iteration. If one optimizer step skips due to NaN/Inf gradients, the scaler still updates its scale factor based on that failure, which can cause the subsequent optimizer steps to also be affected inconsistently.

**Risk:** Inconsistent training behavior where host and seed optimizers may step with different effective learning rates during scale adjustments.

**Recommendation:** Either:
1. Use separate GradScalers for host and seed optimizers (cleanest)
2. Check `scaler.get_scale()` before and after to detect skipped steps
3. Document this as known behavior and monitor for NaN rates diverging between host/seed

### C2: LSTM Hidden State Device Mismatch on Resume

**Location:** Lines 730-742

```python
checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
agent = PPOAgent.load(resume_path, device=device)

# Restore observation normalizer state
metadata = checkpoint.get('metadata', {})
if 'obs_normalizer_mean' in metadata:
    # Create tensors directly on target device to avoid CPU->GPU transfer
    obs_normalizer.mean = torch.tensor(metadata['obs_normalizer_mean'], device=device)
```

**Problem:** The checkpoint is loaded with `map_location=device`, but `PPOAgent.load()` is called separately. If `PPOAgent.load()` doesn't consistently use the same device mapping, the agent's internal state could be on a different device.

Additionally, the LSTM hidden states (`env_state.lstm_hidden`) are initialized as `None` and then populated during training. On resume, new `env_states` are created fresh (line 1104-1107), but the LSTM hidden states from the checkpoint are not restored. This means resumed training always starts with fresh hidden states, losing temporal context.

**Impact:** Temporal context loss on resume; potential device mismatch warnings/errors.

**Recommendation:**
1. Store and restore LSTM hidden states in checkpoints
2. Verify `PPOAgent.load()` respects the device parameter consistently

### C3: Seed Optimizer Missing Gradient Clipping

**Location:** Lines 979-981, 1037-1039

```python
env_state.seed_optimizers[slot_id] = torch.optim.SGD(
    seed_params, lr=task_spec.seed_lr, momentum=0.9
)
# ... later ...
env_state.host_optimizer.step()
for slot_id in slots_to_step:
    env_state.seed_optimizers[slot_id].step()
```

**Problem:** The host optimizer uses `max_grad_norm` clipping via the PPO agent's update, but seed optimizers are stepped directly without gradient clipping. Seeds are smaller networks that can be more susceptible to gradient explosions, especially early in training when their outputs have high variance.

**Impact:** Seed training instability; potential for exploding gradients during seed TRAINING stage.

**Recommendation:** Add gradient clipping before seed optimizer steps:
```python
for slot_id in slots_to_step:
    seed_params = list(model.get_seed_parameters(slot_id))
    if seed_params:
        torch.nn.utils.clip_grad_norm_(seed_params, max_norm=1.0)
    env_state.seed_optimizers[slot_id].step()
```

---

## High-Priority Issues (Performance / Stability)

### H1: Environment Recreation Causes Memory Fragmentation

**Location:** Lines 1103-1107

```python
base_seed = seed + batch_idx * 10000
env_states = [
    create_env_state(i, base_seed)
    for i in range(envs_this_batch)
]
```

**Problem:** Every batch creates entirely new environments (models, optimizers, CUDA streams). While this ensures clean state for each batch, it causes:
1. **Memory fragmentation:** Repeated allocation/deallocation of model parameters
2. **CUDA context overhead:** Stream creation/destruction has non-trivial cost
3. **No model weight sharing:** Host models could share frozen backbone weights

**Impact:** Suboptimal GPU memory utilization; increased allocation overhead.

**Recommendation:** Consider environment pooling where models are reset rather than recreated:
```python
def reset_env_state(env_state: ParallelEnvState, seed: int) -> None:
    """Reset environment state without reallocating tensors."""
    env_state.model.reset_to_checkpoint()  # Restore host weights
    env_state.episode_rewards.clear()
    env_state.zero_accumulators()
    # ... reset other state
```

### H2: Redundant Device Validation in Hot Path

**Location:** Lines 939-941, 1066-1068

```python
if env_state.stream and inputs.is_cuda:
    inputs.record_stream(env_state.stream)
    targets.record_stream(env_state.stream)
```

**Problem:** The `.is_cuda` check happens inside the stream context for every batch. Since `env_state.stream` existence already implies CUDA usage, this is redundant.

**Impact:** Minor CPU overhead per batch; pattern repeated multiple times.

**Recommendation:** Simplify to:
```python
if env_state.stream:
    inputs.record_stream(env_state.stream)
    targets.record_stream(env_state.stream)
```

### H3: Synchronization Point Before Validation Loop

**Location:** Lines 1233-1236

```python
# Sync all streams ONCE at epoch end
for env_state in env_states:
    if env_state.stream:
        env_state.stream.synchronize()
```

**Problem:** This is correct for training, but validation immediately follows and requires another sync at line 1369-1371. Consider whether validation can begin while training results are being synchronized, using stream dependencies instead of full synchronization.

**Impact:** Sequential execution where overlap might be possible.

**Recommendation:** Consider using CUDA events for dependency instead of synchronization:
```python
# After training
train_complete_events = []
for env_state in env_states:
    if env_state.stream:
        event = torch.cuda.Event()
        event.record(env_state.stream)
        train_complete_events.append(event)

# Validation can start on default stream after event
for event in train_complete_events:
    torch.cuda.current_stream().wait_event(event)
```

### H4: Inefficient Tensor Creation in Bootstrap Path

**Location:** Lines 1954-1956

```python
post_action_state = torch.tensor(
    [post_action_features], dtype=torch.float32, device=device
)
```

**Problem:** For truncated episodes (every episode ends truncated per line 1906), this creates a new tensor from a Python list. This involves:
1. Python list iteration
2. Tensor allocation
3. Memory copy

This happens for every environment at every episode end.

**Impact:** Unnecessary CPU-GPU transfer overhead at episode boundaries.

**Recommendation:** Pre-allocate and reuse:
```python
# In env_state initialization
env_state.bootstrap_state_buffer = torch.zeros(1, state_dim, device=device)

# In bootstrap computation
env_state.bootstrap_state_buffer[0].copy_(
    torch.as_tensor(post_action_features, dtype=torch.float32)
)
post_action_normalized = obs_normalizer.normalize(env_state.bootstrap_state_buffer)
```

### H5: Dictionary Comprehension in Hot Path

**Location:** Lines 1568-1571

```python
masks_batch = {
    key: torch.stack([m[key] for m in all_masks]).to(device)
    for key in all_masks[0].keys()
}
```

**Problem:** This creates intermediate lists and stacks tensors inside a dict comprehension every epoch. The masks are boolean tensors that could be pre-allocated.

**Impact:** Memory churn and GC pressure per epoch.

**Recommendation:** Pre-allocate mask tensors in buffer:
```python
# Pre-allocated in buffer
batch_masks = {
    "slot": torch.zeros(n_envs, num_slots, dtype=torch.bool, device=device),
    # ... etc
}

# In loop, copy instead of stack
for i, mask in enumerate(all_masks):
    for key in mask:
        batch_masks[key][i] = mask[key]
```

---

## Medium-Priority Issues (Best Practices)

### M1: Type Annotation Inconsistency

**Location:** Line 313

```python
def _emit_anomaly_diagnostics(
    hub: any,  # Should be: hub: Any | None
```

**Problem:** Using lowercase `any` instead of `typing.Any`.

**Impact:** No runtime effect, but breaks static type checking.

### M2: Unused Parameter in Nested Function

**Location:** Lines 800-816

```python
def configure_slot_telemetry(
    env_state: ParallelEnvState,
    *,
    inner_epoch: int | None = None,
    global_epoch: int | None = None,
) -> None:
```

**Problem:** `inner_epoch` and `global_epoch` are passed but only forwarded to `apply_slot_telemetry`. If `apply_slot_telemetry` doesn't use them, they're noise.

**Impact:** Code clarity; potential confusion about what these parameters control.

### M3: Magic Numbers in Gradient Health Calculation

**Location:** Lines 2136-2137

```python
grad_health = 1.0 if 0.01 <= ppo_grad_norm <= 100.0 else max(0.0, 1.0 - abs(ppo_grad_norm - 50) / 100)
```

**Problem:** Magic numbers `0.01`, `100.0`, `50`, `100` are not defined as constants.

**Impact:** Hard to tune; unclear semantics.

**Recommendation:** Extract to named constants in leyline:
```python
from esper.leyline import (
    GRAD_NORM_HEALTHY_MIN,
    GRAD_NORM_HEALTHY_MAX,
    # ...
)
```

### M4: Potential Division by Zero

**Location:** Lines 2194-2196

```python
avg_step_time_ms = throughput_step_time_ms_sum / max(max_epochs, 1)
avg_dataloader_wait_ms = throughput_dataloader_wait_ms_sum / max(max_epochs, 1)
```

**Problem:** While protected against zero division, if `max_epochs` is somehow 0 (which should be invalid), the function would return misleading metrics instead of failing fast.

**Recommendation:** Add validation at function entry:
```python
if max_epochs <= 0:
    raise ValueError(f"max_epochs must be positive, got {max_epochs}")
```

### M5: Inconsistent Exception Handling

**Location:** Lines 2245-2250

```python
try:
    layer_stats = collect_per_layer_gradients(agent.policy)
    layer_health = aggregate_layer_gradient_health(layer_stats)
    payload.update(layer_health)
except Exception:
    pass  # Graceful degradation if collection fails
```

**Problem:** Bare `except Exception: pass` suppresses all errors including bugs. If `agent.policy` is accessed incorrectly (it should be `agent.network`), this silently fails.

**Impact:** Hidden bugs; debugging difficulty.

**Recommendation:**
```python
except Exception as e:
    logger.debug(f"Layer gradient collection failed: {e}")
```

### M6: torch.compile Compatibility - Potential Graph Break

**Location:** Lines 947-956

```python
for slot_id in slots:
    if not model.has_active_seed_in_slot(slot_id):
        continue
    slot_state = model.seed_slots[slot_id].state
    if slot_state and slot_state.stage == SeedStage.GERMINATED:
        gate_result = model.seed_slots[slot_id].advance_stage(SeedStage.TRAINING)
        if not gate_result.passed:
            raise RuntimeError(...)
```

**Problem:** The conditional control flow with dict access and method calls can cause graph breaks under `torch.compile`. While the process_train_batch function runs model.forward which IS compiled, this setup code could interfere with whole-function compilation.

**Impact:** Potential performance regression if torch.compile is applied at higher scope.

**Recommendation:** Consider marking this setup section with `torch.compiler.disable` or ensuring the compiled boundary is at the model.forward call.

---

## Low-Priority Suggestions

### L1: Consider Using torch.inference_mode() Consistently

**Location:** Lines 1070-1076

```python
model.eval()
with torch.inference_mode():
    outputs = model(inputs)
```

**Observation:** This correctly uses `inference_mode()` for validation, which is more efficient than `no_grad()`. Good practice.

### L2: Consider Fusing Accumulator Updates

**Location:** Lines 1198-1201

```python
with stream_ctx:
    env_state.train_loss_accum.add_(loss_tensor)
    env_state.train_correct_accum.add_(correct_tensor)
```

**Observation:** These are two separate in-place operations. For maximum efficiency, consider fusing:
```python
# If losses and correct were stacked, could use single add_
```

This is very minor and the current pattern is clean.

### L3: Documentation Inconsistency

**Location:** Lines 413-446 (docstring)

**Observation:** The docstring lists many parameters but some newer ones like `slots`, `max_seeds`, `reward_family` are documented inline rather than in the main args section. Consider consolidating.

### L4: Consider Using torch.utils.checkpoint for Memory

**Location:** process_train_batch function

**Observation:** For very large host models, activation checkpointing could reduce memory at the cost of recomputation. Not currently needed for CIFAR-10 but worth considering for future scaling.

---

## Positive Patterns (Worth Preserving)

### P1: Excellent CUDA Stream Usage

The implementation correctly:
- Uses `record_stream()` to prevent premature tensor deallocation
- Synchronizes streams before accessing `.item()` values
- Uses stream context managers for scoped execution

### P2: Async-Safe Gradient Collection

The split between `collect_*_async()` and `materialize_*()` functions correctly defers GPU-CPU synchronization until after stream sync.

### P3: Pre-Allocated Accumulators

`ParallelEnvState.init_accumulators()` and `zero_accumulators()` properly reuse tensor memory instead of reallocating each epoch.

### P4: GPU-Accurate Timing

`CUDATimer` using CUDA events provides accurate GPU kernel timing, avoiding the common mistake of measuring CPU-side queue time.

### P5: Proper Bootstrap Value Handling

The truncation handling at lines 1905-1975 correctly computes V(s_{t+1}) for truncated episodes, avoiding the systematic downward bias that would occur from treating truncation as termination.

### P6: EMA-Based Observation Normalization

Using EMA mode for `RunningMeanStd` (momentum=0.99) prevents distribution shift during long training runs while maintaining responsiveness to changing observation distributions.

---

## Summary of Recommendations

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| Critical | C1: GradScaler multi-optimizer | Medium | Training stability |
| Critical | C2: LSTM hidden state resume | Medium | Checkpoint correctness |
| Critical | C3: Seed gradient clipping | Low | Training stability |
| High | H1: Environment recreation | High | Memory efficiency |
| High | H4: Bootstrap tensor creation | Low | Per-episode overhead |
| High | H5: Mask tensor allocation | Low | Per-epoch overhead |
| Medium | M3: Magic numbers | Low | Maintainability |
| Medium | M5: Exception handling | Low | Debuggability |

---

## Appendix: torch.compile Compatibility Notes

The codebase uses `torch.compile(mode="default")` on the PPO network. Key observations:

1. **MaskedCategorical**: Has `@torch.compiler.disable` on validation (line mentioned in network.py) - good practice
2. **Dynamic control flow**: The main training loop has extensive Python control flow which would cause graph breaks if the whole function were compiled
3. **_foreach_norm usage**: This is a stable internal API used by `clip_grad_norm_` - compilation compatible

**Recommendation:** The current compilation boundary (network only) is appropriate. Do not attempt to compile the full training loop.
