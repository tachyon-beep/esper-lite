# Simic Training Loop Analysis

**Module:** `esper.simic` (training subsystem)
**Files Analyzed:**
- `/home/john/esper-lite/src/esper/simic/training.py`
- `/home/john/esper-lite/src/esper/simic/vectorized.py`
- `/home/john/esper-lite/src/esper/simic/gradient_collector.py`

**Reviewer:** DRL + PyTorch SME
**Date:** 2025-12-02

---

## 1. training.py

### Purpose
Single-environment PPO training loop for the Tamiyo seed lifecycle controller. Provides both PPO-based (`train_ppo`, `run_ppo_episode`) and heuristic-based (`train_heuristic`, `run_heuristic_episode`) training functions for comparison experiments.

### Key Classes/Functions

| Name | Lines | Description |
|------|-------|-------------|
| `_loss_and_correct()` | 25-43 | Utility for computing cross-entropy loss and accuracy for classification/LM tasks |
| `run_ppo_episode()` | 50-352 | Single episode runner with seed lifecycle state machine, gradient collection, and action execution |
| `train_ppo()` | 359-479 | Outer training loop managing episodes, PPO updates, and model checkpointing |
| `run_heuristic_episode()` | 486-678 | Heuristic policy episode runner for baseline comparisons |
| `train_heuristic()` | 681-746 | Heuristic training loop wrapper |

### DRL Assessment

**Training Loop Correctness:**
- GAE computation is deferred to `RolloutBuffer.compute_returns_and_advantages()` which implements standard GAE correctly
- Advantage normalization occurs in `PPOAgent.update()` (lines ppo.py:181)
- Episode structure is correct: state -> action -> reward -> store_transition -> epoch advance

**PPO Implementation:**
- Clip ratio applied correctly in surrogate loss (ppo.py:202-205)
- Value function clipping implemented (ppo.py:208-214)
- Entropy bonus with linear annealing support (ppo.py:148-159)
- Gradient clipping via `clip_grad_norm_` (ppo.py:227)

**Issues:**

| Severity | Issue | Line | Description |
|----------|-------|------|-------------|
| **MEDIUM** | Redundant stage handling branches | 120-234 | The training loop has 6 separate `elif` branches for different `SeedStage` values (GERMINATED, TRAINING, BLENDING, SHADOWING/PROBATIONARY, FOSSILIZED). These could be consolidated since GERMINATED immediately transitions to TRAINING and most branches share identical logic. |
| **MEDIUM** | Seed optimizer recreation on every episode | 136-139 | Creates new SGD optimizer each time seed enters TRAINING stage. Momentum buffers are lost between germination cycles. Intentional behavior but may cause training instability. |
| **LOW** | Missing early stopping | 110-352 | No mechanism to terminate episodes early on plateau or divergence |

### PyTorch Assessment

**Memory Patterns:**
- Training loop creates new model per episode (line 76) - appropriate for RL reset semantics
- Gradient stats collected only on last batch per epoch (lines 149-150, 172-173) - memory efficient

**CUDA Usage:**
- Single device training, no multi-GPU patterns used
- No explicit memory management (adequate for single-env training)
- Uses `torch.no_grad()` for validation (line 245)

**Issues:**

| Severity | Issue | Line | Description |
|----------|-------|------|-------------|
| **LOW** | Sync `collect_seed_gradients` called inside training loop | 150, 173 | Uses synchronous gradient collection which blocks CPU-GPU overlap. The async version exists but is not used here. Acceptable for single-env training but limits throughput. |
| **LOW** | Missing `torch.inference_mode()` | 245 | Uses `torch.no_grad()` instead of `inference_mode()` for validation. Minor efficiency loss from version tracking overhead. |

---

## 2. vectorized.py

### Purpose
High-performance vectorized PPO training using multiple parallel environments, CUDA streams for async GPU execution, and inverted control flow (batch-first iteration). Designed for multi-GPU scaling.

### Key Classes/Functions

| Name | Lines | Description |
|------|-------|-------------|
| `ParallelEnvState` | 48-84 | Dataclass holding per-environment state including model, optimizers, CUDA stream, and DataLoaders |
| `_advance_active_seed()` | 87-111 | Lifecycle advancement helper for fossilization transitions |
| `train_ppo_vectorized()` | 118-826 | Main vectorized training loop with CUDA stream orchestration |
| `create_env_dataloaders()` | 196-207 | Factory for per-env DataLoaders with unique RNG seeds |
| `create_env_state()` | 291-346 | Factory for complete environment state initialization |
| `process_train_batch()` | 348-429 | Single batch training step with CUDA stream context |
| `process_val_batch()` | 431-453 | Single batch validation step with CUDA stream context |

### DRL Assessment

**Training Loop Correctness:**
- Inverted control flow correctly batches across environments then iterates dataloader steps
- Per-env RNG seeds ensure reproducibility (line 294-295)
- Governor watchdog provides fail-safe rollback on catastrophic loss spikes (lines 612-618, 667-695)
- Action execution properly handles lifecycle transitions per Leyline constraints

**PPO Implementation:**
- Batched action sampling via `get_action_batch()` (line 650) - single forward pass for all envs
- Observation normalization via `RunningMeanStd` (lines 646-647)
- Updates after each batch of environments completes (line 741)

**Issues:**

| Severity | Issue | Line | Description |
|----------|-------|------|-------------|
| **HIGH** | State normalization applied AFTER storing transition | 725-732 | The raw (unnormalized) state is passed to `store_transition()`, but the normalized state was used for action selection. This creates a distribution mismatch during PPO updates where old_log_probs came from normalized states but training happens on unnormalized buffer states. |
| **MEDIUM** | PPO update happens per batch, not per episode | 741 | With `n_envs=4` and `max_epochs=25`, each PPO update uses 100 transitions. Standard PPO typically uses larger rollout buffers. May cause high variance in policy updates. |
| **MEDIUM** | last_value always 0.0 | 741 | `update(last_value=0.0)` assumes terminal states. For non-terminal truncation at `max_epochs`, should bootstrap with value estimate. Currently acceptable since all episodes truly terminate. |
| **LOW** | Governor snapshot every 5 epochs | 612-613 | Fixed snapshot interval regardless of loss stability. May miss optimal rollback points. |

### PyTorch Assessment

**Multi-GPU Patterns:**
- CUDA streams per environment (line 319): `torch.cuda.Stream(device=env_device)`
- Round-robin device assignment (line 289): `devices[i % len(devices)]`
- Non-blocking data transfers (lines 370-371): `inputs.to(env_dev, non_blocking=True)`
- Stream synchronization before CPU-GPU sync (lines 524-527, 561-564)

**Memory Patterns:**
- GPU-side accumulation to avoid per-batch `.item()` calls (lines 488-490, 534-536)
- Async gradient collection with deferred materialization (lines 417-418, 577-578)
- Tensors kept on device until epoch-end sync

**CUDA Stream Usage:**
- Proper stream context managers via `torch.cuda.stream()` (lines 366-368, 441-442)
- Stream synchronization before accessing tensor values (lines 524-527)

**Issues:**

| Severity | Issue | Line | Description |
|----------|-------|------|-------------|
| **CRITICAL** | Race condition in gradient accumulation | 518-522 | `train_loss_accum[i].add_(loss_tensor)` is performed inside a stream context, but the tensor was created in a previous stream context block (line 512). If multiple environments share the same GPU device, different streams may have overlapping writes to `train_loss_accum`. The accumulator tensors need proper stream ordering guarantees. |
| **HIGH** | Stream context not covering accumulation | 507-522 | The `process_train_batch` returns tensors that were computed in one stream context. The subsequent `with stream_ctx:` block for accumulation creates a new context that may not have proper dependency on the prior computation. Need `stream.wait_stream()` or record/wait events. |
| **MEDIUM** | Observation normalizer not on device | 233, 249-251 | `RunningMeanStd` initialized without device, then manually moved via `obs_normalizer.to(device)` only when resuming. May cause CPU-GPU transfers during normalization. |
| **MEDIUM** | DataLoader iterator recreation every epoch | 495, 538 | Creates new iterators `train_iters = [iter(env_state.trainloader) ...]` each epoch. This is correct but prevents using persistent_workers effectively since iterator reset triggers worker respawn. |
| **LOW** | Mixed precision not utilized | - | Training uses full FP32. Could benefit from `torch.amp.autocast()` for memory and throughput gains, especially on multi-GPU. |
| **LOW** | No gradient accumulation | - | With batch_size=512 per env, may hit OOM on smaller GPUs. Consider micro-batching with gradient accumulation. |

---

## 3. gradient_collector.py

### Purpose
Lightweight gradient statistics collector for seed telemetry. Provides vectorized gradient norm computation using `torch._foreach_norm` for efficient multi-parameter analysis without the overhead of hook-based trackers.

### Key Classes/Functions

| Name | Lines | Description |
|------|-------|-------------|
| `SeedGradientCollector` | 19-105 | Stateless collector with configurable vanishing/exploding thresholds |
| `SeedGradientCollector.collect()` | 43-57 | Synchronous collection (calls `.item()`) |
| `SeedGradientCollector.collect_async()` | 59-105 | Async-safe collection returning tensors |
| `materialize_grad_stats()` | 108-151 | Convert async tensor stats to Python values after stream sync |
| `collect_seed_gradients()` | 154-177 | Convenience sync function |
| `collect_seed_gradients_async()` | 179-201 | Convenience async function |

### DRL Assessment

**Gradient Health Metrics:**
- Vanishing detection threshold: 1e-7 (reasonable for FP32)
- Exploding detection threshold: 100.0 (conservative)
- Health score formula penalizes exploding more heavily (0.8 vs 0.5 for vanishing)

**Issues:**

| Severity | Issue | Line | Description |
|----------|-------|------|-------------|
| **LOW** | Per-parameter norm averaging | 135 | `gradient_norm = (total_squared_norm ** 0.5) / n_grads` computes average L2 norm per parameter, not the global gradient norm. This is unconventional but consistent with per-layer health analysis. |

### PyTorch Assessment

**Vectorized Operations:**
- Uses `torch._foreach_norm()` (line 86) for batch L2 norm computation - efficient for many small tensors
- Stacks norms for GPU-side aggregation (line 89)

**Async Patterns:**
- Clean separation between tensor computation and materialization
- Explicit documentation of sync requirements
- Returns sentinel dict for empty gradient case (lines 74-81)

**Issues:**

| Severity | Issue | Line | Description |
|----------|-------|------|-------------|
| **MEDIUM** | `torch._foreach_norm` is private API | 86 | Uses `torch._foreach_norm()` which is a private/internal function (underscore prefix). May break between PyTorch versions. Consider using `torch.linalg.vector_norm` with manual iteration or checking for `torch.compile` friendly alternatives. |
| **LOW** | Iterator exhaustion | 72 | `parameters` iterator consumed on first call; calling `collect()` twice on same iterator yields empty. Documented behavior but could cause silent bugs. |

---

## Summary of Issues by Severity

### CRITICAL (1)

1. **vectorized.py:518-522** - CUDA stream race condition in gradient accumulation. Multiple environments on the same device may have concurrent writes to accumulator tensors without proper stream synchronization.

### HIGH (2)

1. **vectorized.py:507-522** - Stream context discontinuity between batch processing and accumulation. Missing stream dependency tracking.

2. **vectorized.py:725-732** - Normalized vs unnormalized state mismatch between action selection and buffer storage. This will cause the PPO ratio computation to use mismatched log probabilities.

### MEDIUM (7)

1. **training.py:120-234** - Redundant code paths for different seed stages
2. **training.py:136-139** - Seed optimizer momentum reset on each germination
3. **vectorized.py:741** - Small rollout buffer per PPO update
4. **vectorized.py:233** - Observation normalizer device placement
5. **vectorized.py:495** - DataLoader iterator recreation prevents persistent_workers benefit
6. **gradient_collector.py:86** - Use of private PyTorch API `torch._foreach_norm`
7. **vectorized.py:741** - last_value=0.0 assumption

### LOW (7)

1. **training.py:110-352** - No early stopping
2. **training.py:150** - Sync gradient collection
3. **training.py:245** - `no_grad` vs `inference_mode`
4. **vectorized.py:612-613** - Fixed snapshot interval
5. **vectorized.py** - No mixed precision
6. **vectorized.py** - No gradient accumulation
7. **gradient_collector.py:72** - Iterator exhaustion documentation

---

## Recommendations

### Immediate (CRITICAL/HIGH)

1. **Fix CUDA stream synchronization in vectorized.py:**
```python
# After process_train_batch returns, ensure accumulation waits for batch computation
for i, env_state in enumerate(env_states):
    if env_batches[i] is None:
        continue
    # Process batch returns tensors computed in env's stream
    loss_tensor, correct_tensor, total, grad_stats = process_train_batch(...)

    # Accumulate in same stream context that produced the tensors
    # Or use explicit event synchronization:
    # env_state.stream.wait_stream(compute_stream)
    with torch.cuda.stream(env_state.stream):
        train_loss_accum[i].add_(loss_tensor)
        train_correct_accum[i].add_(correct_tensor)
```

2. **Fix state normalization consistency:**
```python
# Normalize BEFORE storing, not just for action selection
states_batch = torch.tensor(all_features, dtype=torch.float32, device=device)
obs_normalizer.update(states_batch)
normalized_states = obs_normalizer.normalize(states_batch)

# Store normalized state
agent.store_transition(
    normalized_states[env_idx].cpu(),  # Use normalized version
    action_idx,
    log_prob,
    value,
    reward,
    done
)
```

### Short-term (MEDIUM)

1. **Consolidate stage handling in training.py:**
   - Extract common optimizer stepping into helper function
   - Reduce 6 branches to 3: no-seed, active-seed, fossilized

2. **Replace private API in gradient_collector.py:**
```python
# Option A: Use public API with manual loop
per_param_norms = [torch.linalg.vector_norm(g, ord=2) for g in grads]

# Option B: Use torch.compile for fusion (PyTorch 2.0+)
@torch.compile
def compute_grad_norms(grads):
    return torch.stack([torch.linalg.vector_norm(g, 2) for g in grads])
```

3. **Move observation normalizer to device on init:**
```python
obs_normalizer = RunningMeanStd((state_dim,), device=device)
```

### Long-term

1. **Add mixed precision support** for vectorized training:
```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

2. **Consider torch.compile for episode runner:**
```python
@torch.compile(mode="reduce-overhead")
def process_train_batch(...):
    ...
```
   Note: Requires careful handling of dynamic control flow from seed stage transitions.

3. **Implement persistent DataLoader iterators** to leverage `persistent_workers=True` across epochs.
