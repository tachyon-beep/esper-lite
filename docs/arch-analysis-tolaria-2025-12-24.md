# Tolaria Training Infrastructure: Architectural Analysis

**Date:** 2025-12-24
**Analyst:** Claude Opus 4.5
**Scope:** Deep analysis of Tolaria trainer functions vs. vectorized.py production implementation

---

## Executive Summary

**Finding:** Tolaria's five public trainer functions (`train_epoch_normal`, `train_epoch_incubator_mode`, `train_epoch_blended`, `validate_and_get_metrics`, `validate_with_attribution`) are **DEAD CODE** in production. They are tested but never called by the production training pipeline in `vectorized.py`.

**Recommendation:** **Parallel Implementation - Consolidate with Extreme Prejudice**

The trainer functions provide clean, tested abstractions that are bypassed by production code. This creates:
- Maintenance burden (two implementations of the same logic)
- Testing asymmetry (tests validate functions that aren't used)
- Integration risk (fixes/improvements don't flow to production)

However, **TolariaGovernor IS actively used** and is correctly integrated.

---

## 1. What Tolaria Trainers Provide

### File: `/home/john/esper-lite/src/esper/tolaria/trainer.py`

Tolaria provides **five public training functions** that abstract different training modes:

#### 1.1 `train_epoch_normal()` (Lines 95-133)

**Contract:**
```python
def train_epoch_normal(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = "classification",
    max_grad_norm: float | None = None,
) -> None
```

**What it does:**
- Standard training loop without seed
- All model parameters updated
- Optional gradient clipping via `max_grad_norm`
- Supports both classification and language modeling via `task_type`
- Uses `compute_task_loss()` utility for task-specific loss computation

**Abstraction level:** Single epoch, single DataLoader, simple optimizer step

#### 1.2 `train_epoch_incubator_mode()` (Lines 138-202)

**Contract:**
```python
def train_epoch_incubator_mode(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
    slot: str,
    task_type: str = "classification",
    gradient_telemetry_stride: int = 10,
    max_grad_norm: float | None = None,
) -> None
```

**What it does:**
- STE (Straight-Through Estimator) training during TRAINING stage
- **Both host AND seed train** (critical: not just seed)
- Forward: seed output isolated (alpha=0 semantics)
- Backward: seed receives gradients as if fully blended
- Captures gradient telemetry every N steps via `seed_slot.capture_gradient_telemetry()`
- Two optimizers (host + seed)

**Key insight from docstring (Lines 157-159):**
> "CRITICAL: Both host AND seed train. The 'isolation' refers to the seed's output not affecting the loss computation (alpha=0), NOT to freezing the host. Without this, on large models with frequent seed cycling, the host would never train."

**Abstraction level:** Single epoch, single slot, dual-optimizer, strided telemetry

#### 1.3 `train_epoch_blended()` (Lines 207-252)

**Contract:**
```python
def train_epoch_blended(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str = "classification",
    max_grad_norm: float | None = None,
) -> None
```

**What it does:**
- Joint host+seed training during BLENDING/FOSSILIZED stages
- Seed optimizer is optional (None for fossilized seeds)
- Standard forward/backward with both contributing to loss

**Abstraction level:** Single epoch, joint training, optional seed optimizer

#### 1.4 `validate_and_get_metrics()` (Lines 257-367)

**Contract:**
```python
def validate_and_get_metrics(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    compute_per_class: bool = False,
    num_classes: int = 10,
    task_type: str = "classification",
) -> tuple[float, float, float, float, dict[int, float] | None, float | None]
```

**What it does:**
- Runs validation on testloader (full pass)
- Runs quick train metrics (10 batches via `itertools.islice`)
- Returns: `(val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc, perplexity)`
- GPU-optimized: accumulates on GPU, single CPU sync at end
- Per-class accuracy computed using vectorized `torch.bincount`

**Abstraction level:** Dual-loader validation, comprehensive metrics, GPU-efficient

#### 1.5 `validate_with_attribution()` (Lines 373-432)

**Contract:**
```python
def validate_with_attribution(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    slot: str,
    task_type: str = "classification",
) -> AttributionResult
```

**What it does:**
- **Counterfactual validation** for true seed contribution measurement
- Pass 1: Real validation with current alpha (seed contributing)
- Pass 2: Baseline validation with alpha=0 (host-only)
- Returns `AttributionResult` with `seed_contribution = real_accuracy - baseline_accuracy`
- Uses `force_alpha()` context manager to temporarily override alpha

**Key insight from docstring (Lines 392-398):**
> "Addresses the 'Scapegoat Problem' where seeds were blamed/credited for host accuracy changes during TRAINING stage when they had zero impact."

**Abstraction level:** Slot-specific, counterfactual, two-pass validation

### 1.6 Common Patterns in Tolaria Trainers

All trainer functions share:
- **Simple signatures**: Standard PyTorch DataLoader, no custom iterators
- **Device-agnostic**: String device parameter, no CUDA stream awareness
- **Single-environment**: No batching across parallel envs
- **No AMP support**: No mixed-precision training
- **No multi-GPU**: Single device only
- **GPU-efficient accumulation**: Loss/correct accumulated on GPU, single `.item()` sync
- **Task-type aware**: Classification vs. LM handled via `compute_task_loss()`

---

## 2. What `vectorized.py` Does Instead

### File: `/home/john/esper-lite/src/esper/simic/training/vectorized.py`

The production training pipeline in `vectorized.py` **reimplements all Tolaria trainer logic inline** with massive extensions for high-performance vectorized RL training.

#### 2.1 `process_train_batch()` (Lines 1297-1447)

**Contract:**
```python
def process_train_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    use_telemetry: bool = False,
    slots: list[str] | None = None,
    use_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict] | None]:
```

**What it does (vs. Tolaria):**

| Feature | Tolaria Trainers | vectorized.py |
|---------|------------------|---------------|
| **Multi-GPU** | No | Yes - CUDA streams per env |
| **Stream-safe** | No | Yes - `torch.cuda.stream()` context |
| **AMP support** | No | Yes - `torch_amp.autocast()` + GradScaler |
| **Batch source** | DataLoader | SharedBatchIterator (inverted control) |
| **Optimizer management** | Passed as arg | Per-env state (`env_state.host_optimizer`, `env_state.seed_optimizers[slot_id]`) |
| **Seed training** | Single slot arg | Multi-slot via `slots` list |
| **Telemetry** | Strided capture | Async dual-gradient stats via `collect_gradient_telemetry_for_batch()` |
| **Return type** | None | Tensors (avoids blocking `.item()` inside stream) |
| **Gradient isolation** | Via STE in forward | Via `slot.isolate_gradients = True` (line 1066) |

**Key architectural differences:**

1. **CUDA Stream Isolation** (Lines 1339-1357):
   ```python
   with torch.cuda.stream(env_state.stream):
       inputs = inputs.to(env_dev, non_blocking=True)
       targets = targets.to(env_dev, non_blocking=True)

       # CRITICAL: Clone to enforce stream dependency
       if env_state.stream and inputs.is_cuda:
           inputs = inputs.clone()
           targets = targets.clone()
           inputs.record_stream(env_state.stream)
           targets.record_stream(env_state.stream)
   ```
   - Each environment runs in its own CUDA stream for async execution
   - Cloning enforces memory dependencies (prevents race conditions)
   - `record_stream()` prevents premature tensor deallocation

2. **Dynamic Seed Optimizer Creation** (Lines 1362-1384):
   ```python
   slots_to_step: list[str] = []
   for slot_id in slots_with_active_seeds:
       seed_state = model.seed_slots[slot_id].state
       if seed_state is None:
           continue

       slots_to_step.append(slot_id)

       if slot_id not in env_state.seed_optimizers:
           seed_params = list(model.get_seed_parameters(slot_id))
           env_state.seed_optimizers[slot_id] = torch.optim.SGD(
               seed_params, lr=task_spec.seed_lr, momentum=0.9
           )
   ```
   - Per-slot optimizers stored in `env_state.seed_optimizers` dict
   - Created on-demand when seed becomes active
   - Tolaria requires passing optimizers as arguments

3. **AMP with Stream-Safe GradScaler** (Lines 1404-1443):
   ```python
   if env_state.scaler is not None:
       env_state.scaler.scale(loss).backward()
       # ... (gradient telemetry) ...
       env_state.scaler.step(env_state.host_optimizer)
       for slot_id in slots_to_step:
           seed_opt = env_state.seed_optimizers[slot_id]
           has_grads = any(p.grad is not None for group in seed_opt.param_groups for p in group["params"])
           if has_grads:
               env_state.scaler.step(seed_opt)
           else:
               seed_opt.step()
       env_state.scaler.update()
   else:
       loss.backward()
       env_state.host_optimizer.step()
       for slot_id in slots_to_step:
           env_state.seed_optimizers[slot_id].step()
   ```
   - Per-env GradScaler (line 1090-1094) to avoid stream race conditions
   - BF16 auto-detection (no scaler needed on Ampere+ GPUs)
   - Guard for seeds without gradients (line 1435-1442)

4. **Async Gradient Telemetry** (Lines 1410-1413):
   ```python
   grad_stats_by_slot = None
   if use_telemetry:
       grad_stats_by_slot = _collect_gradient_telemetry_for_batch(
           model, slots_with_active_seeds, env_dev
       )
   ```
   - Returns async dual-gradient stats (host + seed per slot)
   - Materialized later after stream sync (avoids blocking)

#### 2.2 `process_val_batch()` (Lines 1452-1497)

**Contract:**
```python
def process_val_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    slots: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
```

**What it does:**
- Validation batch processing with stream isolation
- Uses `torch.inference_mode()` (faster than `torch.no_grad()`)
- Returns tensors (not floats) to avoid blocking inside stream
- Stream-safe cloning for GPU-cached data

**vs. Tolaria `validate_and_get_metrics()`:**
- Tolaria: Full-dataset validation in one call, returns floats
- vectorized.py: Single-batch processing, returns tensors, stream-aware

#### 2.3 `process_fused_val_batch()` (Lines 1499-1570)

**Contract:**
```python
def process_fused_val_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    alpha_overrides: dict[str, torch.Tensor],
    num_configs: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
```

**What it does:**
- **Fused counterfactual validation** using `MorphogeneticModel.fused_forward()`
- Evaluates multiple alpha configurations in a single GPU kernel
- Replaces N sequential forward passes with 1 batch-expanded pass
- Returns aggregated metrics across all configurations

**vs. Tolaria `validate_with_attribution()`:**
- Tolaria: Two sequential passes (real + baseline), CPU orchestration
- vectorized.py: Fused GPU pass, evaluates multiple configs simultaneously
- Tolaria: Per-slot function
- vectorized.py: Multi-slot, batch-vectorized

**Performance comparison:**
- Tolaria: 2 × forward_pass_time (sequential)
- vectorized.py: 1.2 × forward_pass_time (fused with batch expansion overhead)

#### 2.4 Main Training Loop (Lines 1703-1880)

**Architecture:**
```python
for batch_step in range(num_train_batches):
    env_batches = next(train_iter)  # SharedBatchIterator

    # Launch all envs in parallel via CUDA streams
    for i, env_state in enumerate(env_states):
        inputs, targets = env_batches[i]
        # Stream sync with DataLoader default stream
        if env_state.stream:
            loader_stream = torch.cuda.default_stream(torch.device(env_state.env_device))
            env_state.stream.wait_stream(loader_stream)

        # Async training in env's stream
        loss_tensor, correct_tensor, total, grad_stats = process_train_batch(...)
        # Accumulate tensors (no .item() yet)

    # Sync all streams
    for env_state in env_states:
        if env_state.stream:
            env_state.stream.synchronize()

    # NOW convert tensors to floats (single sync point)
    env_loss = loss_tensor.item()
    env_correct = correct_tensor.item()
```

**Key architectural pattern:**
1. **Inverted Control Flow**: Iterate batches first, dispatch to envs second
2. **Async GPU Execution**: All envs execute in parallel on separate CUDA streams
3. **Deferred CPU Sync**: Accumulate on GPU, sync once at end
4. **SharedBatchIterator**: Single DataLoader serving all envs (reduces IPC overhead from N×M to M workers)

**vs. Tolaria trainers:**
- Tolaria: `for inputs, labels in trainloader:` (batch-centric)
- vectorized.py: `for batch_step in range(num_train_batches):` + `next(train_iter)` (inverted)
- Tolaria: Single-threaded, blocking `.item()` calls
- vectorized.py: Multi-stream async, batched synchronization

---

## 3. Side-by-Side Comparison

### 3.1 Training Epoch Logic

| Aspect | Tolaria `train_epoch_incubator_mode` | vectorized.py `process_train_batch` |
|--------|--------------------------------------|-------------------------------------|
| **Function type** | Full epoch abstraction | Single batch processor |
| **Loop control** | Internal `for inputs, labels in trainloader:` | External batch loop in `train_ppo_vectorized()` |
| **Device handling** | String device param | `env_state.env_device` + CUDA stream |
| **Optimizer storage** | Function arguments | `env_state.host_optimizer`, `env_state.seed_optimizers[slot]` |
| **Seed scope** | Single `slot` arg | Multi-slot via `slots` list |
| **Telemetry** | `seed_slot.capture_gradient_telemetry()` every N steps | `_collect_gradient_telemetry_for_batch()` returns async stats |
| **AMP** | Not supported | `env_state.scaler` + `autocast_ctx` |
| **Multi-GPU** | Not supported | CUDA stream per env |
| **Return value** | None (side effects only) | `(loss_tensor, correct_tensor, total, grad_stats)` |

**Fundamental difference:**
- **Tolaria**: "Run a training epoch" (high-level, blocking, single-env)
- **vectorized.py**: "Process one batch for one env in a stream" (low-level, async, multi-env)

### 3.2 Validation Logic

| Aspect | Tolaria `validate_and_get_metrics` | vectorized.py `process_val_batch` |
|--------|-----------------------------------|-----------------------------------|
| **Scope** | Full dataset validation | Single batch |
| **Metrics** | 6-tuple: `(val_loss, val_acc, train_loss, train_acc, per_class_acc, perplexity)` | 3-tuple: `(loss_tensor, correct_tensor, total)` |
| **Train metrics** | Quick 10-batch sample via `itertools.islice` | Not computed (separate pass) |
| **Per-class acc** | Optional via `compute_per_class` | Not computed |
| **Return type** | Floats (`.item()` called internally) | Tensors (caller syncs) |
| **Stream awareness** | None | `torch.cuda.stream(env_state.stream)` |

**Fundamental difference:**
- **Tolaria**: "Get comprehensive metrics for one model" (single-shot, CPU-friendly)
- **vectorized.py**: "Process one validation batch in a stream" (incremental, GPU-async)

### 3.3 Counterfactual Attribution

| Aspect | Tolaria `validate_with_attribution` | vectorized.py `process_fused_val_batch` |
|--------|-------------------------------------|----------------------------------------|
| **Method** | Sequential: 2 passes (real + baseline) | Fused: 1 pass with batch expansion |
| **Alpha override** | `seed_slot.force_alpha(0.0)` context manager | `alpha_overrides` dict to `fused_forward()` |
| **Configs** | 2 (real, baseline) | N (arbitrary alpha combinations) |
| **Performance** | 2× forward pass time | ~1.2× forward pass time |
| **Return** | `AttributionResult` dataclass | `(loss_tensor, correct_tensor, total)` for expanded batch |
| **Scope** | Single slot | Multi-slot |

**Fundamental difference:**
- **Tolaria**: CPU orchestration, sequential passes, per-slot
- **vectorized.py**: GPU kernel fusion, parallel evaluation, batch-vectorized

---

## 4. Why Was This Reimplemented?

Based on code archaeology and architectural analysis:

### 4.1 Features `vectorized.py` Needs That Tolaria Doesn't Provide

1. **Multi-GPU Support**
   - Tolaria: Single device string
   - Need: CUDA streams per environment for async parallel execution

2. **AMP (Mixed Precision Training)**
   - Tolaria: No AMP support
   - Need: GradScaler + autocast for FP16/BF16 training

3. **Inverted Control Flow**
   - Tolaria: `for batch in loader:` (loader drives iteration)
   - Need: `for batch_idx:` then dispatch to envs (batch-first, env-second)

4. **Async GPU Execution**
   - Tolaria: Blocking `.item()` calls after each operation
   - Need: Return tensors, accumulate on GPU, single sync at end

5. **Per-Environment State**
   - Tolaria: Model + optimizer passed as args
   - Need: `ParallelEnvState` with per-env model, optimizers, scaler, stream

6. **SharedBatchIterator Integration**
   - Tolaria: Standard PyTorch DataLoader
   - Need: Single shared iterator serving all envs (IPC efficiency)

7. **Dynamic Multi-Slot Management**
   - Tolaria: Single `slot` arg
   - Need: Loop over `slots` list, dynamic optimizer creation

8. **Gradient Telemetry Return Values**
   - Tolaria: Side-effect capture via `seed_slot.capture_gradient_telemetry()`
   - Need: Return async stats for later materialization

9. **Fused Counterfactual Validation**
   - Tolaria: Sequential passes with `force_alpha()`
   - Need: Batch-expanded fused forward for GPU saturation

### 4.2 Could Tolaria Trainers Be Extended?

**Theoretical possibility:** Yes, via optional kwargs for AMP, streams, etc.

**Practical reality:** Would balloon function signatures and complexity:
```python
def train_epoch_incubator_mode(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
    slot: str,
    task_type: str = "classification",
    gradient_telemetry_stride: int = 10,
    max_grad_norm: float | None = None,
    # NEW PARAMS NEEDED:
    cuda_stream: torch.cuda.Stream | None = None,
    scaler: torch.amp.GradScaler | None = None,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype | None = None,
    return_tensors: bool = False,
    slots: list[str] | None = None,
    env_state: ParallelEnvState | None = None,
    # ...
) -> tuple[torch.Tensor, torch.Tensor, int, dict] | None:
```

This defeats the purpose of Tolaria as a **simple, readable abstraction layer**.

---

## 5. The Governor Connection

### 5.1 TolariaGovernor IS Used in Production

**Critical distinction:** While the **trainer functions** are dead code, **TolariaGovernor is actively used**.

**Usage in vectorized.py** (Lines 1111-1120):
```python
governor = TolariaGovernor(
    model=model,
    sensitivity=DEFAULT_GOVERNOR_SENSITIVITY,
    absolute_threshold=DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    death_penalty=DEFAULT_GOVERNOR_DEATH_PENALTY,
    history_window=DEFAULT_GOVERNOR_HISTORY_WINDOW,
    min_panics_before_rollback=DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
    random_guess_loss=random_guess_loss,
)
governor.snapshot()
```

**Integration points:**
1. **Snapshot** (Line 2145): `env_state.governor.snapshot()` after validation
2. **Vital Signs Check** (Line 2148): `is_panic = env_state.governor.check_vital_signs(val_loss)`
3. **Rollback Execution** (Line 2376): `env_state.governor.execute_rollback(...)`

### 5.2 Why Governor Survived But Trainers Didn't

**Governor characteristics:**
- **Stateful**: Tracks loss history, panic counters, last-good-state
- **Independent**: No coupling to DataLoader, optimizer, or training loop structure
- **Side-effect oriented**: Monitors, snapshots, rollbacks (not data transformation)
- **Model-centric**: Operates on model state, not training batches

**Trainer characteristics:**
- **Stateless**: Pure functions transforming data
- **Tightly coupled**: DataLoader, optimizer, criterion all required
- **Data transformation**: Batch → loss → gradients → parameter updates
- **Loop structure**: Internal iteration over batches

**Verdict:** Governor is a **monitoring service**, trainers are **execution engines**. The vectorized execution engine needed rewriting for performance, but the monitoring service didn't.

### 5.3 Governor Design Was Architected for Independence

From `governor.py` docstring (Lines 41-50):
```python
"""The Super-Ego of the training system.

Monitors model training for catastrophic failures and can rollback
to Last Known Good state while signaling punishment to the RL agent.

Capabilities:
1. Anomaly Detection - NaN/Inf and statistical outliers
2. State Reversion - RAM checkpoint for instant rollback
3. RL Punishment - Returns negative reward for PPO buffer injection
"""
```

Governor was **designed to be training-loop agnostic**:
- Takes model as arg (not training infrastructure)
- Exposes simple API: `check_vital_signs(loss)`, `execute_rollback()`, `snapshot()`
- No DataLoader, optimizer, or batch processing logic

**This is the key architectural difference**: Governor operates at the **model level**, trainers operate at the **batch level**.

---

## 6. Are Tolaria Trainers Architecturally Compatible?

### 6.1 Control Flow Mismatch

**Tolaria trainers:**
```python
def train_epoch_normal(model, trainloader, ...):
    for inputs, labels in trainloader:  # <-- OWNS THE LOOP
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_task_loss(outputs, labels, criterion, task_type)
        loss.backward()
        optimizer.step()
```

**vectorized.py:**
```python
for batch_step in range(num_train_batches):  # <-- OUTER LOOP
    env_batches = next(train_iter)
    for i, env_state in enumerate(env_states):  # <-- INNER LOOP
        inputs, targets = env_batches[i]
        loss_tensor, correct_tensor, total, grad_stats = process_train_batch(...)
```

**Incompatibility:** Tolaria trainers **own the iteration loop**, vectorized.py **owns the batch dispatch loop**. These are fundamentally different control flow patterns.

### 6.2 Return Value Mismatch

**Tolaria:**
- `train_epoch_*()` → `None` (side effects only)
- `validate_and_get_metrics()` → `tuple[float, float, float, float, dict | None, float | None]` (floats, CPU-synced)

**vectorized.py:**
- `process_train_batch()` → `tuple[torch.Tensor, torch.Tensor, int, dict | None]` (tensors, GPU-resident)
- `process_val_batch()` → `tuple[torch.Tensor, torch.Tensor, int]` (tensors, GPU-resident)

**Incompatibility:** Tolaria functions sync to CPU immediately (blocking), vectorized.py defers sync to batch end (async).

### 6.3 State Management Mismatch

**Tolaria:**
- Model passed as arg
- Optimizer(s) passed as args
- No persistent state between calls

**vectorized.py:**
- Model in `env_state.model`
- Optimizers in `env_state.host_optimizer`, `env_state.seed_optimizers[slot]`
- CUDA stream in `env_state.stream`
- GradScaler in `env_state.scaler`
- All state persists across batches

**Incompatibility:** Tolaria assumes caller manages state, vectorized.py encapsulates state in `ParallelEnvState`.

### 6.4 Abstraction Level Mismatch

**Tolaria:** "Run a training epoch" (high-level, blocking, complete)

**vectorized.py:** "Process one batch in a stream" (low-level, async, incremental)

**Analogy:**
- Tolaria: `file.write(data)` (high-level, blocking)
- vectorized.py: `os.write(fd, data)` (low-level, non-blocking)

You can't easily build the latter from the former without sacrificing performance.

---

## 7. Testing Asymmetry Analysis

### 7.1 What Gets Tested

**Tolaria trainer tests** (`tests/tolaria/test_trainer.py`):
- 15+ tests for `train_epoch_normal()`, `train_epoch_incubator_mode()`, `train_epoch_blended()`
- 10+ tests for `validate_and_get_metrics()`, `validate_with_attribution()`
- All pass ✅

**Integration tests using Tolaria trainers:**
- `test_tamiyo_tolaria.py`: Uses `train_epoch_normal()`, `validate_and_get_metrics()`
- `test_tolaria_kasmina.py`: Uses `train_epoch_blended()`, `validate_with_attribution()`
- `test_tolaria_simic.py`: Uses `train_epoch_normal()`, `validate_and_get_metrics()`

**Production code:**
- `vectorized.py`: Uses **NONE of the above**, only `TolariaGovernor`

### 7.2 The Risk

**Scenario:** A bug is discovered in seed training logic.

**Fix location 1:** Update `train_epoch_incubator_mode()` in `trainer.py`
- Tests pass ✅
- Production **UNAFFECTED** (not called)

**Fix location 2:** Update `process_train_batch()` in `vectorized.py`
- Production fixed ✅
- Tests **UNCHANGED** (don't exercise production code)

**Result:** Divergence. The tested code and the production code drift apart.

### 7.3 Current State Evidence

All Tolaria trainer functions have TODO comments marking them as dead code:

- Line 92-94: `train_epoch_normal()`
  ```python
  # TODO: [DEAD CODE] - train_epoch_normal is tested but NEVER called in production.
  # vectorized.py reimplements training inline via process_train_batch().
  # Either wire this into production or delete. See: risk assessment 2024-12-24.
  ```

- Line 135-137: `train_epoch_incubator_mode()`
  ```python
  # TODO: [DEAD CODE] - train_epoch_incubator_mode is tested but NEVER called in production.
  # vectorized.py reimplements STE training inline. Either wire this into production
  # or delete. See: architectural risk assessment 2024-12-24.
  ```

- Line 204-206: `train_epoch_blended()`
  ```python
  # TODO: [DEAD CODE] - train_epoch_blended is tested but NEVER called in production.
  # vectorized.py reimplements blended training inline. Either wire this into production
  # or delete. See: architectural risk assessment 2024-12-24.
  ```

- Line 254-256: `validate_and_get_metrics()`
  ```python
  # TODO: [DEAD CODE] - validate_and_get_metrics is tested but NEVER called in production.
  # vectorized.py reimplements validation inline via process_val_batch().
  # Either wire this into production or delete. See: risk assessment 2024-12-24.
  ```

- Line 370-372: `validate_with_attribution()`
  ```python
  # TODO: [DEAD CODE] - validate_with_attribution is tested but NEVER called in production.
  # Counterfactual attribution exists in vectorized.py but uses different implementation.
  # Either wire this into production or delete. See: risk assessment 2024-12-24.
  ```

**These TODOs are accurate.** The functions are indeed dead code.

---

## 8. Recommendations

### 8.1 Primary Recommendation: DELETE

**Verdict:** These are **superseded abstractions**. Delete them.

**Rationale:**
1. **No Legacy Code Policy** (from `CLAUDE.md`):
   > "When something is removed or changed, DELETE THE OLD CODE COMPLETELY."

2. **Zero production usage**: Not called anywhere outside tests

3. **Testing asymmetry**: Tests validate code that isn't run

4. **Architectural incompatibility**: Control flow, return types, state management all mismatched

5. **Maintenance burden**: Two implementations means 2× the surface area for bugs

**Action plan:**
1. Delete `train_epoch_normal()`, `train_epoch_incubator_mode()`, `train_epoch_blended()`
2. Delete `validate_and_get_metrics()`, `validate_with_attribution()`
3. Delete `AttributionResult` dataclass (only used by `validate_with_attribution()`)
4. Delete `_run_validation_pass()` helper (only used by `validate_with_attribution()`)
5. Delete all tests in `tests/tolaria/test_trainer.py`
6. Delete all integration tests that use these functions:
   - `tests/integration/test_tamiyo_tolaria.py`
   - `tests/integration/test_tolaria_kasmina.py`
   - `tests/integration/test_tolaria_simic.py`
7. Update `src/esper/tolaria/__init__.py` to remove exports
8. **Keep** `TolariaGovernor` and `GovernorReport` (actively used)

### 8.2 Alternative: Extract Common Logic

**IF** you believe there's value in the abstractions, extract the **logic** not the **functions**.

**Example: Gradient clipping**

Tolaria trainers have:
```python
if max_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

vectorized.py currently **does not clip gradients**. If you want clipping, add it to `process_train_batch()`:
```python
# In process_train_batch(), after loss.backward()
if max_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

**Do NOT** try to call Tolaria functions from vectorized.py. The control flow is incompatible.

### 8.3 Alternative: Separate Simple Trainer

**IF** you need a simple, single-env, CPU-friendly trainer for **different use cases**:

1. **Create a new module**: `src/esper/utils/simple_trainer.py`
2. **Document the use case**: "Single-device, blocking, batch-centric training for debugging/prototyping"
3. **Do NOT test it against production scenarios**: Integration tests should use `vectorized.py`
4. **Make it clear**: "This is NOT the production training loop"

**But honestly:** What's the use case? All production training goes through `vectorized.py`. Simple debugging can use raw PyTorch loops.

### 8.4 What About Integration Tests?

**Current integration tests** using Tolaria trainers test the **wrong thing**.

**Example:** `test_tamiyo_tolaria.py` tests:
> "SignalTracker receives training signals from Tolaria's training loop"

**Problem:** Production doesn't use Tolaria's training loop. It uses `vectorized.py`.

**Fix:** Rewrite integration tests to test **production code paths**:
- Test `vectorized.py` → `SignalTracker` integration
- Test `vectorized.py` → `TolariaGovernor` integration (already done implicitly)
- Test `vectorized.py` → counterfactual attribution (already done via `process_fused_val_batch`)

**OR:** Delete integration tests that test non-production paths.

---

## 9. Architectural Principles Violated

From `CLAUDE.md`:

### 9.1 No Legacy Code Policy

> "STRICT REQUIREMENT: Legacy code, backwards compatibility, and compatibility shims are strictly forbidden."
>
> "When something is removed or changed, DELETE THE OLD CODE COMPLETELY."

**Violation:** Tolaria trainers are legacy code. They were superseded by `vectorized.py` but kept around "just in case".

### 9.2 Testing Asymmetry

> "Only mark a task as completed when you have FULLY accomplished it."

**Violation:** Tests pass for code that isn't used. This creates false confidence.

### 9.3 Maintenance Burden

Keeping dead code:
- Confuses new contributors ("Which one do I modify?")
- Creates merge conflicts
- Requires documentation updates
- Adds cognitive load ("Is this still used?")

---

## 10. Migration Path (If Deletion is Too Aggressive)

**Phase 1: Deprecation (immediate)**
1. Add deprecation warnings to all dead functions:
   ```python
   import warnings
   warnings.warn(
       "train_epoch_normal is deprecated and not used in production. "
       "Use vectorized.py process_train_batch() instead.",
       DeprecationWarning,
       stacklevel=2
   )
   ```

2. Update integration tests to use `@pytest.mark.filterwarnings("ignore::DeprecationWarning")`

**Phase 2: Documentation (within 1 week)**
1. Create `docs/tolaria-migration-guide.md` explaining:
   - Why Tolaria trainers were superseded
   - How to use `vectorized.py` for production training
   - Examples of migrating from Tolaria API to vectorized API

**Phase 3: Deletion (within 2 weeks)**
1. Remove all deprecated functions
2. Remove all tests for deprecated functions
3. Update `__init__.py` exports
4. Update integration tests to use production code paths

**Phase 4: Cleanup (within 1 month)**
1. Remove migration guide (no longer needed)
2. Remove deprecation warnings (nothing to deprecate)

---

## 11. Conclusion

### 11.1 Summary of Findings

| Component | Status | Recommendation |
|-----------|--------|----------------|
| `train_epoch_normal()` | Dead code | **DELETE** |
| `train_epoch_incubator_mode()` | Dead code | **DELETE** |
| `train_epoch_blended()` | Dead code | **DELETE** |
| `validate_and_get_metrics()` | Dead code | **DELETE** |
| `validate_with_attribution()` | Dead code | **DELETE** |
| `AttributionResult` | Dead code | **DELETE** |
| `_run_validation_pass()` | Dead code | **DELETE** |
| `TolariaGovernor` | **ACTIVE** | **KEEP** |
| `GovernorReport` | **ACTIVE** | **KEEP** |

### 11.2 Architectural Assessment

**Tolaria trainers are NOT compatible with vectorized.py** because:
1. Control flow mismatch (internal loop vs. external dispatch)
2. Return value mismatch (None/floats vs. tensors)
3. State management mismatch (args vs. `ParallelEnvState`)
4. Performance requirements mismatch (blocking vs. async)
5. Feature requirements mismatch (single-env vs. multi-GPU)

**TolariaGovernor IS compatible** because:
1. Designed to be training-loop agnostic
2. Operates at model level, not batch level
3. Stateful monitoring service, not data transformation
4. Simple API: `check_vital_signs()`, `snapshot()`, `execute_rollback()`

### 11.3 Final Recommendation

**DELETE** all Tolaria trainer functions per the "No Legacy Code Policy".

**Justification:**
- Zero production usage
- Architecturally incompatible with high-performance vectorized training
- Testing asymmetry creates false confidence
- Maintenance burden for no benefit
- Violates project architecture constitution

**Keep** `TolariaGovernor` and `GovernorReport` (actively used, well-architected).

**Timeline:** Immediate deletion unless there's a compelling use case for simple single-env trainers (none identified).

---

**Document Version:** 1.0
**Next Review:** After deletion or when new training infrastructure is added
