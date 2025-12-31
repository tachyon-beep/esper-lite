# Hot Path Analysis & Optimization Strategies
**Date:** January 1, 2026
**Author:** PyTorch 2.9 Specialist Agent
**Scope:** `src/esper/simic/training/vectorized.py` and associated subsystems.

## 1. Executive Summary

The current `train_ppo_vectorized` implementation represents a sophisticated "v2.0" vectorized architecture. It correctly employs **Inverted Control Flow** (batch-first iteration) and **CUDA Streams** to maximize GPU utilization during the supervised training phase.

However, the architecture suffers from a classic **"CPU Trench"** phenomenon during the Reinforcement Learning (RL) step phase. While the heavy lifting of gradient descent is parallelized on GPU, the *management* of that training (Lifecycle decisions, Reward computation, Model surgery) drops into a sequential, single-threaded CPU loop for every environment, every epoch.

**Primary Bottleneck:** The sequential `for env_idx in range(n_envs)` loop at the end of each epoch limits the theoretical maximum throughput (EPS) by introducing a strictly serial CPU component that cannot overlap with GPU work (Amdahl's Law).

**Recommendation:** Transition from a "Python-managed Vectorization" to a **"Tensor-Driven State Machine"**. By moving the environment state, reward calculation, and lifecycle logic to GPU tensors, we can achieve fully compiled, graph-captured execution.

---

## 2. Detailed Analysis

### 2.1 The "CPU Trench" (Sequential Action Execution)
**Location:** `src/esper/simic/training/vectorized.py` (Lines ~2420 - 2750)

At the end of every epoch, the code executes this pattern:
```python
actions = policy.get_action(...) # GPU -> CPU Sync
for env_idx in range(n_envs):    # Sequential CPU Execution
    # 1. Parse Action (CPU Logic)
    # 2. Compute Reward (CPU Math via `rewards` module)
    # 3. Execute Lifecycle Op (Model Surgery: Germinate/Prune)
    # 4. Update Telemetry (Dict manipulation)
```
**Impact:**
*   **Serialization:** If reward computation takes 1ms and we have 16 envs, that's 16ms of dead GPU time per step.
*   **GIL Contention:** Python object creation (dictionaries for telemetry) fights for the GIL.
*   **Cache Thrashing:** Jumping between env states on CPU evicts L1/L2 cache lines.

### 2.2 The "Sync Wall" (Policy Inference)
**Location:** `vectorized.py` (Lines ~2340)
```python
actions_cpu = {key: actions_dict[key].cpu().numpy() for key in HEAD_NAMES}
values_cpu = values_tensor.cpu()
```
**Impact:**
*   Forces a hard synchronization between the Policy GPU and the CPU.
*   Prevents the CPU from queuing up the *next* batch of training data while the GPU is finishing the current one.
*   While necessary for the current CPU-based logic, it is the barrier preventing fully asynchronous execution.

### 2.3 Dynamic Graph Breaks (Model Surgery)
**Location:** `MorphogeneticModel.germinate_seed` / `prune_seed`
**Impact:**
*   Adding/Removing `nn.Parameter` objects invalidates any captured CUDA Graphs.
*   Forces PyTorch Autograd to rebuild the graph structure on every topology change.
*   Prevents `torch.compile` from effectively optimizing the `forward` pass across episode boundaries.

---

## 3. Optimization Strategies (PyTorch 2.9 Paradigm)

### Strategy A: Tensor-Driven Reward Kernel (Immediate Win)
**Goal:** Eliminate the sequential CPU reward loop.

**Implementation:**
1.  **Vectorize Inputs:** Instead of passing `SeedInfo` objects, pass Struct-of-Arrays (SoA) tensors: `val_accs[n_envs]`, `param_counts[n_envs]`, `ages[n_envs]`.
2.  **JIT Compile:** Rewrite `compute_reward` as a pure PyTorch function decorated with `@torch.compile(mode="max-autotune")`.
3.  **Batch Execution:** Compute rewards for ALL environments in a single GPU kernel launch.
    ```python
    # Shape: [n_envs]
    rewards = compiled_reward_kernel(actions_batch, metrics_batch, ...)
    ```
4.  **Result:** Reduces O(N) CPU time to O(1) GPU launch latency.

### Strategy B: Static Graph Topology (The "Sleeper" Pattern)
**Goal:** Enable CUDA Graphs and full `torch.compile` by removing dynamic model surgery.

**Implementation:**
1.  **Pre-allocation:** Instead of `ModuleDict` growing/shrinking, initialize the `MorphogeneticModel` with a *fixed pool* of "Sleeper Seeds" (e.g., 4 slots per segment).
2.  **Masked Execution:**
    *   **Active Seed:** `mask = 1.0`. Gradients flow, weights update.
    *   **Pruned/Empty:** `mask = 0.0`. Gradients zeroed, output ignored.
    *   **Germination:** Reset weights of a "Sleeper" seed in-place and set `mask = 1.0`.
3.  **Benefit:** The computational graph becomes **static**.
    *   Allows `torch.compile(mode="reduce-overhead")` on the entire training loop.
    *   Enables **CUDA Graph Capture** of the full epoch, including the "virtual" germination/pruning.

### Strategy C: Asynchronous Telemetry Stream
**Goal:** Remove telemetry from the critical path.

**Implementation:**
1.  **Dedicated Stream:** Create a `telemetry_stream = torch.cuda.Stream()`.
2.  **Async Collection:** Issue gradient norm/stat collection kernels into `telemetry_stream`.
3.  **Lazy Retrieval:** Do not `cpu()` the telemetry stats immediately. Let them accumulate in a GPU buffer and fetch them asynchronously or only when needed (e.g., for logging, not every step).

### Strategy D: Consolidated Host-Device Transfers
**Goal:** Minimize PCIe latency.

**Implementation:**
1.  **Stacked Transfers:** As noted in the code comments, stack all action heads into a single tensor `[n_envs, n_heads]` before moving to CPU.
2.  **Pinned Memory Ring Buffer:** If moving to CPU is strictly necessary (for complex logic), use a pre-allocated `pin_memory` buffer to overlap transfer with computation.

## 4. Implementation Plan (Ranked by ROI)

1.  **Refactor `process_train_batch` Loop:** (Medium Effort, High Impact)
    *   Currently, it launches per-env streams but syncs at epoch end.
    *   Ensure `germinate`/`prune` logic (even if CPU based) issues its GPU init work into the *correct* stream.
    *   **Current state check:** The code does `env_state.stream.wait_stream` correctly, but `germinate` calls `slot_obj.germinate`. If `germinate` uses `torch.zeros(..., device=d)`, it runs on the default stream. **Action:** Wrap lifecycle ops in `with torch.cuda.stream(env_state.stream):`.

2.  **Vectorize Reward Calculation:** (High Effort, Medium Impact)
    *   Convert `src/esper/simic/rewards` to operate on Batched Tensors.
    *   This is a prerequisite for Strategy A.

3.  **"Sleeper" Seed Architecture:** (Extreme Effort, Paradigm Shift)
    *   This is the "PyTorch 2.9" endgame. It requires rewriting `MorphogeneticModel` to be static.
    *   **Recommendation:** Prototype this on a small scale (e.g. `VectorizedSleeperModel`) to verify graph capture benefits before full refactor.

## 5. Specific Code Recommendations

**`src/esper/simic/training/vectorized.py`**

**Fix 1: Stream-safe Lifecycle Ops**
```python
# Around line 2560
with torch.cuda.stream(env_state.stream):
    model.germinate_seed(...) 
    # Ensure germinate_seed uses 'device' kwarg and doesn't implicitly use default stream
```

**Fix 2: Stacked Action Transfer**
```python
# Around line 2340
# Stack on GPU first
all_heads = torch.stack([actions_dict[k] for k in _HEAD_NAMES_FOR_TELEM]) # [8, n_envs]
# Single Transfer
all_heads_cpu = all_heads.cpu().numpy()
# Unpack on CPU (fast)
actions_cpu = {k: all_heads_cpu[i] for i, k in enumerate(_HEAD_NAMES_FOR_TELEM)}
```

**Fix 3: Disable "Device-Side Assert" Debugging in Prod**
The comment mentions `_compiled_loss_and_correct` is unstable. This is often due to `torch.compile` struggling with dynamic shapes or specific CUDA kernels.
*   **Action:** Try `torch.compile(..., dynamic=False)` if batch size is fixed.
*   **Action:** Verify inputs to `CrossEntropyLoss` are within bounds (0 <= target < C). The profile script checks this, but compiling the check itself is hard.
