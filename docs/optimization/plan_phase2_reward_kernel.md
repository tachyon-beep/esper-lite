# Optimization Plan: Phase 2 - Tensor-Driven Reward Kernel
**Date:** January 1, 2026
**Author:** PyTorch 2.9 Specialist Agent
**Goal:** Replace the sequential, CPU-based `compute_contribution_reward` with a JIT-compiled PyTorch kernel (`compute_reward_batched`) that operates on Struct-of-Arrays (SoA) tensors.

## 1. Concept

The current `compute_reward` function contains complex branching logic (if/else), enum handling, and math operations. PyTorch 2.0+ `torch.compile` handles branching well *if* the inputs are tensors and the branches can be lowered to masked operations (e.g., `torch.where`).

**Key Transformation:**
*   **Scalar -> Tensor:** Process $N$ environments in parallel.
*   **Enum -> Int:** `LifecycleOp` and `SeedStage` become `torch.long` tensors.
*   **Objects -> SoA:** `SeedInfo` becomes a collection of tensors (e.g., `seed_ages`, `total_improvements`).
*   **Branching -> Masking:** `if action == GERMINATE` becomes `mask = (actions == OP_GERMINATE); rewards = torch.where(mask, ...)`

## 2. Kernel Interface

The new kernel will live in `src/esper/simic/rewards/compiled_rewards.py` (new file).

```python
import torch

@torch.compile(mode="max-autotune")
def compute_reward_batched(
    # --- Actions ---
    actions: torch.Tensor,              # [N] int64 (LifecycleOp)
    
    # --- Signals ---
    seed_contributions: torch.Tensor,   # [N] float32 (NaN if None)
    val_accs: torch.Tensor,             # [N] float32
    acc_deltas: torch.Tensor,           # [N] float32
    
    # --- Seed State (SoA) ---
    has_seed_mask: torch.Tensor,        # [N] bool
    stages: torch.Tensor,               # [N] int64 (SeedStage)
    total_improvements: torch.Tensor,   # [N] float32
    epochs_in_stage: torch.Tensor,      # [N] int64
    seed_ages: torch.Tensor,            # [N] int64
    previous_stages: torch.Tensor,      # [N] int64
    previous_epochs: torch.Tensor,      # [N] int64
    
    # --- Context ---
    epoch: int,                         # Scalar (constant for batch)
    max_epochs: int,                    # Scalar
    
    # --- Config Constants (Tensorized) ---
    # To avoid recompilation on config changes, pass weights as tensors
    # or use a closure if config is static per run.
    config_weights: torch.Tensor        # [NumWeights] float32
) -> torch.Tensor:                      # [N] float32 (The rewards)
    pass
```

## 3. Implementation Details

### 3.1 Handling `None` Values
In Python, `seed_contribution` can be `None`. In Tensors, we use `NaN` or a separate `valid_mask`.
*   **Decision:** Use `torch.nan` for missing float values. `torch.isnan(tensor)` is fast.

### 3.2 Handling Enums
We map Enums to Integers matching `src/esper/simic/rewards/rewards.py`.
```python
STAGE_GERMINATED = 2
STAGE_TRAINING = 3
# ...
OP_GERMINATE = 1
OP_FOSSILIZE = 3
```

### 3.3 Vectorizing Components

#### A. Attribution (Ransomware Logic)
```python
# Vectorized Bounded Attribution
def compute_attribution_vectorized(seed_contrib, total_imp, weights):
    # Ransomware Check (Broadcasting)
    is_toxic = seed_contrib < 0
    penalty = weights[CONTRIB_WEIGHT] * seed_contrib
    
    # Sigmoid Discount
    # ... torch.sigmoid(...) ...
    
    # Where mask
    attribution = torch.where(is_toxic, penalty, calculated_attribution)
    return attribution
```

#### B. PBRS (Potential Based Reward Shaping)
We need a tensor lookup table for `STAGE_POTENTIALS`.
```python
# STAGE_POTENTIALS as a tensor
potential_lut = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.5, 0.0, 5.5, 6.0], device=device)

phi_curr = potential_lut[stages] + torch.clamp(epochs_in_stage * progress_bonus, max=max_bonus)
# ... similar for phi_prev ...
pbrs = weight * (gamma * phi_curr - phi_prev)
```

#### C. Action Shaping (Switch Statement)
```python
reward = torch.zeros_like(val_accs)

# Germinate Logic
mask_germ = (actions == OP_GERMINATE)
reward += mask_germ * (germ_cost + torch.where(has_seed_mask, penalty, pbrs_germ))

# Fossilize Logic
mask_foss = (actions == OP_FOSSILIZE)
# ...
```

## 4. Integration Strategy

1.  **Create `compiled_rewards.py`:** Implement the kernel. Unit test it against the Python version to ensure exact parity (or acceptable float epsilon).
2.  **Update `vectorized.py`:**
    *   Initialize SoA tensors in `init_accumulators` or similar.
    *   In the training loop, populate these tensors (Phase 1 task).
    *   Call `compute_reward_batched`.
3.  **Validation:**
    *   Run `scripts/verify_synergy_bonus.py` or similar integration tests.
    *   Verify throughput improvement using `scripts/benchmark_hot_path.py`.

## 5. Risk Assessment

*   **Complexity:** High. Reimplementing complex logic in vectorized form is error-prone.
    *   *Mitigation:* Extensive unit testing comparing `compute_reward(python)` output vs `compute_reward_batched(tensor)[i]` output on random data.
*   **Compilation Overhead:** `torch.compile` takes time to warm up.
    *   *Mitigation:* Run a warmup batch at start of training (already in `vectorized.py` plan).

## 6. Next Steps
Once Phase 1 (Data Vectorization) is complete, immediately begin implementation of `compiled_rewards.py`.
