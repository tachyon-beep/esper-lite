# Task 2i Validation Results: Normalization & Calibration

**Date:** 2025-12-31
**Task:** Phase 2 Observation V3 Feature Normalization Validation
**Reference:** `docs/plans/tamiyo_next/02-phase2-obs-v3-features.md` lines 829-978

---

## 1. TaskConfig Calibration Check

### CIFAR-10 Configuration

```
baseline_loss: 2.3
max_epochs: 25
target_loss: 0.3
typical_loss_delta_std: 0.05
```

**Validation Results:**

| Check | Status | Notes |
|-------|--------|-------|
| `max_epochs` vs episode length | ⚠️ **WARNING** | `max_epochs=25 < 150` (Obs V3 episode length). Epoch normalization may exceed [0,1] for episodes that run beyond epoch 25. |
| `baseline_loss` range | ✓ **PASS** | `baseline_loss=2.3` is reasonable (within [0.1, 5.0]) |
| Loss normalization | ✓ **PASS** | Baseline loss normalizes to 0.4979 using log normalization |

### TinyStories Configuration

```
baseline_loss: 10.8
max_epochs: 50
target_loss: 3.5
typical_loss_delta_std: 0.15
```

**Validation Results:**

| Check | Status | Notes |
|-------|--------|-------|
| `max_epochs` vs episode length | ⚠️ **WARNING** | `max_epochs=50 < 150` (Obs V3 episode length). Episodes may run longer. |
| `baseline_loss` range | ❌ **CRITICAL** | `baseline_loss=10.8` exceeds normalization range! Loss normalization uses `log(1+x)/log(11)` with max output 1.0 at `x=10`. Baseline loss of 10.8 normalizes to **1.0293**, exceeding [0,1] bounds. |
| Loss normalization overflow | ❌ **CRITICAL** | TinyStories baseline loss saturates the normalization function. Policy will see out-of-distribution observations at episode start. |

**Critical Issue Identified:**

The log normalization formula `log(1 + loss) / log(11)` assumes loss values in [0, 10] range:
- `loss=0.0` → `0.0000`
- `loss=10.0` → `1.0000` (upper bound)
- `loss=10.8` → `1.0293` ⚠️ **EXCEEDS [0,1]**

This means TinyStories episodes will start with **out-of-range observations**, violating the normalization contract.

---

## 2. Epoch Normalization Verification

**Test Results:**

```
Epoch   0 → 0.0000
Epoch  50 → 0.3333
Epoch 100 → 0.6667
Epoch 150 → 1.0000
```

**Status:** ✓ **VERIFIED**

All test cases produce normalized values in [0, 1] range. The normalization formula `epoch / MAX_EPOCHS_NORM` (where `MAX_EPOCHS_NORM = 150.0`) correctly handles epochs from 0 to 150.

**Implementation:** `/home/john/esper-lite/src/esper/tamiyo/policy/features.py`

```python
MAX_EPOCHS_NORM = 150.0  # Expected max episode length from Simic
epoch_norm = signal.metrics.epoch / MAX_EPOCHS_NORM
```

---

## 3. Configuration Mismatch Analysis

### The Problem

Both `TaskConfig.for_cifar10()` and `TaskConfig.for_tinystories()` have `max_epochs < 150`, but Observation V3 uses `MAX_EPOCHS_NORM = 150.0` as the normalization denominator.

**This means:**
- Episodes can run longer than the task's expected `max_epochs`
- Epoch normalization will exceed 1.0 if an episode runs beyond `max_epochs`
- `TaskConfig.max_epochs` is NOT used for normalization (by design)

### Why This Exists

The separation is intentional:

- **`TaskConfig.max_epochs`**: Task-specific expectation for when training should end (used by Tolaria for episode termination logic)
- **`MAX_EPOCHS_NORM = 150.0`**: Global normalization constant for Observation V3 (fixed across all tasks)

**Rationale:** Using a fixed normalization constant prevents observation space shift when switching tasks. The PPO policy sees a consistent [0, 1] range regardless of task.

### Risk Assessment

| Scenario | Risk Level | Impact |
|----------|------------|--------|
| CIFAR-10 episode runs to epoch 25 | ✓ **None** | Epoch normalizes to 25/150 = 0.167, well within [0,1] |
| CIFAR-10 episode runs to epoch 150 | ✓ **None** | Epoch normalizes to 150/150 = 1.0, at upper bound |
| CIFAR-10 episode runs beyond 150 | ⚠️ **Low** | Epoch normalization exceeds 1.0, but Simic should terminate at max_steps=150 |
| TinyStories episode runs to epoch 50 | ✓ **None** | Epoch normalizes to 50/150 = 0.333 |
| TinyStories episode runs to epoch 150+ | ⚠️ **Low** | Same as CIFAR-10 |

**Mitigation:** Simic's `max_steps=150` (from `ParallelEnvRunner`) should hard-cap episodes at 150 epochs, preventing normalization overflow.

---

## 4. Normalizer Contract Documentation

### Current State: Deferred

As noted in the plan (lines 875-978), the **Observation Normalizer** is currently deferred. The system does NOT apply running mean/std normalization to observations.

### Why It's Deferred

1. **Obs V3 features are already normalized** to [0, 1] or [-1, 1] ranges using domain knowledge
2. **Running statistics would be fragile** during early training when observation distributions are unstable
3. **Added complexity** for minimal gain (features are already well-scaled)

### Future Normalizer Contract (when implemented)

If a normalizer is added in the future, it MUST follow this contract:

```python
from dataclasses import dataclass
import torch

@dataclass
class ObservationNormalizer:
    """Running mean/std normalization for Obs V3 features.

    Contract:
    - Tracks running mean/std across all parallel envs
    - Applies normalization: (obs - mean) / (std + eps)
    - Updates statistics AFTER each step (not during inference)
    - Serializes state for checkpoint save/load
    """
    running_mean: torch.Tensor  # Shape: (obs_dim,)
    running_std: torch.Tensor   # Shape: (obs_dim,)
    count: int = 0              # Number of observations seen
    eps: float = 1e-8           # Numerical stability
    momentum: float = 0.99      # EMA momentum (high = slow updates)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply normalization (inference mode, no state update)."""
        return (obs - self.running_mean) / (self.running_std + self.eps)

    def update(self, obs: torch.Tensor) -> None:
        """Update running statistics with new observation batch."""
        # Update running_mean and running_std using EMA
        pass

    def state_dict(self) -> dict:
        """Serialize for checkpoint save."""
        return {
            "running_mean": self.running_mean,
            "running_std": self.running_std,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.running_mean = state["running_mean"]
        self.running_std = state["running_std"]
        self.count = state["count"]
```

### Logging Utility (Ready for Future Use)

When the normalizer is implemented, use this logging helper:

```python
def log_normalizer_update(normalizer: ObservationNormalizer, logger) -> None:
    """Log normalizer statistics for debugging distribution shift.

    Call this every N steps (e.g., every 100 steps) to track:
    - Mean/std drift over time
    - Feature saturation (mean near bounds)
    - Instability (std collapse or explosion)
    """
    logger.info(
        f"[ObsNormalizer] count={normalizer.count} "
        f"mean_range=[{normalizer.running_mean.min():.3f}, {normalizer.running_mean.max():.3f}] "
        f"std_range=[{normalizer.running_std.min():.3f}, {normalizer.running_std.max():.3f}]"
    )

    # Alert on pathological cases
    if (normalizer.running_std < 0.01).any():
        logger.warning("[ObsNormalizer] Some features have collapsed std < 0.01 (feature not varying)")
    if (normalizer.running_mean.abs() > 5.0).any():
        logger.warning("[ObsNormalizer] Some features have extreme mean > 5.0 (check for bugs)")
```

**Invocation point:** After `normalizer.update(obs)` in `ParallelEnvRunner.step()`.

---

## Summary

| Validation Task | Status | Notes |
|----------------|--------|-------|
| CIFAR-10 calibration | ✓ **PASS** | `max_epochs=25 < 150` is by design (normalization uses fixed constant). Baseline loss normalizes correctly. |
| TinyStories calibration | ❌ **FAIL** | **CRITICAL:** `baseline_loss=10.8` exceeds normalization range [0, 10], causing out-of-bounds observations (1.0293 instead of max 1.0). |
| Epoch normalization | ✓ **PASS** | Verified [0,1] range for epochs 0-150 |
| Normalizer contract | ✓ **Documented** | Deferred for now, contract provided for future implementation |

---

## Critical Finding: TinyStories Loss Normalization Overflow

**Problem:** The loss normalization function `log(1 + loss) / log(11)` was designed for loss values in [0, 10]:
- Domain: `[0, 10]`
- Range: `[0, 1.0]`

**TinyStories violates this assumption:**
- `baseline_loss = 10.8`
- Normalized value: `log(1 + 10.8) / log(11) = 1.0293`
- **Out of bounds by 2.93%**

**Impact:**
1. Policy network sees observations > 1.0 at episode start (out-of-distribution)
2. PPO assumes bounded observation space (Gym contract violation)
3. May cause training instability or suboptimal policy learning

**Root Cause:**
Language modeling tasks (TinyStories) have higher initial cross-entropy loss than vision tasks (CIFAR-10). The normalization constant `log(11)` was calibrated for CIFAR-10 loss ranges, not LM tasks.

---

## Recommended Fixes

### Option 1: Increase Normalization Range (Preferred)

Change loss normalization to support loss values up to 15:

```python
# In src/esper/tamiyo/policy/features.py
# Old: log(1 + loss) / log(11)  # max loss = 10
# New: log(1 + loss) / log(16)  # max loss = 15
loss_norm = torch.log(1.0 + signal.metrics.val_loss) / math.log(16)
```

**Normalization table with new formula:**

| Loss | Old (log 11) | New (log 16) | Status |
|------|--------------|--------------|--------|
| 0.0  | 0.0000       | 0.0000       | ✓ Same |
| 2.3  | 0.4979       | 0.4311       | ✓ Scaled |
| 10.0 | 1.0000       | 0.8662       | ✓ In bounds |
| 10.8 | **1.0293**   | 0.8954       | ✓ **FIXED** |
| 15.0 | 1.1062       | 1.0000       | ✓ New max |

**Pros:**
- Handles both CIFAR-10 and TinyStories
- Minimal code change
- Preserves [0, 1] contract

**Cons:**
- Slightly reduces resolution for CIFAR-10 (loss 2.3 → 0.43 instead of 0.50)

### Option 2: Task-Specific Normalization

Use `TaskConfig` to specify max loss per task:

```python
@dataclass
class TaskConfig:
    baseline_loss: float
    target_loss: float
    max_epochs: int
    typical_loss_delta_std: float
    max_loss_for_normalization: float = 10.0  # NEW

# In features.py
max_loss = task_config.max_loss_for_normalization
loss_norm = torch.log(1.0 + signal.metrics.val_loss) / math.log(1 + max_loss)
```

**Pros:**
- Task-specific calibration
- Can optimize normalization per task

**Cons:**
- More complex
- Observation space changes between tasks (breaks transfer learning)

### Option 3: Clip at Bounds

Add clipping to enforce [0, 1] range:

```python
loss_norm = torch.clamp(
    torch.log(1.0 + signal.metrics.val_loss) / math.log(11),
    min=0.0,
    max=1.0
)
```

**Pros:**
- Minimal change
- Guarantees [0, 1] bounds

**Cons:**
- Loses information (all losses > 10 map to 1.0)
- Saturation at upper bound during early training

---

## Action Items

**Immediate (blocking Task 2i completion):**

1. ❌ **BLOCKER:** Fix TinyStories loss normalization overflow
   - Recommend Option 1 (increase to `log(16)`)
   - Update `src/esper/tamiyo/policy/features.py`
   - Re-run validation tests

2. ✓ Epoch normalization verified working
3. ✓ TaskConfig/normalization constant mismatch documented (by design)
4. ✓ Normalizer contract documented for future use

**Future (low priority):**

5. 🔲 Consider renaming `TaskConfig.max_epochs` → `expected_epochs` for clarity
6. 🔲 Add validation test to catch normalization range violations in CI

---

**Completed:** 2025-12-31
**Validated by:** Claude Code (Task 2i execution)
