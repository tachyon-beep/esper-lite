# Deep RL Code Review: simic/control/ Subfolder

**Reviewer**: DRL Specialist
**Date**: 2025-12-17
**Scope**: `/home/john/esper-lite/src/esper/simic/control/` (normalization.py, __init__.py)

---

## Executive Summary

The `simic/control/` subfolder contains observation and reward normalization infrastructure for PPO training. The implementation is **fundamentally sound** with correct Welford's algorithm for running statistics and appropriate EMA support for long training runs. The code demonstrates good understanding of RL normalization requirements.

**Overall Assessment**: **Good** - Minor improvements possible, no critical bugs.

| Category | Issues Found |
|----------|--------------|
| Critical | 0 |
| High Priority | 1 |
| Medium Priority | 3 |
| Low Priority | 4 |

---

## Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| `normalization.py` | 227 | Running mean/std for observations, reward normalization |
| `__init__.py` | 58 | Re-exports from normalization.py and tamiyo.policy |

---

## Critical Issues

**None identified.**

The core algorithms (Welford's and EMA variance) are correctly implemented with proper numerical stability considerations.

---

## High Priority Issues

### H1. No Test Coverage for EMA Momentum Mode

**Location**: `normalization.py` lines 72-92 (EMA update path)
**Severity**: High - Production code path with no unit tests

**Problem**: The `momentum` parameter enables EMA mode (used in production with `momentum=0.99`), but there are no unit tests specifically exercising this code path. The test suite only tests the default Welford mode (`momentum=None`).

**Evidence from test files**:
- `test_normalization.py`: All tests use default `RunningMeanStd(shape=...)` without momentum
- `test_normalization_properties.py`: All property tests use default mode
- `test_normalizer_state_dict.py`: State dict tests don't verify momentum preservation

**RL Impact**: EMA mode is specifically chosen for long training runs to prevent distribution shift. If the EMA variance calculation has a bug (e.g., the cross-term formula), the policy would see inconsistently normalized observations, causing PPO ratio explosion.

**The EMA formula at lines 84-90**:
```python
delta = batch_mean - self.mean
self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
self.var = (
    self.momentum * self.var
    + (1 - self.momentum) * batch_var
    + self.momentum * (1 - self.momentum) * delta ** 2
)
```

**Verification needed**: The law of total variance cross-term `m*(1-m)*delta^2` is correct mathematically, but should be tested with known inputs to verify the implementation.

**Recommendation**: Add dedicated test cases:
```python
def test_ema_mode_convergence():
    """EMA mode should track shifting distributions."""
    rms = RunningMeanStd(shape=(1,), momentum=0.99)
    # First distribution: mean=0
    for _ in range(100):
        rms.update(torch.randn(32, 1))
    # Second distribution: mean=10
    for _ in range(100):
        rms.update(torch.randn(32, 1) + 10.0)
    # Mean should have shifted toward 10
    assert rms.mean[0] > 5.0

def test_ema_variance_formula():
    """Cross-term in EMA variance must be computed correctly."""
    # Test with known values where cross-term matters
    ...
```

---

## Medium Priority Issues

### M1. RewardNormalizer Uses Python Floats Instead of Tensors

**Location**: `normalization.py` lines 150-225
**Severity**: Medium - Performance concern, not correctness

**Problem**: `RewardNormalizer` operates on Python floats while `RunningMeanStd` uses tensors. This creates a CPU-GPU synchronization point in the training loop.

**Evidence from vectorized.py line 1899**:
```python
normalized_reward = reward_normalizer.update_and_normalize(reward)
```

The `reward` variable is already a Python float (scalar), so the current implementation is consistent. However, if rewards are computed on GPU (which they could be with batched reward computation), this forces a sync.

**Current flow**:
1. `compute_reward()` returns Python float
2. `reward_normalizer.update_and_normalize(float)` operates on CPU
3. Result stored in buffer

**Impact**: Minor. The current design is intentional (rewards are per-step scalars), but future batched reward normalization would require refactoring.

**Recommendation**: Document the design decision. If batched rewards become important, create a `BatchedRewardNormalizer` variant.

---

### M2. Deferred Normalizer Update Could Cause Stale Statistics on Resume

**Location**: `normalization.py` lines 49-63 (update method) + `vectorized.py` lines 289-292

**Problem**: Observation normalizer statistics are updated AFTER the PPO update (correct for training stability), but checkpoint saves occur before this update completes for the final batch.

**Evidence from vectorized.py**:
```python
# Line 289-292: Update normalizer AFTER PPO updates
if raw_states_for_normalizer_update and update_metrics:
    all_raw_states = torch.cat(raw_states_for_normalizer_update, dim=0)
    obs_normalizer.update(all_raw_states)

# Line 2432-2435: Save normalizer state (happens after update in same batch)
'obs_normalizer_mean': obs_normalizer.mean.tolist(),
```

**Scenario**: If training is interrupted between collecting rollouts and the PPO update, the next resume will have normalizer statistics that don't include the last batch of observations.

**Impact**: Low in practice - the deferred update design is correct for training stability. The statistical impact of missing one batch of observations is minimal given the EMA momentum of 0.99.

**Recommendation**: Document the intentional design. Consider logging a warning if the raw_states buffer is non-empty at checkpoint time.

---

### M3. Missing Validation for Momentum Parameter Range

**Location**: `normalization.py` lines 33-46 (`__init__`)

**Problem**: The `momentum` parameter is not validated to be in the valid range [0, 1].

```python
def __init__(
    self,
    shape: tuple[int, ...],
    epsilon: float = 1e-4,
    device: str = "cpu",
    momentum: float | None = None,  # No validation
):
```

**Impact**: Momentum values outside [0, 1] would produce incorrect EMA statistics:
- `momentum < 0`: Negative weights cause oscillating statistics
- `momentum > 1`: Weights sum to > 1, causing variance explosion
- `momentum = 1`: Statistics never update (stuck at initialization)

**Recommendation**: Add validation:
```python
if momentum is not None and not (0.0 <= momentum < 1.0):
    raise ValueError(f"momentum must be in [0, 1), got {momentum}")
```

Note: `momentum=1.0` should be disallowed since it prevents any adaptation.

---

## Low Priority Issues

### L1. Count Initialization Inconsistency Between Normalizers

**Location**: `normalization.py` lines 43 vs 175

**Problem**: `RunningMeanStd` initializes count to epsilon (1e-4), while `RewardNormalizer` initializes count to 0.

```python
# RunningMeanStd (line 43)
self.count = torch.tensor(epsilon, device=device)

# RewardNormalizer (line 175)
self.count = 0  # Start at 0, not epsilon
```

**Rationale for difference**:
- `RunningMeanStd`: Epsilon count prevents division by zero in Welford update (line 96: `delta * batch_count / tot_count`)
- `RewardNormalizer`: Checks `count < 2` before using variance, so zero-init is safe

**Impact**: None - both are correct for their use cases. The inconsistency is justified.

**Recommendation**: Add a comment to `RunningMeanStd` explaining why count starts at epsilon:
```python
# Initialize count to epsilon (not 0) to prevent division by zero
# in Welford's update before first batch arrives
self.count = torch.tensor(epsilon, device=device)
```

---

### L2. Device Type Annotation Could Be More Precise

**Location**: `normalization.py` lines 122-128

**Problem**: The `to()` method accepts `str | torch.device` but stores `str(device)`:

```python
def to(self, device: str | torch.device) -> "RunningMeanStd":
    """Move stats to device."""
    self.mean = self.mean.to(device)
    self.var = self.var.to(device)
    self.count = self.count.to(device)
    self._device = str(device)  # Always converts to string
    return self
```

**Impact**: Minimal - the string representation works for most cases. However, `str(torch.device("cuda:0"))` gives `"cuda:0"` which works, but this loses type information.

**Recommendation**: Consider using `torch.device` consistently:
```python
self._device = torch.device(device)
```

---

### L3. state_dict() Doesn't Include Momentum for RunningMeanStd

**Location**: `normalization.py` lines 135-141

**Problem**: The `state_dict()` method doesn't include the `momentum` parameter, which is critical for EMA mode:

```python
def state_dict(self) -> dict[str, torch.Tensor]:
    """Return state dictionary for checkpointing."""
    return {
        "mean": self.mean.clone(),
        "var": self.var.clone(),
        "count": self.count.clone(),
        # Missing: "momentum": self.momentum
    }
```

**Mitigation**: The vectorized training loop manually saves/restores momentum (lines 2435, 745), so this is not a bug in practice.

**Recommendation**: Include momentum in state_dict for self-contained checkpointing:
```python
def state_dict(self) -> dict[str, torch.Tensor | float | None]:
    return {
        "mean": self.mean.clone(),
        "var": self.var.clone(),
        "count": self.count.clone(),
        "momentum": self.momentum,
    }
```

---

### L4. RewardNormalizer.epsilon Not Restored from State Dict

**Location**: `normalization.py` lines 220-224

**Problem**: `load_state_dict()` doesn't restore `epsilon` or `clip`:

```python
def load_state_dict(self, state: dict[str, float | int]) -> None:
    """Load state from dictionary."""
    self.mean = state["mean"]
    self.m2 = state["m2"]
    self.count = state["count"]
    # Missing: self.epsilon, self.clip
```

**Impact**: Low - these are configuration parameters typically set at construction. The current design assumes the same config is used when loading.

**Recommendation**: Either:
1. Document that config params must match on load, OR
2. Include epsilon/clip in state_dict with optional restoration

---

## Integration Analysis

### Observation Normalization Flow

The flow is correct for PPO stability:

1. **Rollout Collection** (vectorized.py lines 1573-1582):
   - Raw states accumulated in `raw_states_for_normalizer_update`
   - Normalization uses FROZEN statistics
   - Prevents "normalizer drift" within a batch

2. **PPO Update** (lines 287-292):
   - After successful update, normalizer statistics updated
   - All states in batch normalized with same mean/var

3. **Checkpoint** (lines 2432-2439):
   - Mean, var, count, momentum all saved
   - Reward normalizer state also saved

This is the **correct pattern** for PPO. Updating normalizer statistics during rollout would cause states early in the batch to have different normalization than states late in the batch, breaking the PPO ratio calculation.

### Reward Normalization Flow

The reward normalization divides by std only (no mean subtraction), which is correct:

```python
# Line 198-202: Normalize by std only
std = max(self.epsilon, (self.m2 / (self.count - 1)) ** 0.5)
normalized = reward / std
```

**Why std-only is correct for RL** (documented in lines 153-160):
- The critic learns E[R] through its value function target
- Subtracting running mean from rewards creates non-stationary targets
- When mean shifts, the critic must constantly recalibrate
- Dividing by std preserves reward semantics while stabilizing magnitudes

### EMA vs Welford Design Decision

The choice of `momentum=0.99` for observation normalization (vectorized.py line 551) is appropriate:

**Welford (momentum=None)**:
- All history equally weighted
- Mean/var converge to true population statistics
- Problem: In RL, the observation distribution shifts as policy improves

**EMA (momentum=0.99)**:
- Recent observations weighted more heavily
- Tracks non-stationary distributions
- 0.99 momentum = effective window of ~100 batches (1/(1-0.99))

For a PPO agent managing seed lifecycle, observation distributions will shift as:
1. Seeds progress through stages (DORMANT -> TRAINING -> BLENDING -> FOSSILIZED)
2. Policy learns better strategies
3. Host model improves

EMA is the correct choice here.

---

## Summary of Recommendations

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| H1 | No EMA test coverage | Add unit tests for momentum mode |
| M1 | RewardNormalizer uses floats | Document design decision |
| M2 | Deferred update checkpoint timing | Document design, consider warning |
| M3 | No momentum validation | Add range check [0, 1) |
| L1 | Count init inconsistency | Add comment explaining epsilon init |
| L2 | Device type annotation | Consider using torch.device consistently |
| L3 | state_dict missing momentum | Include momentum in state_dict |
| L4 | RewardNormalizer epsilon not restored | Document or include in state |

---

## Appendix: Algorithm Correctness Verification

### Welford's Algorithm (Lines 94-106)

The implementation follows the standard parallel/batched Welford's algorithm:

```python
delta = batch_mean - self.mean
tot_count = self.count + batch_count
new_mean = self.mean + delta * batch_count / tot_count
m_a = self.var * self.count
m_b = batch_var * batch_count
m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
new_var = m2 / tot_count
```

This is **correct**. The key insight is that when merging two groups:
- `m_a = var_a * n_a` is the sum of squared deviations for group A
- `m_b = var_b * n_b` is the sum of squared deviations for group B
- The cross-term `delta^2 * n_a * n_b / n_total` accounts for between-group variance

Reference: Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products"

### EMA Variance with Cross-Term (Lines 84-90)

The EMA variance formula includes the law of total variance cross-term:

```python
self.var = (
    self.momentum * self.var
    + (1 - self.momentum) * batch_var
    + self.momentum * (1 - self.momentum) * delta ** 2
)
```

This is **correct**. For weighted combination of two distributions:
```
Var_combined = w1*Var1 + w2*Var2 + w1*w2*(Mean1 - Mean2)^2
```

With w1 = momentum, w2 = (1-momentum), and w1 + w2 = 1, the cross-term coefficient is `m*(1-m)`.

The delta is computed BEFORE updating the mean (line 84), which is critical for correctness.

### Reward Normalizer Welford (Lines 188-202)

```python
self.count += 1
delta = reward - self.mean
self.mean += delta / self.count
delta2 = reward - self.mean  # Note: uses NEW mean
self.m2 += delta * delta2
```

This is the **correct single-sample Welford's algorithm**. The key is using `delta` (old mean) and `delta2` (new mean) together, which gives `m2 = sum((x - mean)^2)`.

The sample variance computation `m2 / (count - 1)` is correct for unbiased estimation with `count >= 2` check.

---

*Report generated by DRL Specialist reviewing simic/control/ normalization infrastructure.*
