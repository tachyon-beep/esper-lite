# PyTorch Code Review: simic/control/ Subfolder

**Date:** 2025-12-17
**Reviewer:** Claude (PyTorch Engineering Specialist)
**Scope:** `/home/john/esper-lite/src/esper/simic/control/`
**Files Reviewed:**
- `normalization.py` (227 lines)
- `__init__.py` (58 lines)

---

## Executive Summary

The `simic/control/` subfolder contains observation and reward normalization infrastructure for PPO training. The code is well-structured with good documentation and follows modern PyTorch patterns. However, the review identified several issues ranging from a potential correctness bug to performance opportunities.

**Overall Assessment:** Solid implementation with one medium-priority correctness issue and several optimization opportunities.

| Severity | Count | Summary |
|----------|-------|---------|
| Critical | 0 | None |
| High | 1 | EMA variance computation may accumulate bias |
| Medium | 3 | Performance and robustness improvements |
| Low | 4 | Minor improvements and suggestions |

---

## Critical Issues

None identified.

---

## High-Priority Issues

### H1. EMA Variance Computation May Accumulate Numerical Error Over Long Runs

**File:** `normalization.py`
**Lines:** 72-92
**Type:** Correctness / Numerical Stability

The EMA variance update uses the law of total variance cross-term correctly, but there's a subtle issue: the cross-term `momentum * (1 - momentum) * delta ** 2` scales quadratically with the mean shift but linearly with momentum factors. Over very long training runs (thousands of batches), this can lead to variance underestimation when the distribution shifts significantly.

**Current Code:**
```python
def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor,
                         batch_count: int) -> None:
    if self.momentum is not None:
        delta = batch_mean - self.mean
        self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
        self.var = (
            self.momentum * self.var
            + (1 - self.momentum) * batch_var
            + self.momentum * (1 - self.momentum) * delta ** 2
        )
```

**Issue:** When `momentum = 0.99` (as used in `vectorized.py:551`), the cross-term weight is only `0.99 * 0.01 = 0.0099`. This means sudden distribution shifts contribute minimally to variance, which is intentional for stability but can cause the variance estimate to lag significantly behind reality during distribution shift events.

**Impact:** In practice, this manifests as:
- Observation normalization may under-scale during rapid reward regime changes
- Could contribute to PPO ratio instability during early training

**Recommendation:** Consider adding a variance floor or warming up the EMA differently:
```python
# Option 1: Ensure variance doesn't collapse too far below empirical batch variance
if self.var.min() < 0.1 * batch_var.min():
    self.var = torch.maximum(self.var, 0.1 * batch_var)
```

Alternatively, this may be acceptable behavior given the intended use case (preventing distribution shift in long training). Document the trade-off explicitly.

---

## Medium-Priority Issues

### M1. Missing `@torch.inference_mode()` on `normalize()` Method

**File:** `normalization.py`
**Line:** 108
**Type:** Performance

The `update()` method correctly uses `@torch.inference_mode()`, but `normalize()` does not. While `normalize()` doesn't modify stats, it performs tensor operations that don't need gradient tracking. Adding inference mode would avoid unnecessary autograd tracking overhead.

**Current Code:**
```python
def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    """Normalize observation using running stats."""
    # Auto-migrate stats to input device (one-time cost)
    if self.mean.device != x.device:
        self.to(x.device)

    return torch.clamp(
        (x - self.mean) / torch.sqrt(self.var + self.epsilon),
        -clip, clip
    )
```

**Recommended Fix:**
```python
@torch.inference_mode()
def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
```

**Note:** This is safe because the caller (PPO update loop) operates on detached states, but explicit annotation is clearer and provides a micro-optimization.

---

### M2. Device Migration in Hot Path Could Cause Graph Breaks

**File:** `normalization.py`
**Lines:** 56-57, 114-115
**Type:** torch.compile Compatibility

The auto-migration pattern `if self.mean.device != x.device: self.to(x.device)` in both `update()` and `normalize()` introduces a conditional that could cause graph breaks under `torch.compile`. While the comment says "one-time cost," from the compiler's perspective, this is a data-dependent branch.

**Current Code:**
```python
# Auto-migrate stats to input device (one-time cost)
if self.mean.device != x.device:
    self.to(x.device)
```

**Impact:** If `RunningMeanStd` is ever used inside a compiled region, this conditional will cause a graph break every time until migration happens.

**Recommendation:** Consider initializing the normalizer on the correct device from the start (which `vectorized.py:551` already does correctly). Add a warning or assertion if device mismatch is detected in production:

```python
def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    # In production, stats should already be on correct device
    # Auto-migration is a fallback for convenience in testing/debugging
    if self.mean.device != x.device:
        import warnings
        warnings.warn(
            f"RunningMeanStd device migration from {self.mean.device} to {x.device}. "
            "Initialize with correct device for optimal performance.",
            RuntimeWarning,
            stacklevel=2
        )
        self.to(x.device)
```

---

### M3. RewardNormalizer Uses Python Floats Instead of Tensors

**File:** `normalization.py`
**Lines:** 150-227
**Type:** Performance / Design Consistency

`RewardNormalizer` uses Python floats for all its internal state (`mean`, `m2`, `count`), unlike `RunningMeanStd` which uses tensors. This means every call to `update_and_normalize()` involves Python scalar operations rather than potentially-batched tensor ops.

**Current Code:**
```python
class RewardNormalizer:
    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0
```

**Impact:**
- Cannot benefit from GPU acceleration for batch reward normalization
- Cannot be used with `torch.compile` in any meaningful way
- Inconsistent API compared to `RunningMeanStd`

**Consideration:** This may be intentional since rewards are processed one at a time in the current training loop (`vectorized.py:1899`). However, if batch reward normalization becomes desirable (e.g., for multi-step returns), this would need refactoring.

**Recommendation:** Document the design choice or consider a tensor-based implementation for future extensibility.

---

## Low-Priority Issues

### L1. Type Hint for `device` Parameter is Too Narrow

**File:** `normalization.py`
**Lines:** 37, 122
**Type:** API Design

The `device` parameter in `__init__` is typed as `str` but `to()` accepts `str | torch.device`. PyTorch commonly uses `torch.device` objects, so accepting both in `__init__` would be more ergonomic.

**Current Code:**
```python
def __init__(
    self,
    shape: tuple[int, ...],
    epsilon: float = 1e-4,
    device: str = "cpu",
    ...
```

**Recommended Fix:**
```python
def __init__(
    self,
    shape: tuple[int, ...],
    epsilon: float = 1e-4,
    device: str | torch.device = "cpu",
    ...
```

---

### L2. `state_dict()` Clones Tensors But `load_state_dict()` Doesn't Clone

**File:** `normalization.py`
**Lines:** 135-147
**Type:** API Consistency

`state_dict()` returns cloned tensors, but `load_state_dict()` directly assigns the loaded tensors (after `.to()`). This is asymmetric but probably fine since the tensors come from `torch.save`/`torch.load` which already creates new tensors.

**Current Code:**
```python
def state_dict(self) -> dict[str, torch.Tensor]:
    return {
        "mean": self.mean.clone(),
        "var": self.var.clone(),
        "count": self.count.clone(),
    }

def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self.mean = state["mean"].to(self._device)
    self.var = state["var"].to(self._device)
    self.count = state["count"].to(self._device)
```

**Recommendation:** Consider adding `.clone()` to `load_state_dict()` for symmetry and safety (prevents aliasing if caller reuses state dict):

```python
def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self.mean = state["mean"].clone().to(self._device)
    self.var = state["var"].clone().to(self._device)
    self.count = state["count"].clone().to(self._device)
```

---

### L3. `_device` String Stored But `device` Property Returns `torch.device`

**File:** `normalization.py`
**Lines:** 127, 130-133
**Type:** Minor Inconsistency

The internal `_device` is stored as a string (via `str(device)`), but the `device` property returns `torch.device`. This is fine functionally but slightly inconsistent.

**Current Code:**
```python
def to(self, device: str | torch.device) -> "RunningMeanStd":
    ...
    self._device = str(device)
    return self

@property
def device(self) -> torch.device:
    """Current device of the stats."""
    return self.mean.device  # Returns torch.device, not _device
```

**Recommendation:** This is minor since `self.mean.device` is the authoritative source. Consider removing `_device` entirely and using `self.mean.device` consistently, or document that `_device` is only used for `load_state_dict()` to know where to move loaded tensors.

---

### L4. `__init__.py` Re-exports Private Symbol `_validate_action_mask`

**File:** `__init__.py`
**Line:** 34
**Type:** API Design

The `__all__` list includes `_validate_action_mask`, which has a leading underscore convention indicating it's private/internal.

**Current Code:**
```python
from esper.tamiyo.policy.action_masks import (
    ...
    _validate_action_mask,
)

__all__ = [
    ...
    "_validate_action_mask",
]
```

**Recommendation:** Either rename to `validate_action_mask` (no underscore) if it's intended to be public, or remove it from `__all__` if it's internal-only.

---

## Integration Observations

### I1. Checkpoint Loading Has Potential Device Mismatch

**File:** `vectorized.py` (outside scope, noted for awareness)
**Lines:** 737-745

When loading checkpoints, `obs_normalizer` is restored by directly setting attributes rather than using `load_state_dict()`. This bypasses the device safety of `load_state_dict()`:

```python
obs_normalizer.mean = torch.tensor(metadata['obs_normalizer_mean'], device=device)
obs_normalizer.var = torch.tensor(metadata['obs_normalizer_var'], device=device)
```

**Observation:** This works correctly because the device is explicitly passed to `torch.tensor()`. However, using the `load_state_dict()` method would be more consistent.

---

## Test Coverage Assessment

The normalization code has good test coverage:
- `tests/simic/test_normalization.py` - Unit tests for both normalizers
- `tests/simic/control/test_normalizer_state_dict.py` - State dict roundtrip tests
- `tests/simic/properties/test_normalization_properties.py` - Property-based tests

**Gaps Identified:**
1. No tests for EMA mode (`momentum != None`) - all property tests use Welford's algorithm
2. No tests for edge cases like `momentum=0.0` or `momentum=1.0`
3. No tests for `load_state_dict()` on a different device than original

---

## torch.compile Compatibility Summary

| Component | Compatibility | Notes |
|-----------|---------------|-------|
| `RunningMeanStd.update()` | Good | Uses `@torch.inference_mode()` |
| `RunningMeanStd.normalize()` | Good | No dynamic control flow (migration branch is one-time) |
| `RewardNormalizer` | N/A | Pure Python, not compiled |
| Re-exported action_masks | Good | Uses `@torch.compiler.disable` appropriately |

---

## Recommendations Summary

1. **[High]** Document or mitigate EMA variance lag during distribution shifts
2. **[Medium]** Add `@torch.inference_mode()` to `normalize()`
3. **[Medium]** Consider warning on device migration instead of silent auto-migration
4. **[Medium]** Document that `RewardNormalizer` is intentionally scalar-based
5. **[Low]** Accept `torch.device` in `__init__` type hints
6. **[Low]** Add clone to `load_state_dict()` for safety
7. **[Test]** Add property tests for EMA mode
