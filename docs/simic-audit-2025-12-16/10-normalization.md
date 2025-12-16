# Simic Audit: normalization.py

**File:** `/home/john/esper-lite/src/esper/simic/normalization.py`
**Audit Date:** 2025-12-16
**Lines:** 186

## Executive Summary

The normalization module provides running statistics for observation and reward normalization in RL training. It implements Welford's numerically stable online algorithm and EMA variants. The code is well-designed with strong GPU-native patterns but has several areas for improvement around torch.compile compatibility, numerical precision, and type consistency.

**Overall Assessment:** Solid implementation with minor issues. The module correctly prioritizes numerical stability and GPU efficiency.

---

## 1. torch.compile Compatibility

### 1.1 RunningMeanStd.update() - Dynamic Control Flow

**Severity:** Medium
**Location:** Lines 72-92

```python
if self.momentum is not None:
    # EMA update
    ...
else:
    # Welford's algorithm
    ...
```

**Issue:** The `momentum is not None` branch creates a potential graph break if `momentum` varies at runtime. Since momentum is set at `__init__` and never changes, this is safe for a single instance, but dynamo may not optimize this path elimination.

**Recommendation:** Consider using a compile-time dispatch pattern or two separate classes if torch.compile is used in the hot path.

### 1.2 Device Migration Triggers Graph Breaks

**Severity:** Low
**Location:** Lines 56-57, 100-101

```python
if self.mean.device != x.device:
    self.to(x.device)
```

**Issue:** Device comparisons and `.to()` calls within forward/update paths will cause graph breaks when triggered. This is designed as a "one-time cost" but the check itself happens every call.

**Recommendation:** For compiled code, ensure device consistency at construction time. The auto-migration is a sensible fallback but should be avoided in compiled paths.

### 1.3 RewardNormalizer Uses Python Scalars

**Severity:** Low (not typically compiled)
**Location:** Lines 144-182

**Issue:** `RewardNormalizer` operates on Python floats, not tensors. This is intentional for scalar rewards but makes it incompatible with torch.compile vectorization. The class is used for single-value reward normalization which is inherently sequential.

**Assessment:** Not a concern since rewards are processed one-at-a-time per environment step.

---

## 2. Device Placement

### 2.1 Correct GPU-Native Design

**Severity:** None (positive finding)
**Location:** Lines 40-43, 108-114

```python
self.mean = torch.zeros(shape, device=device)
self.var = torch.ones(shape, device=device)
self.count = torch.tensor(epsilon, device=device)
```

**Assessment:** Excellent pattern. Count stored as tensor avoids CPU-GPU sync during Welford updates. The `to()` method correctly moves all three tensors together.

### 2.2 Device Property Returns torch.device

**Severity:** None (correct)
**Location:** Lines 116-119

```python
@property
def device(self) -> torch.device:
    return self.mean.device
```

**Assessment:** Correct - returns actual `torch.device` from tensor, not the cached string.

### 2.3 _device String vs torch.device Inconsistency

**Severity:** Low
**Location:** Lines 45, 113

```python
self._device = device  # In __init__: str
self._device = str(device)  # In to(): explicitly str
```

**Issue:** The `_device` attribute is stored as `str` but the constructor accepts `str` input. If passed a `torch.device`, it would be stored as-is initially but converted to string in `to()`. Minor inconsistency but does not affect functionality since `_device` is unused after initialization.

**Recommendation:** Either remove `_device` (use `self.device` property) or consistently handle type conversion.

---

## 3. Gradient Flow

### 3.1 Correct @torch.no_grad() Usage

**Severity:** None (positive finding)
**Location:** Line 48

```python
@torch.no_grad()
def update(self, x: torch.Tensor) -> None:
```

**Assessment:** Correct. Statistics updates should not flow gradients back through observations.

### 3.2 normalize() Allows Gradient Flow

**Severity:** None (correct design)
**Location:** Lines 94-106

**Assessment:** The `normalize()` method correctly does NOT have `@torch.no_grad()`, allowing gradients to flow through normalized observations when needed (e.g., for differentiable environments or meta-learning).

---

## 4. Numerical Stability

### 4.1 Division by tot_count Could Underflow

**Severity:** Low
**Location:** Lines 84, 87-88

```python
new_mean = self.mean + delta * batch_count / tot_count
m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
new_var = m2 / tot_count
```

**Issue:** If `tot_count` becomes very large (billions of samples), the division may lose precision. This is unlikely in practice since count would need to exceed float32 precision (~16M exact integers).

**Recommendation:** For extremely long training runs, consider periodic resets or using float64 for count.

### 4.2 Epsilon Initialization of Count

**Severity:** Low
**Location:** Line 43

```python
self.count = torch.tensor(epsilon, device=device)
```

**Issue:** Count is initialized to `epsilon` (1e-4) rather than 0. This prevents division-by-zero but means the first update's contribution is weighted against this tiny ghost population. This is standard practice in observation normalization.

**Note:** `RewardNormalizer` correctly uses `count = 0` with explicit `if count < 2` checks for the sample variance case.

### 4.3 sqrt(var + epsilon) Pattern

**Severity:** None (correct)
**Location:** Line 104

```python
(x - self.mean) / torch.sqrt(self.var + self.epsilon)
```

**Assessment:** Correct epsilon placement inside sqrt prevents division by zero when variance is exactly zero.

---

## 5. Memory and Performance

### 5.1 No Memory Leaks

**Severity:** None (positive finding)

**Assessment:** No gradient accumulation, no tensor lists that grow unbounded. The class correctly uses in-place-style updates (though actually creates new tensors due to Python semantics).

### 5.2 In-Place Operations Could Reduce Allocations

**Severity:** Low
**Location:** Lines 75-78, 81-92

```python
self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
```

**Issue:** Creates new tensors for each update. For high-frequency updates, this creates allocation pressure.

**Recommendation:** Consider using `mul_()` and `add_()` operations:
```python
self.mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
```

This is a minor optimization; current pattern is clearer and PyTorch's tensor allocation is efficient.

---

## 6. Integration Risks

### 6.1 RewardNormalizer Not GPU-Native

**Severity:** Medium
**Location:** Lines 122-182

```python
class RewardNormalizer:
    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = 0.0  # Python float
        self.m2 = 0.0    # Python float
        self.count = 0   # Python int
```

**Issue:** Unlike `RunningMeanStd`, `RewardNormalizer` uses Python scalars. This means:
1. Every `update_and_normalize()` call involves CPU operations
2. If rewards come from GPU tensors, `.item()` synchronization is required at call sites
3. Cannot be batched for parallel environments

**Current Usage:** In `vectorized.py:822`, used for per-step scalar rewards which is acceptable.

**Recommendation:** If multi-environment vectorized reward normalization is needed, create a tensor-based variant.

### 6.2 Serialization/Checkpointing Not Implemented

**Severity:** Medium
**Location:** Both classes

**Issue:** Neither class implements `state_dict()` / `load_state_dict()` patterns. This means:
1. Normalization statistics are lost on checkpoint restore
2. Training resumes with fresh statistics, causing distribution shift

**Recommendation:** Add state dict methods:
```python
def state_dict(self) -> dict:
    return {"mean": self.mean, "var": self.var, "count": self.count}

def load_state_dict(self, state: dict) -> None:
    self.mean = state["mean"]
    self.var = state["var"]
    self.count = state["count"]
```

### 6.3 EMA Momentum 0.99 Hardcoded in vectorized.py

**Severity:** Low
**Location:** `vectorized.py:818`

```python
obs_normalizer = RunningMeanStd((state_dim,), device=device, momentum=0.99)
```

**Issue:** Magic number 0.99 hardcoded. Should be a configurable parameter for tuning adaptation speed.

---

## 7. Code Quality

### 7.1 Excellent Documentation

**Severity:** None (positive finding)

**Assessment:** Module docstring, class docstrings, and inline comments clearly explain:
- GPU-native design rationale
- Welford's algorithm purpose
- Why reward normalization divides by std only (not mean)
- EMA vs Welford trade-offs

### 7.2 Type Annotations Complete

**Severity:** None (positive finding)

**Assessment:** All public methods have proper type hints including `tuple[int, ...]` for shape.

### 7.3 __all__ Export List Present

**Severity:** None (positive finding)
**Location:** Line 185

```python
__all__ = ["RunningMeanStd", "RewardNormalizer"]
```

### 7.4 Test Coverage Good

**Severity:** None (positive finding)

**Assessment:** Two test files cover:
- Basic functionality (`test_simic_normalization.py`)
- Property-based tests with Hypothesis (`test_normalization_properties.py`)
- GPU device tests
- Edge cases (constant values, single updates)
- Welford convergence properties

---

## 8. Summary of Findings

| Issue | Severity | Category |
|-------|----------|----------|
| Missing state_dict/load_state_dict | Medium | Integration |
| RewardNormalizer not GPU-native | Medium | Integration |
| Device comparison in hot path | Low | torch.compile |
| _device str inconsistency | Low | Code Quality |
| In-place operations could reduce allocations | Low | Memory |
| EMA momentum hardcoded | Low | Integration |
| Division precision at extreme counts | Low | Numerical |

---

## 9. Recommendations

### Priority 1 (Should Fix)
1. **Add state_dict/load_state_dict** to both classes for checkpoint compatibility
2. **Make momentum configurable** at the training loop level rather than hardcoded

### Priority 2 (Consider)
1. **Remove unused `_device` attribute** or make it consistent
2. **Create vectorized RewardNormalizer** if multi-env batched rewards become needed

### Priority 3 (Minor/Optional)
1. **Use in-place tensor ops** in update methods for reduced allocation pressure
2. **Add float64 count option** for extremely long training runs

---

## 10. Conclusion

The normalization module is well-engineered with appropriate attention to numerical stability and GPU efficiency. The main gaps are around checkpointing (missing state dict) and the scalar-only RewardNormalizer. The code follows PyTorch best practices and the test coverage is thorough. No critical issues found.
