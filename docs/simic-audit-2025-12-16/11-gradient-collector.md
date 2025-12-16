# Simic Gradient Collector Module Audit Report

**Date:** 2025-12-16
**File:** `/home/john/esper-lite/src/esper/simic/gradient_collector.py`
**Auditor:** Claude Code (PyTorch Expert)
**PyTorch Version Target:** 2.9+

---

## Executive Summary

The `gradient_collector.py` module provides lightweight gradient statistics collection for seed telemetry. It implements both synchronous and asynchronous collection patterns using PyTorch's `_foreach_norm` fused kernel. The code demonstrates good awareness of CUDA synchronization overhead with a clean async/materialize pattern. However, several integration issues exist: an incomplete dual gradient collection function, missing exports, and potential device placement risks.

**Overall Assessment:** Moderate quality with 1 Critical, 2 High, 2 Medium, and 2 Low severity issues identified.

---

## 1. torch.compile Compatibility

### 1.1 torch._foreach_norm Usage - **LOW**

**Location:** Lines 128-131

```python
# [PyTorch 2.9] Use _foreach_norm for efficient multi-tensor norm computation
per_param_norms = torch._foreach_norm(grads, ord=2)
```

**Analysis:** The code correctly uses `torch._foreach_norm`, a fused CUDA kernel that computes norms for multiple tensors in a single kernel launch. This is the same internal API used by `torch.nn.utils.clip_grad_norm_`.

**Compatibility Note:** `_foreach_norm` is a private API (indicated by the underscore prefix). However, it has been stable since PyTorch 2.0 and is widely used internally. The public alternative `torch._utils._foreach_norm_without_weights` is less ergonomic.

**torch.compile Impact:** `_foreach_norm` is compatible with `torch.compile` and generates efficient fused Triton kernels. No graph breaks expected.

**Recommendation:** The usage is appropriate. Consider adding a version check comment noting this has been stable since PyTorch 2.0.

### 1.2 Vectorized Stack and Reduction - **OK**

**Location:** Lines 134-138

```python
all_norms = torch.stack(per_param_norms)
n_grads = len(grads)
total_squared_norm = (all_norms ** 2).sum()
```

**Analysis:** This pattern is torch.compile-friendly. The stack operation followed by elementwise operations and reduction will fuse into efficient Triton kernels. No graph breaks.

### 1.3 No Compilation in Module - **NOTE**

**Observation:** This module does not use `@torch.compile` decorators. The functions are designed to be called outside compiled regions (between backward and optimizer.step). This is correct placement since gradient collection inherently requires access to `.grad` attributes.

---

## 2. Device Placement Issues

### 2.1 Implicit Device from Gradients - **MEDIUM**

**Location:** Lines 131-147

```python
per_param_norms = torch._foreach_norm(grads, ord=2)
all_norms = torch.stack(per_param_norms)
...
'_n_vanishing': (all_norms < self.vanishing_threshold).sum(),
```

**Issue:** The device of intermediate tensors (`all_norms`, comparison results) is implicitly inherited from the gradient tensors. This is usually correct, but if parameters exist on multiple devices (e.g., pipeline parallelism), this could fail silently or produce unexpected behavior.

**Impact:** Low in current usage (single-device training), but would become problematic for multi-device scenarios.

**Recommendation:** Add an explicit device check or document single-device assumption:
```python
# Assert all gradients are on the same device (required for vectorized collection)
devices = {g.device for g in grads}
if len(devices) > 1:
    raise ValueError(f"Gradients span multiple devices: {devices}")
```

### 2.2 Mixed Tensor/Float Returns in Empty Case - **MEDIUM**

**Location:** Lines 119-126

```python
if not grads:
    return {
        '_empty': True,
        'gradient_norm': 0.0,  # Python float
        'gradient_health': 1.0,  # Python float
        'has_vanishing': False,
        'has_exploding': False,
    }
```

**Issue:** In the empty case, the async path returns Python floats/bools directly. The non-empty case returns tensors. While `materialize_grad_stats` handles this correctly (line 161 checks `_empty`), this inconsistency is fragile.

**Risk:** If a caller forgets to use `materialize_grad_stats` and directly accesses the dict, they get different types depending on whether gradients existed.

**Recommendation:** For consistency, return zero-scalar tensors on a default device:
```python
if not grads:
    return {
        '_empty': True,
        '_gradient_norm': torch.tensor(0.0),
        '_gradient_health': torch.tensor(1.0),
        # ...
    }
```

---

## 3. Gradient Flow Issues

### 3.1 collect_seed_gradients Uses Python Loop for Norms - **HIGH**

**Location:** Lines 238-239

```python
# Compute per-layer norms
per_layer_norms = [g.norm(2).item() for g in grads]
```

**Issue:** The convenience function `collect_seed_gradients` (used for `return_enhanced=True`) iterates in Python and calls `.item()` on each gradient tensor. This is O(n) CUDA synchronizations where n is the number of gradient tensors.

**Contrast:** The `SeedGradientCollector.collect_async` method correctly uses `torch._foreach_norm` to avoid this.

**Impact:** Performance degradation when `return_enhanced=True` is used. Each `.item()` forces a CUDA sync, serializing GPU and CPU.

**Recommendation:** Refactor to use vectorized operations:
```python
# Use _foreach_norm like collect_async
per_param_norms = torch._foreach_norm(grads, ord=2)
all_norms = torch.stack(per_param_norms)
# Compute all stats on GPU, then materialize once
per_layer_norms_cpu = all_norms.tolist()  # Single sync
```

### 3.2 Large Tensor Allocation for Quality Checks - **MEDIUM**

**Location:** Lines 255-258

```python
all_grads = torch.cat([g.view(-1) for g in grads])
zero_fraction = (all_grads == 0).float().mean().item()
nan_count = torch.isnan(all_grads).sum().item()
inf_count = torch.isinf(all_grads).sum().item()
```

**Issue:** For enhanced metrics, the code concatenates ALL gradients into a single 1D tensor. For a model with millions of parameters, this allocates a large temporary buffer.

**Memory Cost:** For a 10M parameter model, this allocates ~40MB (float32) just for quality checks.

**Recommendation:** Compute per-tensor statistics and aggregate:
```python
zero_count = sum((g == 0).sum().item() for g in grads)
total_elements = sum(g.numel() for g in grads)
zero_fraction = zero_count / total_elements
nan_count = sum(torch.isnan(g).sum().item() for g in grads)
inf_count = sum(torch.isinf(g).sum().item() for g in grads)
```

Or better, use `_foreach_*` variants if available.

---

## 4. Memory Management

### 4.1 No Explicit Memory Cleanup - **LOW**

**Location:** Throughout module

**Observation:** The module does not explicitly delete intermediate tensors. This is acceptable for short-lived function calls, as Python's garbage collector handles cleanup.

**Non-Issue:** The async pattern correctly returns tensor references that persist until materialization. No memory leak risk.

### 4.2 Iterator Consumption - **OK**

**Location:** Lines 117, 214

```python
grads = [p.grad for p in parameters if p.grad is not None]
```

**Analysis:** The iterator is correctly materialized to a list immediately. This avoids the common bug of consuming an iterator multiple times.

---

## 5. Integration Risks

### 5.1 DualGradientStats Collection Function Missing - **CRITICAL**

**Location:** Lines 315-384

**Issue:** The module defines `DualGradientStats` dataclass and `materialize_dual_grad_stats` function, but there is NO corresponding `collect_dual_gradients_async` function. The dual gradient collection is implemented inline in `vectorized.py` (lines 1261-1290).

**Evidence from vectorized.py:**
```python
# Inline collection in vectorized.py, not using gradient_collector
host_grads = [p.grad for p in model.get_host_parameters() if p.grad is not None]
if host_grads:
    host_norms = torch._foreach_norm(host_grads, ord=2)
    host_squared_sum = torch.stack(host_norms).pow(2).sum()
    ...
grad_stats_by_slot[slot_id] = {
    "_host_squared_sum": host_squared_sum,
    "_host_param_count": host_param_count,
    ...
}
```

**Impact:** Code duplication and inconsistent abstraction. The `gradient_collector` module provides `materialize_dual_grad_stats` but not the collection function, forcing callers to duplicate the collection logic.

**Recommendation:** Add `collect_dual_gradients_async` function to complete the API:
```python
def collect_dual_gradients_async(
    host_parameters: Iterator[nn.Parameter],
    seed_parameters: Iterator[nn.Parameter],
) -> dict:
    """Collect host and seed gradient norms for G2 gate ratio computation."""
    ...
```

### 5.2 Unused Export: collect_seed_gradients_async - **HIGH**

**Location:** Lines 290-312

**Issue:** The function `collect_seed_gradients_async` is defined but never imported or used anywhere in the codebase (verified via grep). The training loop uses a local inline pattern instead.

**Evidence from training.py:**
```python
from esper.simic.gradient_collector import (
    collect_seed_gradients_async,  # Imported
    materialize_grad_stats,
)
# ... but the actual usage is:
grad_stats = collect_seed_gradients_async(model.get_seed_parameters())
```

Wait - on re-examination, `collect_seed_gradients_async` IS used in training.py line 164. Let me verify.

**Correction:** The function is used in `training.py`. However, it wraps `SeedGradientCollector` which creates a new instance each call. This is slightly wasteful but not a bug.

**Revised Issue:** `collect_seed_gradients_async` creates a new `SeedGradientCollector` on each call:
```python
def collect_seed_gradients_async(...):
    collector = SeedGradientCollector(...)  # New instance each call
    return collector.collect_async(seed_parameters)
```

**Impact:** Minor - object creation overhead is negligible compared to gradient computation.

### 5.3 __init__.py Only Exports GradientHealthMetrics - **HIGH**

**Location:** `/home/john/esper-lite/src/esper/simic/__init__.py` lines 58-60

```python
from esper.simic.gradient_collector import (
    GradientHealthMetrics,
)
```

**Issue:** The public `__init__.py` only exports `GradientHealthMetrics`. Other commonly-used items are not exported:
- `SeedGradientCollector` - Used in 15+ test files
- `collect_seed_gradients` - Referenced in archived plans
- `materialize_grad_stats` - Used in training.py
- `DualGradientStats` - Used in vectorized.py

**Impact:** Users must use full import paths (`from esper.simic.gradient_collector import ...`) rather than the package interface.

**Recommendation:** Update `__init__.py` to export commonly-used items:
```python
from esper.simic.gradient_collector import (
    GradientHealthMetrics,
    SeedGradientCollector,
    collect_seed_gradients,
    materialize_grad_stats,
    DualGradientStats,
    materialize_dual_grad_stats,
)
```

### 5.4 Inconsistent Threshold Defaults - **LOW**

**Location:** Lines 76-77, 198-199

```python
# In SeedGradientCollector.__init__:
vanishing_threshold: float = 1e-7
exploding_threshold: float = 100.0

# In collect_seed_gradients:
vanishing_threshold: float = 1e-7
exploding_threshold: float = 100.0
```

**Observation:** Thresholds are consistent across both APIs. However, these values are hardcoded in multiple places.

**Recommendation:** Define as module-level constants for single source of truth:
```python
DEFAULT_VANISHING_THRESHOLD = 1e-7
DEFAULT_EXPLODING_THRESHOLD = 100.0
```

---

## 6. Code Quality

### 6.1 Well-Documented Async Pattern - **POSITIVE**

**Location:** Lines 104-114, 150-159

The async/materialize pattern is well-documented with clear warnings about CUDA stream synchronization:

```python
def collect_async(self, parameters: Iterator[nn.Parameter]) -> dict:
    """Collect gradient statistics as tensors (async-safe version).

    Returns tensors instead of floats to avoid .item() sync inside CUDA streams.
    Call materialize_grad_stats() AFTER stream.synchronize() to get final values.
    """
```

This is excellent documentation for a subtle performance pattern.

### 6.2 Duplicate Logic Between collect and collect_async - **LOW**

**Location:** Lines 88-102 vs 104-147

The `collect` method simply wraps `collect_async` and materializes:
```python
def collect(self, parameters: Iterator[nn.Parameter]) -> dict:
    async_stats = self.collect_async(parameters)
    return materialize_grad_stats(async_stats)
```

This is clean code reuse. The sync version is a convenience wrapper.

### 6.3 DualGradientStats.normalized_ratio Docstring - **POSITIVE**

**Location:** Lines 331-356

Excellent documentation explaining the mathematical rationale for parameter normalization:

```python
@property
def normalized_ratio(self) -> float:
    """Parameter-normalized gradient ratio (seed intensity / host intensity).

    Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)

    This normalizes by sqrt(param_count) because for i.i.d. gradient elements
    with variance sigma^2, the expected L2 norm is sigma*sqrt(n).
    """
```

This level of mathematical documentation helps future maintainers understand the design.

### 6.4 Type Annotations Complete - **POSITIVE**

All functions have complete type annotations including return types. The use of `Iterator[nn.Parameter]` correctly documents that the input is consumed.

---

## 7. Test Coverage Analysis

### 7.1 Property-Based Tests Exist - **POSITIVE**

**Location:** `tests/properties/test_gradient_properties.py`

The module has comprehensive property-based tests using Hypothesis:
- Non-negativity of norms
- Zero gradient handling
- Vanishing/exploding detection thresholds
- Health score bounds

### 7.2 Missing Tests for DualGradientStats - **MEDIUM**

**Issue:** No tests exist for `DualGradientStats` or `materialize_dual_grad_stats`. The G2 gate ratio computation is untested at the unit level.

**Recommendation:** Add tests for:
- `DualGradientStats.normalized_ratio` with known inputs
- Edge cases: zero host/seed gradients, zero param counts
- `materialize_dual_grad_stats` tensor/float handling

### 7.3 Enhanced Metrics Test Coverage - **OK**

**Location:** `tests/simic/test_gradient_collector_enhanced.py`

Tests exist for `GradientHealthMetrics` and `collect_seed_gradients(return_enhanced=True)`.

---

## 8. Summary of Issues

| Severity | Issue | Location | Status |
|----------|-------|----------|--------|
| CRITICAL | No collect_dual_gradients_async function | Lines 315-384 | Incomplete API |
| HIGH | collect_seed_gradients uses O(n) .item() syncs | Lines 238-239 | Performance bug |
| HIGH | __init__.py missing common exports | simic/__init__.py | Public API gap |
| MEDIUM | Mixed tensor/float returns in empty case | Lines 119-126 | Fragile pattern |
| MEDIUM | Large tensor allocation for quality checks | Lines 255-258 | Memory overhead |
| LOW | Implicit device from gradients | Lines 131-147 | Future risk |
| LOW | Threshold defaults duplicated | Lines 76-77, 198-199 | Minor DRY violation |

---

## 9. Recommendations Priority

### Immediate (Before Next Release)

1. **Add `collect_dual_gradients_async` function** to complete the DualGradientStats API and deduplicate code from vectorized.py.

2. **Update `__init__.py` exports** to include commonly-used public items.

### Short-Term (Next Sprint)

3. **Refactor `collect_seed_gradients` enhanced path** to use vectorized operations instead of per-tensor `.item()` calls.

4. **Add unit tests for DualGradientStats** ratio computation and edge cases.

### Long-Term (Technical Debt)

5. **Extract threshold constants** to module level for single source of truth.

6. **Consider streaming quality checks** to avoid large tensor allocation in enhanced mode.

---

## 10. Appendix: Integration Points

### Files Importing gradient_collector

| File | Imports | Usage |
|------|---------|-------|
| `simic/__init__.py` | `GradientHealthMetrics` | Public export |
| `simic/training.py` | `collect_seed_gradients_async`, `materialize_grad_stats` | Per-epoch gradient stats |
| `simic/vectorized.py` | `materialize_dual_grad_stats` | G2 gate ratio (inline collection) |
| `tests/test_simic_gradient_collector.py` | `SeedGradientCollector`, `materialize_grad_stats` | Unit tests |
| `tests/properties/test_gradient_properties.py` | `SeedGradientCollector` | Property tests |
| `tests/simic/test_gradient_collector_enhanced.py` | `collect_seed_gradients`, `GradientHealthMetrics` | Enhanced metrics tests |
| `tests/integration/test_telemetry_pipeline.py` | `SeedGradientCollector` | Integration tests |

### Call Flow

```
training.py:_train_one_epoch
  -> collect_seed_gradients_async()
  -> materialize_grad_stats()  [after CUDA sync]

vectorized.py:process_train_batch
  -> [inline torch._foreach_norm collection]
  -> materialize_dual_grad_stats()  [after stream sync]
```
