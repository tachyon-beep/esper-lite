# Simic Audit: features.py

**File:** `/home/john/esper-lite/src/esper/simic/features.py`
**Date:** 2025-12-16
**Auditor:** Claude (PyTorch Engineering Specialist)

## Executive Summary

The `features.py` module is a **HOT PATH** component responsible for extracting features from training observations for the RL policy. It is deliberately designed with minimal dependencies (leyline only) to avoid import-time overhead in the vectorized training loop.

**Overall Assessment:** LOW RISK - Well-designed pure Python module with clean separation of concerns. No PyTorch operations in the core feature extraction path.

| Category | Risk Level | Issues |
|----------|------------|--------|
| torch.compile | N/A | No compiled code in this module |
| Device Placement | N/A | Pure Python, no tensors |
| Gradient Flow | N/A | Feature extraction only |
| Memory | LOW | List allocations on hot path |
| Integration | MEDIUM | Normalization contract incomplete |
| Code Quality | LOW | Minor documentation issues |

---

## 1. Architecture Analysis

### 1.1 Module Purpose

This module provides:
1. `safe()` - Safe float conversion with NaN/inf handling
2. `obs_to_multislot_features()` - 50-dimensional feature extraction from observation dicts
3. `TaskConfig` - Task-specific configuration dataclass
4. `MULTISLOT_FEATURE_SIZE` - Constant for observation space dimensionality

### 1.2 Import Discipline

The module header emphasizes HOT PATH constraints:

```python
# HOT PATH: ONLY leyline imports allowed!
```

**Current imports:**
- `dataclasses.dataclass`
- `math`
- `typing.TYPE_CHECKING`

**Positive:** No leyline imports at runtime despite the comment allowing them. This is even stricter than documented, which is good for performance.

### 1.3 Feature Layout (50 dimensions)

| Index | Feature | Notes |
|-------|---------|-------|
| 0-1 | Timing (epoch, global_step) | Raw values, unbounded |
| 2-4 | Losses (train, val, delta) | Clipped via `safe()` |
| 5-7 | Accuracies (train, val, delta) | Raw percentages |
| 8-10 | Trends (plateau, best_acc, best_loss) | Mixed scales |
| 11-15 | Loss history (5 values) | Clipped via `safe()` |
| 16-20 | Accuracy history (5 values) | Raw percentages |
| 21 | Total params | Raw count, unbounded |
| 22 | Seed utilization | Normalized [0, 1] |
| 23-31 | Early slot (4 state + 5 blueprint) | Mixed scales |
| 32-40 | Mid slot (4 state + 5 blueprint) | Mixed scales |
| 41-49 | Late slot (4 state + 5 blueprint) | Mixed scales |

---

## 2. Detailed Findings

### 2.1 Normalization Contract Inconsistency

**Severity: MEDIUM**
**Location:** Lines 76-164, entire `obs_to_multislot_features()` function

**Issue:** Features have inconsistent normalization across dimensions:

| Feature | Range | Problem |
|---------|-------|---------|
| `epoch` | 0 to max_epochs (e.g., 50) | Unbounded, varies by task |
| `global_step` | 0 to 50000+ | Unbounded, varies by task |
| `train_accuracy` | 0 to 100 | Different scale than [0,1] features |
| `total_params` | 1000 to 10M+ | Extremely large magnitude |
| `seed_utilization` | 0 to 1 | Properly normalized |
| `improvement` | Raw percentage points | TODO comment acknowledges this issue |

**Evidence:**
```python
# Line 151-154
# TODO: [OBS NORMALIZATION AUDIT] - Audit PPO observation scaling/clamping for
# per-slot improvement/counterfactual (currently raw percentage points) and
# align with the ~[-1, 1] normalization contract for stable policy learning.
float(slot.get('improvement', 0.0)),
```

**Impact:** PPO policy learning can be destabilized by features with wildly different scales. The network must learn implicit normalization, wasting capacity.

**Integration Context:** The `RunningMeanStd` normalizer in `simic/normalization.py` handles this at the vectorized training level, but the raw feature extraction should ideally produce pre-normalized values for more stable learning.

**Recommendation:** Either:
1. Normalize features in `obs_to_multislot_features()` using `TaskConfig` bounds
2. Document that normalization is deferred to `RunningMeanStd` (current approach)

---

### 2.2 List Allocation on Hot Path

**Severity: LOW**
**Location:** Lines 123-163

**Issue:** Multiple list allocations and extensions on every feature extraction call:

```python
base = [
    float(obs['epoch']),
    ...
    *[safe(v, 10.0) for v in obs['loss_history_5']],  # List comprehension
    *obs['accuracy_history_5'],
]

slot_features = []
for slot_id in ['early', 'mid', 'late']:
    slot_features.extend([...])  # Multiple extends
    blueprint_one_hot = [0.0] * _NUM_BLUEPRINT_TYPES
    slot_features.extend(blueprint_one_hot)

return base + slot_features  # List concatenation
```

**Impact:** In typical vectorized training (4+ environments, 25+ epochs per episode), this function is called thousands of times. The allocations are small but cumulative.

**Benchmark Estimate:** ~0.5-1.0 microseconds per call (negligible compared to PyTorch forward passes).

**Recommendation:** Current implementation is acceptable. Pre-allocating a fixed-size list would add complexity without meaningful performance gain.

---

### 2.3 TaskConfig Placement Concern

**Severity: LOW**
**Location:** Lines 172-213

**Issue:** `TaskConfig` is defined in `features.py` but used by multiple domains:

```
# Usage across codebase:
src/esper/kasmina/slot.py:69:    from esper.simic.features import TaskConfig
src/esper/runtime/tasks.py:11:from esper.simic.features import TaskConfig
```

**Concern:** Per CLAUDE.md, shared types should live in `leyline`:

> **All new constants, enums, and shared types MUST be placed in `leyline`.**

**Counter-argument:** `TaskConfig` is currently simic-specific (RL observation normalization) and only imported by domains that integrate with simic training. Moving to leyline would create a dependency inversion (leyline depending on training concepts).

**Recommendation:** Keep in `features.py` until `TaskConfig` is needed by domains that don't depend on simic.

---

### 2.4 Documentation Discrepancy

**Severity: LOW**
**Location:** Lines 76-118

**Issue:** Docstring claims "Returns: List of 35 floats" but actual return is 50 floats.

```python
def obs_to_multislot_features(obs: dict, total_seeds: int = 0, max_seeds: int = 1) -> list[float]:
    """Extract features including per-slot state (50 dims).
    ...
    Returns:
        List of 35 floats  # <-- INCORRECT
    """
```

The docstring header correctly states "50 dims" but the Returns section is stale.

**Recommendation:** Update Returns section to "List of 50 floats".

---

### 2.5 Blueprint Mapping Hardcoded

**Severity: LOW**
**Location:** Lines 66-73

**Issue:** Blueprint-to-index mapping is hardcoded and must be kept in sync with `BlueprintAction` enum:

```python
_BLUEPRINT_TO_INDEX = {
    "noop": 0,
    "conv_light": 1,
    "attention": 2,
    "norm": 3,
    "depthwise": 4,
}
_NUM_BLUEPRINT_TYPES = 5
```

**Risk:** If `BlueprintAction` enum is modified, this mapping becomes stale. There's no compile-time or test-time enforcement of consistency.

**Mitigating Factor:** Tests in `test_features.py` exercise the blueprint encoding, which would catch some mismatches.

**Recommendation:** Add a test that validates `_BLUEPRINT_TO_INDEX` against the actual `BlueprintAction` enum from leyline.

---

### 2.6 safe() Function Edge Cases

**Severity: LOW**
**Location:** Lines 37-52

**Analysis:** The `safe()` function is well-designed:

```python
def safe(v, default: float = 0.0, max_val: float = 100.0) -> float:
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return max(-max_val, min(float(v), max_val))
```

**Positive aspects:**
1. Handles `None` gracefully
2. Handles `NaN` and `inf` for floats
3. Clips to bounded range

**Edge case not handled:** Integer `inf` (which doesn't exist in Python, so this is fine).

**Minor concern:** The type check `isinstance(v, float)` means integer NaN checks are skipped. However, Python integers cannot be NaN/inf, so this is correct behavior.

---

## 3. torch.compile Considerations

**Status:** Not applicable

This module contains no PyTorch operations. It is pure Python feature extraction that produces `list[float]` outputs.

**Integration point:** The feature lists are converted to tensors in `simic/ppo.py` at line 137:

```python
features = obs_to_multislot_features(obs, total_seeds=total_seeds, max_seeds=max_seeds)
```

The tensor conversion and any compilation happens in the calling code, not here.

---

## 4. Device Placement Considerations

**Status:** Not applicable

No tensors are created or manipulated in this module. Device placement is handled by:
1. `TamiyoRolloutBuffer` (stores features as tensors)
2. `RunningMeanStd` (normalizes on correct device)

---

## 5. Gradient Flow Considerations

**Status:** Not applicable

Feature extraction is a preprocessing step that occurs before tensor creation. No gradients flow through this code.

---

## 6. Memory Considerations

### 6.1 Per-Call Allocation

**Status:** LOW concern

Each call to `obs_to_multislot_features()` allocates:
- 1 list of 23 floats (base features)
- 1 list of 27 floats (slot features)
- 1 final list of 50 floats (concatenation)
- 3 intermediate blueprint one-hot lists of 5 floats each

Total: ~400 bytes per call (rough estimate including Python object overhead).

At 4 environments x 25 epochs x 1 step/epoch = 100 calls per episode, this is ~40KB per episode. Negligible.

### 6.2 No Caching or Memoization

The module does not cache any results. Given the dynamic nature of observations, this is correct.

---

## 7. Integration Risk Assessment

### 7.1 Upstream Dependencies

```
features.py
    <- dataclasses (stdlib)
    <- math (stdlib)
    <- typing.TYPE_CHECKING (stdlib)
```

**Risk:** NONE - No external dependencies beyond stdlib.

### 7.2 Downstream Consumers

```
features.py exports:
    -> obs_to_multislot_features (used by ppo.py)
    -> MULTISLOT_FEATURE_SIZE (used by vectorized.py)
    -> TaskConfig (used by slot.py, tasks.py)
    -> safe (used by features.py internally, exported via __init__.py)
```

**Key integration:** `vectorized.py` line 814:
```python
state_dim = MULTISLOT_FEATURE_SIZE + (SeedTelemetry.feature_dim() * 3 if use_telemetry else 0)
```

If `MULTISLOT_FEATURE_SIZE` changes without updating this calculation, dimension mismatch errors will occur.

### 7.3 Contract with SeedTelemetry

When telemetry is enabled, `signals_to_features()` in `ppo.py` appends `SeedTelemetry.to_features()` output:

```python
# ppo.py line 154
telemetry_features.extend(report.telemetry.to_features())
```

**Total with telemetry:** 50 (base) + 30 (3 slots x 10 telemetry features) = 80 dimensions.

This matches `vectorized.py` line 814 calculation.

---

## 8. Code Quality Assessment

### 8.1 Strengths

1. **Clear HOT PATH documentation** - Header warns about import constraints
2. **Good test coverage** - `test_features.py` covers main scenarios
3. **Type hints** - Return types and parameters annotated
4. **Defensive coding** - `safe()` handles edge cases
5. **Explicit feature layout** - Docstring documents index ranges

### 8.2 Areas for Improvement

1. **Stale docstring** - Returns says 35, should be 50
2. **Missing validation** - No runtime check that feature count matches constant
3. **Implicit normalization dependency** - Relies on external normalizer

---

## 9. Recommendations Summary

| Priority | Recommendation | Effort |
|----------|---------------|--------|
| P2 | Fix docstring (35 -> 50 floats) | 5 min |
| P3 | Add test validating blueprint mapping against enum | 30 min |
| P3 | Add assertion that `len(features) == MULTISLOT_FEATURE_SIZE` | 5 min |
| P4 | Consider pre-normalizing high-magnitude features | 2-4 hours |

---

## 10. Conclusion

`features.py` is a well-designed, focused module that serves its purpose as a hot-path feature extractor. The main risks are:

1. **Normalization inconsistency** - Mitigated by `RunningMeanStd` in the training loop
2. **Blueprint mapping drift** - Mitigated by existing tests, could be strengthened
3. **Documentation staleness** - Minor issue, easy fix

No PyTorch-specific concerns exist because this module is pure Python. The integration with the broader simic training infrastructure is clean and well-documented.

**Risk Rating: LOW**

The module is production-ready with the understanding that observation normalization is handled externally. The TODO comment at line 151-153 indicates awareness of the normalization contract issue, suggesting this is a deliberate architectural decision rather than an oversight.
