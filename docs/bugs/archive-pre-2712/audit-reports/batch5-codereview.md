# Batch 5 Code Review: Simic Attribution + Control

**Reviewer:** Python Code Quality Specialist
**Date:** 2025-12-27
**Files Reviewed:**
1. `/home/john/esper-lite/src/esper/simic/attribution/counterfactual_helper.py`
2. `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py`
3. `/home/john/esper-lite/src/esper/simic/attribution/__init__.py`
4. `/home/john/esper-lite/src/esper/simic/control/__init__.py`
5. `/home/john/esper-lite/src/esper/simic/control/normalization.py`

---

## Executive Summary

This batch covers two key subsystems:
1. **Attribution Module**: Counterfactual analysis for computing Shapley values and marginal contributions of seeds
2. **Control Module**: Observation and reward normalization for RL training stability

Overall quality is **good**. The code is well-documented, follows project conventions, and demonstrates solid understanding of the domain (game-theoretic attribution, Welford's online algorithm). Test coverage appears comprehensive. However, there are several findings ranging from potential correctness issues to minor code quality improvements.

**Critical Issues:** 0
**Important Issues:** 3
**Suggestions:** 8

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/simic/attribution/counterfactual_helper.py`

**Purpose:** Bridge between CounterfactualEngine and training loop. Provides a simple interface for computing counterfactual attribution during training.

**Strengths:**
- Clean wrapper pattern with appropriate separation of concerns
- Good docstrings explaining usage patterns
- Telemetry integration via callback pattern (respects DI principles)
- `compute_simple_ablation` utility for lightweight alternative

**Findings:**

#### P2: Unused `epoch` Parameter in `compute_contributions` and `compute_contributions_from_results`

**Location:** Lines 95, 108
**Issue:** The `epoch` parameter is accepted but never used. This suggests either dead code or incomplete implementation.

```python
def compute_contributions_from_results(
    self,
    slot_ids: list[str],
    results: dict[tuple[bool, ...], tuple[float, float]],
    epoch: int | None = None,  # NEVER USED
) -> dict[str, ContributionResult]:
```

**Impact:** Confusing API - callers may think epoch is being recorded or used for telemetry.
**Recommendation:** Either remove the parameter or pass it through to `CounterfactualMatrix.epoch` for proper telemetry emission.

#### P3: Potential Inefficiency in `_process_matrix` Shapley Condition

**Location:** Lines 143-144
**Issue:** The condition `len(matrix.configs) > len(slot_ids) + 1` determines whether to compute Shapley values, but this heuristic may not be optimal.

```python
if len(matrix.configs) > len(slot_ids) + 1:
    shapley_values = self.engine.compute_shapley_values(matrix)
```

**Rationale:** For 2 slots, full factorial = 4 configs, but `2 + 1 = 3`, so Shapley is computed. For ablation_only with 2 slots: 4 configs (all on, each off, all off), which equals `2 + 2`, barely passing. This seems correct but could be clearer.

**Recommendation:** Add a comment explaining the rationale for this threshold.

#### P4: Empty TYPE_CHECKING Block

**Location:** Lines 29-30
```python
if TYPE_CHECKING:
    pass
```

**Impact:** Minor clutter.
**Recommendation:** Remove the empty block.

---

### 2. `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py`

**Purpose:** Core engine for computing counterfactual attribution, including full factorial, Shapley sampling, and interaction terms.

**Strengths:**
- Excellent docstring explaining the distinction between removal cost and causal contribution
- Proper timeout handling between configs (CUDA-safe pattern)
- Antithetic sampling for variance reduction in Shapley computation
- Frozen config dataclass prevents mid-run mutations
- Comprehensive interaction term computation

**Findings:**

#### P1: Non-Deterministic Shapley Values Due to Unseeded RNG

**Location:** Lines 351, 400-402
**Issue:** `random.shuffle(perm)` uses the global random state, making Shapley values non-reproducible. This can cause:
1. Flaky tests
2. Irreproducible attribution results between runs
3. Difficulty debugging attribution anomalies

```python
for _ in range(n_samples // 2):
    perm = list(range(n))
    random.shuffle(perm)  # Uses global state!
```

**Impact:** Research reproducibility is compromised. Two runs with identical inputs may produce different Shapley values.

**Recommendation:** Accept an optional `seed` parameter or use a local `random.Random()` instance:

```python
def _generate_shapley_configs(
    self, slot_ids: list[str], rng: random.Random | None = None
) -> list[tuple[bool, ...]]:
    rng = rng or random.Random()  # Or use self._rng initialized in __init__
```

#### P1: Variance Calculation Uses Population Formula, Claims Sample

**Location:** Lines 423-427
**Issue:** The code computes population variance (`/ len(values)`) but this is appropriate for permutation sampling where we're estimating an expectation, not inferring from a sample. However, the comment references sample variance:

```python
variance = (
    sum((v - mean) ** 2 for v in values) / len(values)
    if len(values) > 1
    else 0.0
)
```

**Impact:** Not a bug per se (both formulas converge), but the inconsistency between code and comments in `ShapleyEstimate` (line 155: confidence intervals imply sample variance) could cause confusion.

**Recommendation:** Clarify in comments whether this is intentional. For confidence intervals in `is_significant`, sample variance (`n-1` denominator) would be more statistically correct.

#### P2: `compute_matrix_from_results` Does Not Set `compute_time_seconds`

**Location:** Lines 280-306
**Issue:** When using pre-computed results, `compute_time_seconds` defaults to 0.0. This makes logging at line 152-154 in the helper misleading:

```python
_logger.debug(
    f"Counterfactual computed: {len(matrix.configs)} configs, "
    f"{matrix.compute_time_seconds:.2f}s"  # Always 0.0 for pre-computed
)
```

**Recommendation:** Accept an optional `compute_time_seconds` parameter or document that this field is only valid for `compute_matrix()`.

#### P3: Magic Number for Interaction Term Threshold

**Location:** Lines 174-176
**Issue:** The regime classification uses hardcoded thresholds:

```python
if self.interaction > 0.5:  # Threshold for "significant"
    return "synergy"
elif self.interaction < -0.5:
```

**Impact:** These thresholds may not be appropriate for all accuracy scales (e.g., if accuracy is 0-1 vs 0-100).

**Recommendation:** Either normalize by baseline accuracy or make thresholds configurable.

#### P3: Defensive `.get()` in `marginal_contribution`

**Location:** Line 123
```python
return self._marginal_contributions.get(slot_id, 0.0)
```

**Question:** Per CLAUDE.md, this could be hiding a bug if `slot_id` is incorrectly spelled or the slot wasn't in the original computation. However, returning 0.0 for unknown slots may be legitimate (graceful degradation for partial matrices).

**Recommendation:** Consider whether this should raise KeyError for unknown slots to fail-fast on integration bugs.

#### P4: Unused Import in TYPE_CHECKING Block

**Location:** Lines 29-30
```python
if TYPE_CHECKING:
    pass
```

**Impact:** Minor clutter.
**Recommendation:** Remove.

---

### 3. `/home/john/esper-lite/src/esper/simic/attribution/__init__.py`

**Purpose:** Public API surface for the attribution submodule.

**Strengths:**
- Clean `__all__` export list
- Appropriate symbols exposed

**Findings:**

#### P4: Missing `InteractionTerm` Export

**Location:** Lines 19-27
**Issue:** `InteractionTerm` dataclass is defined in `counterfactual.py` but not exported. Users calling `get_interaction_terms()` receive `InteractionTerm` objects without access to the type.

```python
__all__ = [
    "CounterfactualEngine",
    "CounterfactualConfig",
    "CounterfactualMatrix",
    "ShapleyEstimate",
    # Missing: "InteractionTerm", "CounterfactualResult"
    ...
]
```

**Recommendation:** Add `InteractionTerm` and `CounterfactualResult` to exports if they're part of the public API.

---

### 4. `/home/john/esper-lite/src/esper/simic/control/__init__.py`

**Purpose:** Public API surface for the control submodule.

**Strengths:**
- Clean, minimal interface

**Findings:**

No issues found. The module correctly exports `RunningMeanStd` and `RewardNormalizer`.

---

### 5. `/home/john/esper-lite/src/esper/simic/control/normalization.py`

**Purpose:** GPU-native observation and reward normalization using Welford's algorithm.

**Strengths:**
- Excellent documentation including thread safety notes
- GPU-native operations to avoid CPU sync
- EMA option for long training stability
- Proper Welford implementation with law of total variance for EMA
- Comprehensive checkpointing support

**Findings:**

#### P1: Type Annotation Mismatch in `state_dict` Return Type

**Location:** Lines 229, 152
**Issue:** `RewardNormalizer.state_dict()` returns `dict[str, float | int]` but `load_state_dict` expects the same. However, `count` could be a float due to `epsilon` initialization... wait, no:

```python
self.count = 0  # Line 193 - starts at 0 (int)
self.count += 1  # Line 205 - increments by 1 (remains int)
```

Actually this is correct. But in `RunningMeanStd`:

```python
self.count = torch.tensor(epsilon, device=device)  # Line 56 - float!
```

The `RunningMeanStd.state_dict` returns `dict[str, torch.Tensor]` which is correct.

No actual issue here upon closer inspection.

#### P2: Device String Type Inconsistency

**Location:** Lines 58, 139-145
**Issue:** `_device` is stored as `str` but `device` property returns `torch.device`. The `to()` method accepts `str | torch.device` but stores as `str(device)`:

```python
def to(self, device: str | torch.device) -> "RunningMeanStd":
    ...
    self._device = str(device)  # Normalizes to string
    return self

@property
def device(self) -> torch.device:
    return self.mean.device  # Returns torch.device
```

This is actually fine since `mean.device` gives the canonical `torch.device`. The `_device` field seems redundant since we can always get device from tensors.

**Minor Recommendation:** Consider removing `_device` field since `self.mean.device` is authoritative.

#### P3: `load_state_dict` Uses `_device` Which May Be Stale

**Location:** Lines 160-164
**Issue:** When loading state, tensors are moved to `self._device`:

```python
def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self.mean = state["mean"].to(self._device)
    self.var = state["var"].to(self._device)
    self.count = state["count"].to(self._device)
```

If `_device` was never updated (e.g., object created then `load_state_dict` called without `to()`), this works correctly since `_device` defaults to constructor arg.

However, if state was saved from a GPU normalizer and loaded into a CPU-initialized normalizer, the tensors stay on CPU. This is probably correct behavior, but could be surprising.

**Recommendation:** Document that `load_state_dict` moves tensors to the normalizer's current device, not the saved device.

#### P4: EMA Variance Update Comment Could Be Clearer

**Location:** Lines 92-98
The comment is excellent and explains the math, but the variable name `delta` is computed before the mean update, then used after. This is correct per the comment but could be clearer:

```python
# The cross-term must be computed BEFORE updating the mean.
delta = batch_mean - self.mean
self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
# delta is still the OLD delta, which is correct
self.var = (
    self.momentum * self.var
    + (1 - self.momentum) * batch_var
    + self.momentum * (1 - self.momentum) * delta ** 2
)
```

**Recommendation:** Rename to `delta_old` or add inline comment clarifying this is intentional.

---

## Cross-Cutting Concerns

### Test Coverage Analysis

Tests exist for:
- `RunningMeanStd`: Comprehensive unit tests, property-based tests for convergence, bounds, multi-dimensional, edge cases
- `RewardNormalizer`: Unit tests including edge cases (first sample, count starts at 0)
- `CounterfactualEngine`: Telemetry emission tests, callback injection tests

**Gaps Identified:**
1. No tests for `compute_simple_ablation` helper function
2. No tests for `compute_interaction_terms` functionality
3. No tests for timeout handling in `compute_matrix`
4. No tests for EMA mode in `RunningMeanStd`
5. No property-based tests for Shapley value correctness (e.g., efficiency axiom: sum of Shapley values = full value - baseline)

### Integration Risks

1. **CounterfactualHelper in ParallelEnvState**: The `counterfactual_helper` field is optional (`| None`). Code using it must null-check, which is appropriate.

2. **Telemetry Contract**: The `AnalyticsSnapshotPayload` has the `shapley_values` field defined correctly. The emission in `counterfactual.py` uses `shapley_values=shapley_dict` which matches the payload contract.

3. **Leyline Contract Compliance**: All telemetry types are imported from `esper.leyline.telemetry`. The `TelemetryEventType.ANALYTICS_SNAPSHOT` is used correctly.

### API Design Quality

The layered design is clean:
- `CounterfactualEngine`: Pure computation engine, no state storage
- `CounterfactualHelper`: Training-loop integration, caches `_last_matrix`
- `CounterfactualMatrix`: Result container with lazy-computed derived metrics

This follows the Single Responsibility Principle well.

---

## Severity-Tagged Findings Summary

### P1 - Important (Should Fix)

| ID | File | Line | Issue |
|----|------|------|-------|
| 1 | counterfactual.py | 351, 400 | Non-deterministic Shapley due to unseeded RNG |
| 2 | counterfactual.py | 423 | Variance formula/comment inconsistency (population vs sample) |
| 3 | (withdrawn) | - | Type annotation issue - upon review, no actual issue |

### P2 - Performance/Correctness Risk

| ID | File | Line | Issue |
|----|------|------|-------|
| 4 | counterfactual_helper.py | 95, 108 | Unused `epoch` parameter |
| 5 | counterfactual.py | 280-306 | `compute_time_seconds` always 0.0 for pre-computed |
| 6 | normalization.py | 58, 139 | `_device` field potentially redundant |

### P3 - Code Quality

| ID | File | Line | Issue |
|----|------|------|-------|
| 7 | counterfactual.py | 174-176 | Magic numbers for interaction regime thresholds |
| 8 | counterfactual.py | 123 | Defensive `.get()` may hide integration bugs |
| 9 | normalization.py | 160-164 | `load_state_dict` device handling documentation |
| 10 | normalization.py | 92-98 | EMA delta variable name could be clearer |
| 11 | counterfactual_helper.py | 143 | Shapley condition threshold lacks explanation |

### P4 - Style/Minor

| ID | File | Line | Issue |
|----|------|------|-------|
| 12 | counterfactual_helper.py | 29-30 | Empty TYPE_CHECKING block |
| 13 | counterfactual.py | 29-30 | Empty TYPE_CHECKING block |
| 14 | __init__.py (attribution) | 19-27 | Missing `InteractionTerm`/`CounterfactualResult` exports |

---

## Recommendations Summary

### High Priority

1. **Add RNG seeding for Shapley reproducibility**: Accept a `seed` or `rng` parameter in `CounterfactualConfig` or `generate_shapley_configs`.

2. **Remove or use `epoch` parameter**: Either wire it through to matrix creation or remove from API.

### Medium Priority

3. **Add tests for untested functionality**:
   - `compute_simple_ablation`
   - `compute_interaction_terms`
   - EMA mode in `RunningMeanStd`
   - Timeout handling

4. **Document `compute_time_seconds` limitation** for pre-computed matrices.

### Low Priority

5. **Clean up empty TYPE_CHECKING blocks**
6. **Export `InteractionTerm` and `CounterfactualResult`** if they're public API
7. **Consider removing redundant `_device` field** in RunningMeanStd
8. **Add comments explaining magic numbers** (interaction thresholds, Shapley condition)

---

## Conclusion

The attribution and control modules are well-implemented with good documentation and appropriate design patterns. The main concerns are around reproducibility (unseeded RNG for Shapley), API completeness (unused parameters, missing exports), and test coverage gaps. The normalization code is particularly solid with excellent handling of numerical stability and GPU/CPU device management.

No critical bugs were found. The P1 issues are important for research reproducibility but do not affect training correctness.
