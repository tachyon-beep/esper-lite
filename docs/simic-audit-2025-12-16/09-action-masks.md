# Simic Audit: action_masks.py

**File:** `/home/john/esper-lite/src/esper/simic/action_masks.py`
**Date:** 2025-12-16
**Auditor:** Claude Code (PyTorch Engineering Specialist)

---

## Executive Summary

The `action_masks.py` module implements action masking for the factored multi-slot control system. It provides two core capabilities:

1. **Mask Computation** - `compute_action_masks()` and `compute_batch_masks()` for determining physically valid actions
2. **Masked Distribution** - `MaskedCategorical` for safe sampling with proper entropy normalization

Overall, this module is **well-designed** with appropriate torch.compile considerations. However, there are several issues ranging from minor code quality concerns to moderate integration risks.

**Severity Summary:**
- Critical (P0): 0
- High (P1): 1
- Medium (P2): 3
- Low (P3): 4
- Informational: 3

---

## 1. torch.compile Compatibility

### 1.1 Validation Guard with @torch.compiler.disable [P3 - LOW]

**Location:** Lines 265-278

```python
@torch.compiler.disable
def _validate_action_mask(mask: torch.Tensor) -> None:
    """Validate that at least one action is valid per batch element."""
    valid_count = mask.sum(dim=-1)
    if (valid_count == 0).any():
        raise InvalidStateMachineError(...)
```

**Analysis:** The `@torch.compiler.disable` decorator is correctly applied to isolate the CPU synchronization (`.any()`) from compiled regions. This is the proper pattern for safety checks that require data-dependent control flow.

**Assessment:** Correct usage. The docstring explicitly acknowledges the synchronization cost trade-off.

---

### 1.2 compute_action_masks() - Not Compilable (By Design) [INFO]

**Location:** Lines 104-184

**Analysis:** The `compute_action_masks()` function uses Python control flow over slot dictionaries and frozensets. This is intentional - the function operates on Python-level state machine logic, not tensor operations.

```python
# Python control flow - not meant for torch.compile
for slot_id in ordered:
    seed_info = slot_states.get(slot_id)
    if seed_info is not None:
        stage = seed_info.stage
        if stage in _FOSSILIZABLE_STAGES:
            op_mask[LifecycleOp.FOSSILIZE] = True
```

**Assessment:** Acceptable. Mask computation happens at the Python level before tensor operations. The compiled network receives pre-computed boolean tensors.

---

### 1.3 MaskedCategorical.entropy() - Potential Graph Break Risk [P2 - MEDIUM]

**Location:** Lines 327-339

```python
def entropy(self) -> torch.Tensor:
    probs = self._dist.probs
    log_probs = self._dist.logits - self._dist.logits.logsumexp(dim=-1, keepdim=True)
    raw_entropy = -(probs * log_probs * self.mask).sum(dim=-1)
    num_valid = self.mask.sum(dim=-1).clamp(min=1)
    max_entropy = torch.log(num_valid)
    return raw_entropy / max_entropy.clamp(min=1e-8)
```

**Issue:** The `num_valid.clamp(min=1)` followed by `torch.log()` is a data-dependent operation that could cause graph breaks when masks have all zeros (though validation should prevent this).

**Risk:** When used inside `FactoredRecurrentActorCritic.evaluate_actions()` (which is compiled), this entropy calculation participates in the backward pass. The `clamp(min=1)` on `num_valid` is safe, but if any mask slips through validation with zeros, the `log(0)` would produce `-inf`.

**Recommendation:** The pattern is sound given the validation guard, but consider adding an assertion in tests that validates entropy is finite for all expected mask configurations.

---

## 2. Device Placement

### 2.1 Consistent Device Handling [INFO - POSITIVE]

**Location:** Lines 109, 129, 134, 141, 145, 148

```python
device = device or torch.device("cpu")
slot_mask = torch.zeros(NUM_SLOTS, dtype=torch.bool, device=device)
```

**Assessment:** Device handling is consistent. All tensors are created on the specified device with a sensible default.

---

### 2.2 MaskedCategorical Device Inference [P3 - LOW]

**Location:** Lines 306-310

```python
mask_value = torch.tensor(
    max(finfo_min, -1e4),
    device=logits.device,
    dtype=logits.dtype,
)
```

**Issue:** A new scalar tensor is created on each `MaskedCategorical` instantiation. While this correctly uses `logits.device`, it allocates a new tensor per call.

**Impact:** Minor memory/performance cost. In batched evaluation (e.g., PPO update), this is called per head per batch.

**Recommendation:** Consider hoisting the mask value to a class constant or using `torch.full_like` pattern:

```python
self.masked_logits = logits.masked_fill(mask < 0.5, -1e4)
```

This avoids the scalar tensor allocation entirely since `masked_fill` accepts a Python scalar.

---

## 3. Gradient Flow

### 3.1 MaskedCategorical Preserves Gradients [INFO - POSITIVE]

**Location:** Lines 288-312

**Analysis:** The masking approach preserves gradient flow correctly:

```python
self.masked_logits = logits.masked_fill(mask < 0.5, mask_value)
self._dist = Categorical(logits=self.masked_logits)
```

- `masked_fill` is differentiable (masked positions get zero gradient)
- Log probabilities flow through valid actions only
- No detach() or gradient-breaking operations

**Assessment:** Correct. Gradients flow through valid actions; masked actions receive zero gradient as expected.

---

### 3.2 Entropy Normalization Gradient Consideration [P2 - MEDIUM]

**Location:** Lines 327-339

```python
num_valid = self.mask.sum(dim=-1).clamp(min=1)
max_entropy = torch.log(num_valid)
return raw_entropy / max_entropy.clamp(min=1e-8)
```

**Issue:** The `num_valid` computation involves the mask, which is typically a boolean tensor not requiring gradients. The division by `max_entropy` normalizes the scale but this normalization factor varies per sample based on the mask.

**Potential Problem:** In PPO updates, entropy loss aggregates across batch. If mask validity varies significantly across the batch, the normalized entropy has different "scales" for different samples. This is intentional per the docstring but may interact unexpectedly with entropy coefficient tuning.

**Assessment:** This is a design choice, not a bug. The docstring correctly explains the normalization rationale. However, users should be aware that `entropy_coef` operates on normalized entropy in [0, 1], not raw nats.

---

## 4. Memory Considerations

### 4.1 compute_batch_masks() Memory Efficiency [P2 - MEDIUM]

**Location:** Lines 187-227

```python
masks_list = [
    compute_action_masks(
        slot_states=slot_states,
        enabled_slots=enabled_slots,
        ...
    )
    for i, slot_states in enumerate(batch_slot_states)
]
return {
    key: torch.stack([m[key] for m in masks_list])
    for key in masks_list[0]
}
```

**Issue:** This creates `batch_size * 4` intermediate tensors before stacking. For large batches, this is inefficient.

**Impact:** With typical batch sizes (64-256), this creates 256-1024 small tensors. Each tensor is `NUM_SLOTS`, `NUM_BLUEPRINTS`, `NUM_BLENDS`, or `NUM_OPS` in size (3, 5, 3, 4 respectively).

**Performance:** For production PPO training with `num_envs=4` and `episode_length=25`, this is called once per rollout, so impact is minimal. However, if scaled to larger batch sizes, this could become a bottleneck.

**Recommendation:** For high-throughput scenarios, consider vectorized mask computation:

```python
# Pre-allocate and fill
op_mask = torch.zeros(batch_size, NUM_OPS, dtype=torch.bool, device=device)
op_mask[:, LifecycleOp.WAIT] = True  # WAIT always valid
# ... vectorized logic
```

---

### 4.2 MaskSeedInfo Dataclass Memory [P3 - LOW]

**Location:** Lines 67-76

```python
@dataclass(frozen=True, slots=True)
class MaskSeedInfo:
    stage: int  # SeedStage.value
    seed_age_epochs: int
```

**Assessment:** Good use of `slots=True` for memory efficiency. The `frozen=True` ensures immutability. Using `int` instead of `SeedStage` enum is documented as torch.compile safety.

---

## 5. Integration Risks

### 5.1 _CULLABLE_STAGES vs VALID_TRANSITIONS Synchronization [P1 - HIGH]

**Location:** Lines 56-64

```python
# Stages from which a seed can be culled
# Derived as: set(active_stages) - {FOSSILIZED} (terminal success)
# See stages.py VALID_TRANSITIONS for authoritative source
_CULLABLE_STAGES = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
    SeedStage.PROBATIONARY.value,
})
```

**Issue:** This is a **manual duplication** of logic from `leyline/stages.py:VALID_TRANSITIONS`. The comment acknowledges this but doesn't programmatically enforce synchronization.

**Risk:** If `VALID_TRANSITIONS` changes (e.g., new stage added with CULLED transition), `_CULLABLE_STAGES` may become stale, causing:
- Seeds in new cullable stages cannot be culled (mask incorrectly blocks)
- Or seeds in stages no longer cullable can still be culled (mask incorrectly allows)

**Evidence of Design Awareness:** The comment explicitly states "See stages.py VALID_TRANSITIONS for authoritative source" - but awareness without enforcement is fragile.

**Recommendation:** Derive `_CULLABLE_STAGES` programmatically at module load:

```python
from esper.leyline.stages import VALID_TRANSITIONS, SeedStage

_CULLABLE_STAGES = frozenset(
    stage.value
    for stage, transitions in VALID_TRANSITIONS.items()
    if SeedStage.CULLED in transitions
)
```

This ensures synchronization with the authoritative source.

---

### 5.2 _FOSSILIZABLE_STAGES Hardcoded [P3 - LOW]

**Location:** Lines 49-52

```python
_FOSSILIZABLE_STAGES = frozenset({
    SeedStage.PROBATIONARY.value,
})
```

**Issue:** Same pattern as `_CULLABLE_STAGES` - hardcoded derivation from `VALID_TRANSITIONS`.

**Recommendation:** Derive programmatically:

```python
_FOSSILIZABLE_STAGES = frozenset(
    stage.value
    for stage, transitions in VALID_TRANSITIONS.items()
    if SeedStage.FOSSILIZED in transitions
)
```

---

### 5.3 MaskedCategorical Mask Comparison with < 0.5 [P3 - LOW]

**Location:** Line 311

```python
self.masked_logits = logits.masked_fill(mask < 0.5, mask_value)
```

**Issue:** The mask is documented as "1.0 = valid, 0.0 = invalid" but the code compares with `< 0.5`. While this works, it assumes the mask is numeric (float).

**Context:** Looking at `compute_action_masks()`, masks are created as `dtype=torch.bool`. The `< 0.5` comparison auto-casts bool to float, which works but is subtle.

**Recommendation:** Use explicit boolean comparison or ensure mask dtype consistency:

```python
# Option A: Explicit boolean
self.masked_logits = logits.masked_fill(~mask, mask_value)

# Option B: Keep current but document the cast
```

The docstring says "1.0 = valid, 0.0 = invalid" which implies float, but actual usage is bool. Minor inconsistency.

---

### 5.4 Network Integration - Duplicate Masking Logic [INFO]

**Location:** In `tamiyo_network.py` lines 213-220 vs `MaskedCategorical` lines 305-312

**Observation:** The network's `forward()` applies masking directly:

```python
# tamiyo_network.py
slot_logits = slot_logits.masked_fill(~slot_mask, _MASK_VALUE)
```

While `MaskedCategorical` also applies masking:

```python
# action_masks.py
self.masked_logits = logits.masked_fill(mask < 0.5, mask_value)
```

**Impact:** When `MaskedCategorical` is used after `forward()`, masking is applied twice. This is harmless (masked_fill on already-masked values is idempotent for the same mask value) but indicates potential code path confusion.

**Assessment:** The dual masking exists because:
1. `forward()` masks for direct logit inspection/value computation
2. `MaskedCategorical` masks for distribution construction

This is acceptable but worth documenting.

---

## 6. Code Quality

### 6.1 __all__ Exports All Public APIs [INFO - POSITIVE]

**Location:** Lines 245-253

```python
__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "slot_id_to_index",
    "MaskedCategorical",
    "InvalidStateMachineError",
]
```

**Assessment:** Complete and accurate exports.

---

### 6.2 Module Docstring is Excellent [INFO - POSITIVE]

**Location:** Lines 1-17

The module docstring clearly explains:
- What is masked (physically impossible actions only)
- Specific rules for each mask type
- What is NOT masked (timing heuristics)
- Multi-slot execution semantics

This is exemplary documentation.

---

### 6.3 Type Annotations are Complete [INFO - POSITIVE]

All functions have complete type annotations including:
- Return types
- Optional parameters with defaults
- TYPE_CHECKING import for `SeedStateReport`

---

### 6.4 Unused Import: Categorical [P3 - LOW - MINOR]

**Location:** Line 25

```python
from torch.distributions import Categorical
```

**Issue:** `Categorical` is imported at module level but only used inside `MaskedCategorical.__init__`. This is fine for direct usage but could be a lazy import for faster module load.

**Assessment:** Minor. The import is used and correct.

---

## 7. Test Coverage Assessment

**Test File:** `/home/john/esper-lite/tests/simic/test_action_masks.py`

**Coverage Analysis:**

| Function | Test Coverage |
|----------|---------------|
| `compute_action_masks()` | Comprehensive |
| `compute_batch_masks()` | Good |
| `build_slot_states()` | Good |
| `slot_id_to_index()` | Basic |
| `MaskedCategorical` | NOT TESTED |
| `InvalidStateMachineError` | NOT TESTED |

**Gap:** `MaskedCategorical` has no direct tests. It is tested indirectly through `tamiyo_network.py` usage, but should have unit tests for:
- Entropy normalization correctness
- Behavior with edge-case masks (1 valid action, all valid actions)
- InvalidStateMachineError raised when no valid actions
- Gradient flow through masked distribution

---

## 8. Recommendations Summary

### Must Fix (P1)

1. **Derive `_CULLABLE_STAGES` and `_FOSSILIZABLE_STAGES` from `VALID_TRANSITIONS`**
   - Eliminates synchronization risk with authoritative state machine

### Should Fix (P2)

2. **Add unit tests for `MaskedCategorical`**
   - Entropy normalization edge cases
   - InvalidStateMachineError trigger
   - Gradient flow verification

3. **Clarify mask dtype in `MaskedCategorical` docstring**
   - Document whether bool or float is expected
   - Update `< 0.5` to `~mask` if bool is canonical

### Consider (P3)

4. **Use `masked_fill` with scalar instead of tensor for mask_value**
   - Avoids tensor allocation per instantiation

5. **Document dual masking pattern**
   - Explain network forward() masking vs MaskedCategorical masking relationship

---

## 9. Conclusion

The `action_masks.py` module is well-implemented with appropriate torch.compile considerations. The main concern is the hardcoded stage sets that could drift from the authoritative `VALID_TRANSITIONS` source. The `MaskedCategorical` class correctly implements masked sampling with proper entropy normalization for reinforcement learning.

The module follows project conventions (no legacy code, proper typing, leyline integration) and has good test coverage for mask computation functions, though `MaskedCategorical` deserves dedicated unit tests.

**Overall Risk Rating:** LOW-MEDIUM

The high-priority finding (`_CULLABLE_STAGES` synchronization) is a latent bug that would only manifest if the stage state machine evolves. Given the project's "no legacy code" policy, any stage machine changes would likely touch this file anyway, but programmatic derivation is strictly safer.
