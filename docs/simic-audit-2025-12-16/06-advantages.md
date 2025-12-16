# Simic Audit: advantages.py

**File:** `/home/john/esper-lite/src/esper/simic/advantages.py`
**Date:** 2025-12-16
**Auditor:** Claude Code (claude-opus-4-5-20251101)

## Executive Summary

The `advantages.py` module implements per-head advantage computation with causal masking for Tamiyo's factored action space. The code is clean, well-documented, and serves a focused purpose. However, there are several PyTorch-specific concerns worth addressing, ranging from compile compatibility to memory efficiency.

**Overall Risk Assessment:** LOW - Well-designed utility with minor optimization opportunities.

---

## 1. Code Overview

### Purpose
Computes per-head advantages with causal masking for Tamiyo's 4-head factored action space (op, slot, blueprint, blend). The masking logic ensures that heads which had no causal effect on the outcome (e.g., `blueprint_head` when `op=WAIT`) receive zero advantage signal, reducing gradient noise.

### Key Function
```python
def compute_per_head_advantages(
    base_advantages: torch.Tensor,  # [batch] or [batch, seq]
    op_actions: torch.Tensor,       # [batch] or [batch, seq]
) -> dict[str, torch.Tensor]:       # Per-head masked advantages
```

---

## 2. torch.compile Compatibility

### Current Status: GOOD (with caveats)

**Strengths:**
- Uses pure tensor operations (comparison, multiplication, boolean logic)
- No Python control flow dependent on tensor values
- No data-dependent branching

**Concerns:**

| Issue | Severity | Description |
|-------|----------|-------------|
| `dict` return type | LOW | Returning a dict triggers graph breaks in some torch.compile backends. The current call site in `ppo.py` accesses keys inside a Python loop, so the graph is already broken there anyway. |
| `base_advantages.clone()` | LOW | Creates an unnecessary copy. See memory section. |

**Recommendation:** The function is torch.compile-safe in isolation. Consider marking with `@torch.compiler.disable` comment to document that it runs inside an already-broken graph region (the PPO update loop).

---

## 3. Device Placement

### Current Status: SAFE

**Analysis:**
- No explicit device placement in the module
- All operations are tensor-to-tensor and preserve device of inputs
- Input tensors arrive from `TamiyoRolloutBuffer.get_batched_sequences()` which explicitly moves data to the target device
- Enum comparisons (`op_actions == LifecycleOp.WAIT`) work correctly because `LifecycleOp` is an `IntEnum` - PyTorch handles the scalar comparison on the correct device

**Call site verification (ppo.py:444-447):**
```python
valid_op_actions = data["op_actions"][valid_mask]  # Already on self.device
per_head_advantages = compute_per_head_advantages(
    valid_advantages, valid_op_actions  # Both on same device
)
```

**Risk:** NONE - Device handling is inherited from inputs correctly.

---

## 4. Gradient Flow

### Current Status: CORRECT (but nuanced)

**Analysis:**
The causal masking design is sound for credit assignment:

| Head | Causal Relevance |
|------|------------------|
| `op` | Always (decides which operation) |
| `slot` | GERMINATE, FOSSILIZE, CULL (not WAIT) |
| `blueprint` | GERMINATE only |
| `blend` | GERMINATE only |

**Gradient implications:**
1. Masking via multiplication by 0.0 zeros out the advantage signal
2. When `ratio * adv = ratio * 0.0`, the policy gradient term contributes nothing
3. This is correct behavior - heads with zero causal effect should not be updated

**Edge case (Severity: INFO):**
Episodes with only WAIT actions result in `blueprint_head` and `blend_head` receiving zero gradients for the entire episode. This is documented in the test (`test_only_wait_episode_sparse_gradients`) and is correct behavior, but could theoretically cause staleness if WAIT-heavy episodes dominate. The current PPO implementation handles this fine.

---

## 5. Memory Analysis

### Current Status: ACCEPTABLE (minor inefficiency)

**Memory pattern:**
```python
op_advantages = base_advantages.clone()  # Copy #1 (unnecessary)
slot_advantages = base_advantages * slot_mask.float()  # Allocation #2
blueprint_advantages = base_advantages * blueprint_mask.float()  # Allocation #3
blend_advantages = base_advantages * blend_mask.float()  # Allocation #4
```

| Issue | Severity | Description |
|-------|----------|-------------|
| Unnecessary clone | LOW | `op_advantages` could reuse `base_advantages` directly since it's never modified. The clone is defensive but wasteful. |
| Boolean-to-float conversion | LOW | `slot_mask.float()` allocates a new tensor. Could use `where()` or in-place multiply. |

**Memory overhead:**
- 4 tensors of shape `[batch]` or `[batch, seq]`
- At typical batch sizes (100-400 valid timesteps), this is ~1.6KB-6.4KB for float32
- Negligible compared to network activations

**Recommendation:** Keep as-is. The clarity benefit outweighs the minor memory cost. If profiling shows this as a hotspot (unlikely), consider:
```python
# Zero-allocation alternative (less readable)
op_advantages = base_advantages  # No clone needed
slot_advantages = torch.where(is_wait, torch.zeros_like(base_advantages), base_advantages)
```

---

## 6. Integration Risks

### 6.1 LifecycleOp Enum Coupling

**Risk:** MEDIUM (Maintainability)

The module imports `LifecycleOp` and hardcodes the causal structure:
```python
is_wait = op_actions == LifecycleOp.WAIT
is_germinate = op_actions == LifecycleOp.GERMINATE
```

**Concern:** If `LifecycleOp` values change (e.g., adding new operations), this module needs updating. The comment block (lines 7-23) documents the decision tree but could become stale.

**Mitigations:**
- The enum uses `IntEnum` with explicit values
- Tests cover all current operations
- The decision tree comment provides context

**Recommendation:** Add a module-level assertion or test to verify enum coverage:
```python
# Ensure all LifecycleOp values are handled
assert set(LifecycleOp) == {LifecycleOp.WAIT, LifecycleOp.GERMINATE,
                            LifecycleOp.FOSSILIZE, LifecycleOp.CULL}
```

### 6.2 Return Type Contract

**Risk:** LOW

The function returns `dict[str, torch.Tensor]` with keys `["op", "slot", "blueprint", "blend"]`. The consumer (ppo.py) iterates over these exact keys:
```python
for key in ["slot", "blueprint", "blend", "op"]:
    adv = per_head_advantages[key]
```

**Concern:** Key mismatch would cause silent bugs (KeyError) or missed advantages.

**Recommendation:** Consider a `TypedDict` or `NamedTuple` for stronger typing:
```python
from typing import TypedDict

class PerHeadAdvantages(TypedDict):
    op: torch.Tensor
    slot: torch.Tensor
    blueprint: torch.Tensor
    blend: torch.Tensor
```

---

## 7. Code Quality

### Strengths
1. **Excellent documentation** - Module docstring explains the full decision tree
2. **Clear naming** - `is_wait`, `is_germinate`, `slot_mask` are self-documenting
3. **Focused responsibility** - Single function, single purpose
4. **Test coverage** - Comprehensive tests in `tests/simic/test_advantages.py` covering all operations and edge cases

### Style Observations

| Item | Status | Note |
|------|--------|------|
| Type hints | GOOD | Full annotations on function signature |
| Docstring | GOOD | Args and Returns documented |
| Import organization | GOOD | Standard library, then torch, then local |
| `__all__` export | GOOD | Explicitly exports public API |

### Minor Suggestions

1. **Shape documentation:** The docstring says `[batch] or [batch, seq]` but doesn't specify which dimension is which in 2D case. Consider:
   ```python
   Args:
       base_advantages: GAE advantages [B] for flat batch or [B, T] for sequences
   ```

2. **Unused variable:** `is_germinate` is used twice (for blueprint and blend). Could create aliases for clarity:
   ```python
   # blueprint and blend share the same causal relevance
   blueprint_mask = blend_mask = is_germinate
   ```

---

## 8. Summary of Findings

| Category | Severity | Issue | Recommendation |
|----------|----------|-------|----------------|
| torch.compile | INFO | Dict return type causes graph break | Document that this runs in broken-graph region |
| Memory | LOW | Unnecessary `clone()` for op_advantages | Keep for clarity (negligible cost) |
| Memory | LOW | Boolean-to-float allocations | Keep for clarity |
| Integration | MEDIUM | LifecycleOp coupling | Add enum coverage assertion |
| Integration | LOW | Dict key contract | Consider TypedDict for stronger typing |
| Code Quality | INFO | Shape documentation | Clarify dimension semantics |

---

## 9. Conclusion

The `advantages.py` module is well-designed and correctly implements causal masking for factored PPO. The code is clear, testable, and integrates cleanly with the PPO training loop.

**No blocking issues found.**

The main improvement opportunity is defensive programming around the `LifecycleOp` coupling to ensure new operations are explicitly handled. The memory and compile observations are informational - the current implementation prioritizes readability over micro-optimization, which is the correct trade-off for this performance-uncritical code path.
