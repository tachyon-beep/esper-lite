# Batch 10 Code Review: Tamiyo Policy Abstraction Layer

**Reviewer Specialization:** PyTorch Engineering
**Files Reviewed:** 9 files in `/src/esper/tamiyo/policy/`
**Date:** 2025-12-27

---

## Executive Summary

The Tamiyo Policy module implements a clean abstraction layer for swappable policy implementations in a deep RL system. The code demonstrates strong PyTorch 2.x patterns including proper torch.compile integration, inference_mode usage, and thoughtful LSTM hidden state management. The codebase is well-structured with comprehensive test coverage.

**Overall Quality:** High. The policy abstraction is well-designed with proper separation of concerns between the protocol, registry, factory, and implementations.

**Critical Issues Found:** 0 P0, 2 P1, 4 P2, 6 P3, 3 P4

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/tamiyo/policy/action_masks.py`

**Purpose:** Computes action masks for the factored action space. Only masks physically impossible actions (not timing heuristics). Includes `MaskedCategorical` distribution for safe action sampling.

**PyTorch Highlights:**
- Proper use of `@torch.compiler.disable` decorator to isolate validation logic from compilation
- Correct device placement via explicit `device` parameter
- MASKED_LOGIT_VALUE (-1e4) is FP16/BF16 safe (avoids overflow from finfo.min)
- Proper boolean tensor creation with explicit `dtype=torch.bool`

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P2** | `build_slot_states()` (lines 87-111) uses `.get()` for slot_reports lookup. While this is a dict lookup (not defensive programming), if `slot_reports` should ALWAYS contain all slot IDs, direct access would fail faster on bugs. Current pattern silently returns None for missing slots. |
| **P3** | `_validate_action_mask` and `_validate_logits` use `.any()` and `.sum()` which trigger CPU sync. The comment acknowledges this, but in high-frequency training these could be bottlenecks. Consider making validation opt-in only in debug mode. |
| **P4** | `slot_id_to_index` (lines 303-337) imports inside the function body. Move to module level to avoid repeated import overhead on each call. |

**Code Quality:**
- Excellent docstrings explaining the masking semantics
- Good use of frozenset for stage validation sets
- Clean separation between single-env and batch mask computation

---

### 2. `/home/john/esper-lite/src/esper/tamiyo/policy/factory.py`

**Purpose:** Factory function for creating configured PolicyBundle instances with optional torch.compile.

**PyTorch Highlights:**
- Correct compile ordering: `.to(device)` BEFORE `.compile()` - this is critical for proper tracing
- Valid compile mode validation against VALID_COMPILE_MODES
- Proper dynamic=True flag for varying batch/sequence lengths

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P1** | **Potential device placement issue after compile** (lines 98-104): The warning comment says "Do not call .to(device) after this" but there's no enforcement. If a caller does `policy = create_policy(...).to(other_device)`, the compiled module would be replaced. Consider adding a guard or using a frozen wrapper. |
| **P3** | compile_mode validation (lines 71-75) could use an Enum instead of string matching for type safety. |

**Code Quality:**
- Clear docstrings with usage examples
- Local imports to avoid circular dependency are properly documented

---

### 3. `/home/john/esper-lite/src/esper/tamiyo/policy/features.py`

**Purpose:** Hot-path feature extraction from observations for RL training. Converts semantic observations to flat tensor features.

**PyTorch Highlights:**
- Good vectorization in `batch_obs_to_features` with pre-allocated tensors
- Proper `.clamp()` usage for numerical stability
- Explicit device placement via `device` parameter

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P2** | **Nested loops for slot features** (lines 431-489): The TODO comment acknowledges this is O(num_slots x num_envs) with individual element writes. For larger slot configurations (25 slots), this could become a bottleneck. Consider pre-extracting to contiguous arrays. |
| **P2** | **Memory allocation per-call** (lines 388-406): Multiple `torch.tensor()` calls create intermediate tensors. Consider pre-allocating a workspace buffer or using `torch.stack` with generators. |
| **P3** | `obs_to_multislot_features` returns `list[float]` instead of tensor (lines 143-350). The comment explains this is for flexibility during construction, but forcing an early tensor conversion could enable vectorized operations upstream. |
| **P3** | **Potential precision loss** (lines 270-291): Multiple float divisions for normalization. Consider using `torch.tensor.div_()` for in-place operations when building the batch tensor. |

**Code Quality:**
- Excellent documentation of feature layout with explicit index ranges
- Good use of local constants for hot-path optimization (`_BLUEPRINT_TO_INDEX`, `_STAGE_TO_INDEX`)
- Clear separation between base features and per-slot features

---

### 4. `/home/john/esper-lite/src/esper/tamiyo/policy/heuristic_bundle.py`

**Purpose:** Rule-based heuristic policy wrapper for ablations and debugging. Raises NotImplementedError for most PolicyBundle methods.

**PyTorch Highlights:**
- Correct stateless behavior (no hidden state, CPU-only)
- Proper no-op implementations for inapplicable methods

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P3** | Class doesn't actually implement `PolicyBundle` protocol despite the comment. It's explicitly NOT registered. The comment is accurate, but the class structure mimics PolicyBundle which could be confusing. Consider using a separate ABC or clearly documenting this is an adapter pattern. |
| **P4** | `is_compiled` always returns False (line 203), but the class could theoretically be subclassed. Consider making it a property that checks for network existence. |

**Code Quality:**
- Clear documentation that this is NOT a full PolicyBundle implementation
- Good pattern for wrapping non-neural policies

---

### 5. `/home/john/esper-lite/src/esper/tamiyo/policy/__init__.py`

**Purpose:** Package initialization with policy registration and re-exports.

**PyTorch Highlights:**
- Import triggers LSTM registration (side-effect import pattern)
- Clean re-export of public API

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P4** | Side-effect import of `lstm_bundle` (line 42) is documented but could be surprising. Consider explicit registration via a `register()` function call for clarity. |

**Code Quality:**
- Well-organized `__all__` list
- Good docstring explaining import behavior

---

### 6. `/home/john/esper-lite/src/esper/tamiyo/policy/lstm_bundle.py`

**Purpose:** LSTM-based recurrent policy implementing PolicyBundle protocol.

**PyTorch Highlights:**
- **Excellent** use of `@torch.inference_mode()` on `get_value` and `initial_hidden`
- Proper handling of torch.compile wrapper via `_orig_mod` access
- Correct dimension handling (2D vs 3D input normalization)
- Good pattern for mask expansion to match feature dimensions

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P1** | **Potential gradient graph leakage** (lines 261-275): `initial_hidden` is decorated with `@torch.inference_mode()` which is correct for rollout collection, but the docstring warning may not be sufficient. If these tensors are accidentally used in `evaluate_actions`, they'll silently not contribute gradients. Consider adding a runtime check or using a marker attribute. |
| **P2** | **Compiled module type safety** (lines 356-358): After `torch.compile`, `self._network` becomes an `OptimizedModule` but the type hint remains `FactoredRecurrentActorCritic`. The `# type: ignore[assignment]` suppresses this, but consider using a Union type or Protocol for better static analysis. |
| **P3** | `expand_mask` helper (lines 145-150) is defined inside `forward()`. This creates a new function object on every call. Consider moving to a module-level helper or using a lambda. |

**Code Quality:**
- Clear documentation of inference vs training mode distinctions
- Proper authorization comments for getattr/hasattr usage
- Good separation between action selection and evaluation paths

---

### 7. `/home/john/esper-lite/src/esper/tamiyo/policy/protocol.py`

**Purpose:** PolicyBundle Protocol definition for swappable policy implementations.

**PyTorch Highlights:**
- `@runtime_checkable` enables isinstance checks
- Clear documentation of torch.compile guidance
- Proper typing for device and dtype properties

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P3** | Protocol methods use `...` (ellipsis) bodies which is correct, but some methods have complex semantics that could benefit from default implementations or abstract base class pattern. Consider adding a `PolicyBundleBase` ABC with shared utilities. |

**Code Quality:**
- Excellent design rationale documentation
- Clear on-policy vs off-policy method grouping
- Good warning about torch.compile location (in Simic, not PolicyBundle)

---

### 8. `/home/john/esper-lite/src/esper/tamiyo/policy/registry.py`

**Purpose:** Policy registration and factory pattern for PolicyBundle implementations.

**PyTorch Highlights:**
- No direct PyTorch usage (pure Python registry pattern)

**Concerns:**

| Severity | Finding |
|----------|---------|
| **P3** | `hasattr` check (lines 72-73) is authorized but only checks structural presence, not signatures. The docstring is excellent about this limitation. Consider adding type stub validation via `typing.get_type_hints` for better protocol compliance checking. |

**Code Quality:**
- Comprehensive error messages with available alternatives
- Clean separation between registration and instantiation
- Good `clear_registry()` for testing

---

### 9. `/home/john/esper-lite/src/esper/tamiyo/policy/types.py`

**Purpose:** Dataclass definitions for PolicyBundle return types (ActionResult, EvalResult, ForwardResult).

**PyTorch Highlights:**
- Proper use of frozen dataclasses with slots
- Correct typing for torch.Tensor fields
- Optional op_logits for telemetry with None default

**Concerns:**

None. This file is clean and well-structured.

**Code Quality:**
- Minimal, focused dataclass definitions
- Good documentation of field semantics

---

## Cross-Cutting Integration Risks

### 1. Feature Dimension Synchronization

**Risk Level:** Medium

The feature dimension is computed in multiple places:
- `features.py::get_feature_size(slot_config)` - canonical source
- `lstm_bundle.py` auto-computes from slot_config if feature_dim is None
- `factory.py` calls `get_feature_size()` when state_dim is None

**Concern:** If slot_config changes between feature extraction and policy construction, dimension mismatch could cause silent failures or shape errors at runtime.

**Recommendation:** Add a dimension validation check in the forward path that compares input features.shape[-1] against expected feature_dim.

### 2. Mask Shape Consistency

**Risk Level:** Low

Masks are computed in `action_masks.py` and consumed by `lstm_bundle.py` and the underlying network. The shape conventions are:
- Single-env: `[batch, action_dim]`
- Sequence: `[batch, seq_len, action_dim]`

The code handles both cases via `.unsqueeze()` expansion, but there's no centralized validation that masks match feature batch dimensions.

### 3. torch.compile Interaction with Hidden States

**Risk Level:** Medium

The LSTM hidden state handling has a subtle interaction:
- `initial_hidden()` returns inference-mode tensors (non-differentiable)
- `evaluate_actions()` expects None for hidden to create fresh gradient-compatible states
- If compiled, the hidden state shapes must be stable (dynamic=True helps but isn't guaranteed)

**Recommendation:** Add explicit documentation in the compile path about hidden state handling, or add a check that warns if inference-mode tensors are passed to training methods.

### 4. Blueprint Mapping Synchronization

**Risk Level:** Low

`features.py::_BLUEPRINT_TO_INDEX` duplicates the mapping from `leyline/factored_actions.py::BlueprintAction`. The comment justifies this for performance, but there's no compile-time validation that they stay synchronized.

**Recommendation:** Add a module-level assertion that validates `_BLUEPRINT_TO_INDEX` matches `BlueprintAction.to_blueprint_id()` for all values.

---

## Severity-Tagged Findings List

### P1 - Correctness Bugs

1. **factory.py:98-104** - No enforcement against `.to(device)` after compile. Could silently break compiled module.

2. **lstm_bundle.py:261-275** - `initial_hidden()` returns inference-mode tensors that silently won't contribute gradients if misused in training.

### P2 - Performance Issues

1. **action_masks.py:87-111** - `build_slot_states()` uses dict.get() which silently returns None for missing slots instead of failing fast.

2. **features.py:431-489** - Nested loops for slot features are O(num_slots x num_envs) with individual element writes.

3. **features.py:388-406** - Multiple `torch.tensor()` calls create intermediate tensors; could be batched.

4. **lstm_bundle.py:356-358** - Compiled module type safety suppressed with `# type: ignore`.

### P3 - Code Quality

1. **action_masks.py** - Validation uses `.any()/.sum()` causing CPU sync; should be debug-only.

2. **factory.py:71-75** - String-based compile_mode validation could use Enum.

3. **features.py:143-350** - `obs_to_multislot_features` returns list[float] instead of tensor.

4. **features.py:270-291** - Multiple float divisions for normalization; could use in-place ops.

5. **heuristic_bundle.py** - Class mimics PolicyBundle but isn't registered; could be confusing.

6. **lstm_bundle.py:145-150** - `expand_mask` helper defined inside method creates new function per call.

7. **protocol.py** - Protocol uses ellipsis bodies; could benefit from shared base class.

8. **registry.py:72-73** - hasattr check only validates structure, not signatures.

### P4 - Style/Minor

1. **action_masks.py:325** - `slot_id_to_index` imports inside function body.

2. **heuristic_bundle.py:203** - `is_compiled` always returns False; could check for network.

3. **__init__.py:42** - Side-effect import for registration could be explicit function call.

---

## Recommendations Summary

1. **Add feature dimension validation** at the forward path entry point to catch mismatches early.

2. **Add blueprint mapping synchronization assertion** in features.py to validate against factored_actions.py.

3. **Consider making MaskedCategorical validation opt-in** via environment variable or configuration for production performance.

4. **Document or enforce the compile-before-device-move constraint** in factory.py more strictly.

5. **Add hidden state mode markers** to distinguish inference-mode tensors from gradient-compatible tensors.

---

## Test Coverage Assessment

The test files (`test_action_masks.py`, `test_features.py`, `test_lstm_bundle.py`) provide excellent coverage:

- Action mask edge cases (empty slots, all disabled, large grids)
- Feature extraction with various slot configurations
- LSTM bundle integration with registry
- MaskedCategorical numerical stability (FP16, BF16)

**Missing Coverage:**
- torch.compile interaction tests (would require CUDA)
- Stress tests for batch feature extraction performance
- Cross-module integration tests validating dimension consistency

---

*Report generated by PyTorch Engineering Specialist*
