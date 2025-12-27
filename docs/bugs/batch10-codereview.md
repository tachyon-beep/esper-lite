# Batch 10 Code Review: Tamiyo Policy Abstraction Layer

**Reviewer:** Claude Opus 4.5 (Senior Code Reviewer)
**Date:** 2025-12-27
**Files Reviewed:** 9 files in `src/esper/tamiyo/policy/`

---

## Executive Summary

The Tamiyo Policy module implements a well-designed abstraction layer for policy implementations in the Esper morphogenetic AI framework. The code demonstrates strong software engineering practices with clear protocol definitions, proper type annotations, and comprehensive test coverage (117 tests, all passing).

**Overall Assessment:** PASS with minor findings

**Key Strengths:**
- Clean Protocol-based design avoids MRO conflicts with nn.Module
- Excellent separation between neural (LSTM) and heuristic policies
- Comprehensive action masking derived from single source of truth (VALID_TRANSITIONS)
- Well-documented design rationale throughout

**Issues Found:** 6 total (0 P0, 0 P1, 2 P2, 2 P3, 2 P4)

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/tamiyo/policy/protocol.py`

**Purpose:** Defines the `PolicyBundle` Protocol - the interface contract for all policy implementations (LSTM, heuristic, future MLP).

**Strengths:**
- `@runtime_checkable` enables validation at registration time
- Comprehensive method documentation with clear guidance on on-policy vs off-policy usage
- Explicit notes about `initial_hidden()` returning inference-mode tensors
- Clean separation: PolicyBundle receives features, not raw TrainingSignals

**Concerns:** None. This is a well-designed protocol.

---

### 2. `/home/john/esper-lite/src/esper/tamiyo/policy/types.py`

**Purpose:** Data classes for policy interface return types (`ActionResult`, `EvalResult`, `ForwardResult`).

**Strengths:**
- Frozen dataclasses with slots for memory efficiency
- Clear attribute documentation
- `op_logits` field in `ActionResult` supports telemetry/decision snapshots

**Concerns:** None.

---

### 3. `/home/john/esper-lite/src/esper/tamiyo/policy/registry.py`

**Purpose:** Policy registration and factory pattern for dynamic policy loading.

**Strengths:**
- Clean decorator-based registration
- Structural validation via hasattr (with explicit authorization comment)
- Clear error messages for missing policies/methods
- `clear_registry()` for testing isolation

**Concerns:**

**[P3-1] Registry validation cannot check method signatures**
Lines 50-80: The registry validates method presence via `hasattr` but cannot verify signatures. The docstring correctly notes this limitation and delegates to static type checking, which is appropriate. However, the list of `required_properties` does not include newer protocol properties like `slot_config`, `feature_dim`, `hidden_dim`, `network`, `compile`, and `is_compiled`.

```python
# Current (line 63-64)
required_properties = ['is_recurrent', 'supports_off_policy', 'device', 'dtype']

# Missing from validation:
# - slot_config, feature_dim, hidden_dim, network (added in protocol)
# - compile, is_compiled (torch.compile integration)
```

**Impact:** Low - these are still checked by static type checkers, but runtime registration would accept incomplete implementations.

**Recommendation:** Add missing properties to `required_properties` list or document why they are intentionally omitted.

---

### 4. `/home/john/esper-lite/src/esper/tamiyo/policy/factory.py`

**Purpose:** High-level factory function `create_policy()` for policy instantiation with torch.compile support.

**Strengths:**
- Single entry point with sensible defaults
- Proper device placement before compile (documented constraint)
- Validates compile_mode against allowed set
- Auto-computes feature_dim from slot_config

**Concerns:**

**[P4-1] Unused `num_slots` parameter creates confusion**
Line 23: `num_slots: int = 4` is marked as deprecated but still used as fallback when `slot_config` is None. The docstring says "deprecated - use slot_config instead" but the parameter remains.

```python
# Line 79-80
if slot_config is None:
    slot_config = SlotConfig.for_grid(rows=1, cols=num_slots)
```

**Impact:** Minimal - backward compatibility maintained, but parameter could confuse new users.

**Recommendation:** Consider removing `num_slots` in a future cleanup or making `slot_config` required.

---

### 5. `/home/john/esper-lite/src/esper/tamiyo/policy/action_masks.py`

**Purpose:** Action masking logic for multi-slot control. Only masks physically impossible actions, letting Tamiyo learn optimal timing.

**Strengths:**
- Stage sets (`_FOSSILIZABLE_STAGES`, `_PRUNABLE_STAGES`, `_ADVANCABLE_STAGES`) derived from `VALID_TRANSITIONS` - single source of truth
- Clear documentation of what gets masked and why
- `MaskedCategorical` with normalized entropy and validation toggle
- Comprehensive handling of alpha_mode constraints for PRUNE and SET_ALPHA_TARGET

**Concerns:**

**[P2-1] `build_slot_states` uses `.get()` for slot_reports lookup**
Line 102: Uses `.get()` which is fine for this use case (external data source), but the docstring says "slot_reports: Slot -> SeedStateReport (active slots only)" which suggests the caller should only pass slots that exist. If a slot_id is in `slots` but not in `slot_reports`, returning `None` is correct behavior for empty slots.

```python
# Line 102
report = slot_reports.get(slot_id)
```

**Impact:** None - this is legitimate handling of optional slot occupancy, not defensive programming hiding bugs.

**Verdict:** No change needed - this is appropriate.

---

**[P2-2] Potential performance concern in `compute_batch_masks`**
Lines 282-300: For each environment in the batch, `compute_action_masks` is called individually, then tensors are stacked. For large batch sizes (e.g., 128 envs), this creates many small tensors.

```python
masks_list = [
    compute_action_masks(...)
    for i, slot_states in enumerate(batch_slot_states)
]
return {
    key: torch.stack([m[key] for m in masks_list])
    for key in masks_list[0]
}
```

**Impact:** Medium - hot path for vectorized training. May cause GPU memory fragmentation with many small tensor allocations.

**Recommendation:** Consider pre-allocating batch tensors and filling them directly, similar to `batch_obs_to_features`. Add TODO comment if not addressing now.

---

### 6. `/home/john/esper-lite/src/esper/tamiyo/policy/features.py`

**Purpose:** Hot-path feature extraction from observations for RL training.

**Strengths:**
- Module docstring correctly warns about import restrictions (HOT PATH - leyline only)
- `batch_obs_to_features` uses vectorized tensor operations for base features
- `_BLUEPRINT_TO_INDEX` is duplicated for performance (documented design decision)
- Comprehensive feature layout documentation with exact dimensions

**Concerns:**

**[P3-2] `_BLUEPRINT_TO_INDEX` duplication risk**
Lines 125-139: Blueprint string-to-index mapping is duplicated from `BlueprintAction` enum for performance. This creates a sync risk if new blueprints are added.

```python
_BLUEPRINT_TO_INDEX = {
    "noop": 0,
    "conv_light": 1,
    # ... 13 entries total
}
_NUM_BLUEPRINT_TYPES = 13
```

The comment at line 123 correctly notes this is intentional for hot-path performance, but there's no automated test verifying sync with `BlueprintAction`.

**Impact:** Low - new blueprints require manual sync.

**Recommendation:** Add a test that verifies `_BLUEPRINT_TO_INDEX` matches `BlueprintAction.to_blueprint_id()` for all values.

---

**[P2-3] Per-slot feature extraction uses nested loops**
Lines 431-488: The per-slot features use nested `for slot_idx ... for env_idx` loops with individual tensor element writes. The TODO comment at lines 377-379 acknowledges this and provides optimization guidance.

```python
for slot_idx, slot_id in enumerate(slot_config.slot_ids):
    # ...
    for env_idx in range(n_envs):
        report = batch_slot_reports[env_idx].get(slot_id)
        # Individual element writes
        features[env_idx, offset] = 1.0
```

**Impact:** Medium - O(num_slots x num_envs) individual tensor writes. The existing TODO correctly captures the optimization path.

**Verdict:** The TODO is appropriate; no immediate action needed but the concern is documented.

---

### 7. `/home/john/esper-lite/src/esper/tamiyo/policy/lstm_bundle.py`

**Purpose:** LSTM-based recurrent policy bundle wrapping `FactoredRecurrentActorCritic`.

**Strengths:**
- Clean delegation to underlying network
- Auto-computes feature_dim from slot_config if not provided
- Proper handling of torch.compile wrapper via `_orig_mod` access (with authorization comments)
- `initial_hidden()` correctly documents inference-mode limitation

**Concerns:**

**[P4-2] `forward()` mask expansion could be clearer**
Lines 145-150: The `expand_mask` helper function silently handles None masks, but the conditional logic is slightly complex.

```python
def expand_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if need_expand and mask.dim() == 2:
        return mask.unsqueeze(1)
    return mask
```

**Impact:** Minimal - correct behavior, slightly dense code.

**Recommendation:** No change needed; code is correct.

---

### 8. `/home/john/esper-lite/src/esper/tamiyo/policy/heuristic_bundle.py`

**Purpose:** Adapter wrapping `HeuristicTamiyo` as a PolicyBundle for ablations and debugging.

**Strengths:**
- Clear documentation that this is NOT registered in the policy registry
- All non-applicable methods raise `NotImplementedError` with helpful messages
- Exposes underlying heuristic via `.heuristic` property for direct usage
- No-op implementations for optional methods (compile, gradient_checkpointing)

**Concerns:** None. This is a well-implemented adapter.

---

### 9. `/home/john/esper-lite/src/esper/tamiyo/policy/__init__.py`

**Purpose:** Package initialization and public API exports.

**Strengths:**
- Imports trigger LSTM policy registration
- Clear separation: neural policies via registry, heuristic via factory function
- Comprehensive `__all__` exports

**Concerns:** None.

---

## Cross-Cutting Integration Risks

### 1. Feature Dimension Consistency (LOW RISK)

The feature dimension is computed dynamically based on `slot_config`:
- `features.py`: `get_feature_size(slot_config) = BASE_FEATURE_SIZE + num_slots * SLOT_FEATURE_SIZE`
- `lstm_bundle.py`: Auto-computes feature_dim if not provided
- `factory.py`: Passes computed feature_dim to policy

All paths appear consistent. The integration test at `tests/integration/test_policy_bundle_integration.py` verifies this.

### 2. Action Head Size Consistency (LOW RISK)

Action head sizes are derived from:
- `leyline/factored_actions.py`: `NUM_BLUEPRINTS`, `NUM_OPS`, etc.
- `action_masks.py`: Uses same constants for mask tensor shapes
- `factored_lstm.py`: Uses `get_action_head_sizes(slot_config)` from leyline

All paths use leyline as single source of truth.

### 3. Stage Encoding Consistency (MEDIUM RISK)

Stage encoding uses two paths:
- `features.py`: `_STAGE_TO_INDEX` from `stage_schema.py` for one-hot encoding
- `action_masks.py`: `_FOSSILIZABLE_STAGES` etc. derived from `VALID_TRANSITIONS`

Both derive from leyline, but the frozensets in `action_masks.py` compare against `stage.value` (int), while `stage_schema.py` uses `SeedStage` enum directly. This is correct but requires care.

### 4. Blueprint Mapping Duplication (MEDIUM RISK)

The `_BLUEPRINT_TO_INDEX` dict in `features.py` must stay in sync with `BlueprintAction` enum. Currently there's no test verifying this. See P3-2.

---

## Findings Summary

| ID | Severity | File | Description |
|----|----------|------|-------------|
| P3-1 | P3 | registry.py | Missing properties in protocol validation |
| P4-1 | P4 | factory.py | Deprecated `num_slots` parameter still present |
| P2-2 | P2 | action_masks.py | Batch mask computation allocates many small tensors |
| P3-2 | P3 | features.py | `_BLUEPRINT_TO_INDEX` duplication risk |
| P2-3 | P2 | features.py | Per-slot nested loops (documented TODO) |
| P4-2 | P4 | lstm_bundle.py | Dense mask expansion logic (no change needed) |

---

## Test Coverage Assessment

The test suite in `tests/tamiyo/policy/` is comprehensive:

- **test_action_masks.py**: 67 tests covering all masking scenarios, edge cases, batch operations, and MaskedCategorical validation
- **test_features.py**: 14 tests for feature extraction including one-hot encoding, normalization, dynamic slot configs
- **test_heuristic_bundle.py**: 9 tests for heuristic adapter
- **test_lstm_bundle.py**: 14 tests for LSTM bundle including get_action, evaluate_actions, state_dict, device management
- **test_protocol.py**: 5 tests for protocol definition
- **test_registry.py**: 8 tests for registration and factory

**Missing Test Coverage:**
1. No test verifying `_BLUEPRINT_TO_INDEX` sync with `BlueprintAction`
2. No performance benchmarks for batch feature extraction (acceptable - not a correctness issue)

---

## Recommendations

### Immediate (P2)

1. **P2-2/P2-3**: The TODO at line 377 in `features.py` already captures the optimization need. Consider adding a similar TODO for `compute_batch_masks` in `action_masks.py`:

```python
# TODO: [FUTURE OPTIMIZATION] - Pre-allocate batch tensors to reduce small tensor allocations
```

### Short-Term (P3)

2. **P3-1**: Add missing properties to registry validation:

```python
required_properties = [
    'is_recurrent', 'supports_off_policy', 'device', 'dtype',
    'slot_config', 'feature_dim', 'hidden_dim', 'network',
    'is_compiled',
]
required_methods.append('compile')  # Already a required method
```

3. **P3-2**: Add a test to verify `_BLUEPRINT_TO_INDEX` consistency:

```python
def test_blueprint_index_sync():
    """Verify _BLUEPRINT_TO_INDEX matches BlueprintAction enum."""
    from esper.tamiyo.policy.features import _BLUEPRINT_TO_INDEX
    from esper.leyline import BlueprintAction

    for bp in BlueprintAction:
        bp_id = bp.to_blueprint_id()
        assert bp_id in _BLUEPRINT_TO_INDEX
        assert _BLUEPRINT_TO_INDEX[bp_id] == bp.value
```

### Low Priority (P4)

4. **P4-1**: Document or remove the deprecated `num_slots` parameter in a future cleanup PR.

---

## Conclusion

The Tamiyo Policy module is well-designed and implemented. The Protocol-based approach provides clean abstraction without nn.Module MRO conflicts. Action masking correctly derives from the single source of truth (`VALID_TRANSITIONS`), and feature extraction handles the hot-path requirements appropriately.

The identified issues are minor and do not affect correctness. The existing TODO comments demonstrate awareness of optimization opportunities. The test suite is comprehensive with 117 passing tests.

**Recommendation:** Approve for merge. Address P2/P3 findings in follow-up PRs as time permits.
