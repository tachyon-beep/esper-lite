# Batch 10 Code Review: Tamiyo Policy - Policy Abstraction Layer

**Reviewer:** DRL Specialist
**Date:** 2025-12-27
**Files Reviewed:** 9 files in `src/esper/tamiyo/policy/`

---

## Executive Summary

The policy abstraction layer is **well-designed and implements DRL best practices correctly**. The factored action space with masked categorical distributions, the PolicyBundle protocol pattern, and the LSTM integration are all sound. The code demonstrates good understanding of:

- Action masking for factored action spaces (critical for RL with constraints)
- Normalized entropy for fair exploration incentives across varying action restrictions
- Proper separation between inference-mode (rollout) and gradient-tracking (training) paths
- LSTM hidden state management for recurrent policies

**Critical findings:** 0 P0 issues
**Important findings:** 2 P1 issues (one potential correctness bug in features.py)
**Moderate findings:** 5 P2 issues
**Minor findings:** 6 P3 issues
**Style findings:** 3 P4 issues

---

## File-by-File Analysis

### 1. `action_masks.py` (483 lines)

**Purpose:** Computes action masks for the factored action space, ensuring only physically possible actions are selectable. Implements `MaskedCategorical` for safe sampling from masked distributions.

**Strengths:**
- Derives valid stage sets from `VALID_TRANSITIONS` (single source of truth)
- Clear docstring explaining what IS and IS NOT masked (only physical impossibility, not timing heuristics)
- `MaskedCategorical.entropy()` returns normalized entropy [0,1] with excellent docstring explaining coefficient calibration
- Validation toggle (`MaskedCategorical.validate`) allows production performance without graph breaks from `.any()` calls
- Uses `MASKED_LOGIT_VALUE = -1e4` from leyline (safe for FP16/BF16)

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| AM-1 | P3 | `build_slot_states` uses `.get()` on `slot_reports` dict (line 102). This is legitimate since `slot_reports` may not contain all slots - empty slots have no report. However, the function name could be clearer (e.g., `build_slot_mask_states`). |
| AM-2 | P3 | `_FOSSILIZABLE_STAGES` and `_PRUNABLE_STAGES` are derived at module import time from `VALID_TRANSITIONS`. If `VALID_TRANSITIONS` changes, this is automatically correct. Good pattern. |
| AM-3 | P4 | `compute_batch_masks` creates individual masks then stacks - for large batch sizes this could be vectorized, but the comment notes this is "once per rollout step" so low impact. |
| AM-4 | P2 | `MaskedCategorical.entropy()` computes `log_probs` via `logits - logsumexp` then uses `probs * log_probs`. This is numerically stable but slightly redundant with the internal Categorical computation. Minor perf concern on hot path. |

### 2. `factory.py` (110 lines)

**Purpose:** Factory function `create_policy()` for instantiating policies with torch.compile support.

**Strengths:**
- Validates `compile_mode` against allowed values upfront
- Proper ordering: device placement BEFORE compile (critical for trace correctness)
- Warning comment about not calling `.to(device)` after compile

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| FA-1 | P3 | `num_slots` parameter is marked deprecated but still has a default value of 4. Consider removing the default and requiring explicit `slot_config` to encourage migration. |
| FA-2 | P4 | The config dict built on lines 88-93 passes `state_dim` as `feature_dim`. The comment explains this, but the naming could confuse maintainers. |

### 3. `features.py` (558 lines)

**Purpose:** Hot-path feature extraction from observations for RL training. Converts raw observations to flat tensor features.

**Strengths:**
- Explicit hot-path warning at module docstring - only leyline imports
- `safe()` function handles None/inf/nan gracefully with clamping
- Feature layout clearly documented with index ranges
- `batch_obs_to_features` is partially vectorized with TODO for full optimization
- `TaskConfig` provides task-specific normalization (CIFAR vs TinyStories)

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| FE-1 | **P1** | **Blueprint-to-index mapping duplicated from leyline.** Lines 125-140 define `_BLUEPRINT_TO_INDEX` which duplicates `BlueprintAction` enum values. The comment says "duplicated for performance" but if `BlueprintAction` gains new members, this will silently produce incorrect one-hot encodings. Should at minimum have a module-level assertion like the ones in `factored_actions.py`. |
| FE-2 | P2 | `obs_to_multislot_features` returns `list[float]` instead of tensor. The docstring explains this is for efficiency (batch conversion later), but the function is 150+ lines with complex nested logic. Consider splitting into smaller functions. |
| FE-3 | P2 | `batch_obs_to_features` uses nested loops with individual element writes (lines 435-488). The TODO mentions vectorization could give 2-4x speedup. For large batch sizes this could be material. |
| FE-4 | P3 | `_DEBUG_STAGE_VALIDATION` uses `assert` which can be disabled by `-O`. Should use explicit if/raise for validation-critical code. |
| FE-5 | P3 | Many `.get()` calls in feature extraction (lines 270-343). The P3 Audit comment explains these are legitimate optional fields for inactive slots, but the code could benefit from a typed SlotFeatureInput dataclass to make field expectations explicit. |

### 4. `heuristic_bundle.py` (218 lines)

**Purpose:** Wrapper making `HeuristicTamiyo` look like a PolicyBundle for compatibility, but most methods raise NotImplementedError.

**Strengths:**
- Clear documentation that this is NOT a neural policy
- Correctly raises NotImplementedError for all RL-specific methods
- Provides `heuristic` property for direct access to underlying decision-maker

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| HB-1 | P3 | `HeuristicPolicyBundle` does not implement `PolicyBundle` protocol (it's intentionally NOT registered). The docstring says this but the class signature could confuse static analysis. Consider adding `# type: ignore[misc]` or explicit non-protocol marker. |

### 5. `__init__.py` (97 lines)

**Purpose:** Package entry point, triggers policy registration on import.

**Strengths:**
- Clear comment about import triggering registration
- Provides `create_heuristic_policy()` factory as the recommended way to create heuristics
- Clean `__all__` export list

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| IN-1 | P4 | The import `from esper.tamiyo.policy import lstm_bundle as _lstm_bundle` is solely for side effects (registration). The `# noqa: F401` is appropriate but a comment explaining the pattern would help. |

### 6. `lstm_bundle.py` (369 lines)

**Purpose:** LSTM-based PolicyBundle implementation wrapping `FactoredRecurrentActorCritic`.

**Strengths:**
- Auto-computes `feature_dim` from `slot_config` if not provided (prevents dimension drift)
- Proper separation: `get_action()` uses `inference_mode`, `evaluate_actions()` tracks gradients
- `initial_hidden()` docstring warns about inference-mode tensors being non-differentiable
- `state_dict`/`load_state_dict` correctly unwrap compiled modules via `_orig_mod`
- `compile()` is idempotent (safe to call twice)

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| LB-1 | **P1** | **`get_value()` uses `inference_mode` decorator but value bootstrap during rollout needs this.** However, if someone calls `get_value()` during training expecting gradients, they'll get silent failures. The decorator is correct for the intended use case, but the method should document this more prominently. |
| LB-2 | P2 | `forward()` has logic to expand 2D inputs to 3D (lines 140-163). This duplicates logic in `FactoredRecurrentActorCritic.get_action()`. Consider consolidating dimension normalization. |
| LB-3 | P2 | The `is_compiled` property uses `hasattr(self._network, '_orig_mod')`. This is the standard torch.compile detection pattern and has AUTHORIZED comment, but it's fragile if torch internals change. Consider adding a fallback check. |
| LB-4 | P3 | `dropout` parameter in `__init__` is documented as "currently unused by network". Should either implement or remove to avoid confusion. |

### 7. `protocol.py` (289 lines)

**Purpose:** Defines the `PolicyBundle` Protocol interface for swappable policy implementations.

**Strengths:**
- Excellent documentation explaining design rationale
- `@runtime_checkable` enables validation at registration time
- Clear separation of on-policy (PPO) vs off-policy (SAC) methods
- Documents that `network` property is an intentional abstraction leak for training infrastructure
- `initial_hidden()` docstring explicitly warns about inference-mode tensors

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| PR-1 | P3 | The protocol requires `compile()` method, but the semantics of what happens if compile fails are not documented. Should specify whether implementations should catch and handle compile errors or let them propagate. |
| PR-2 | P4 | `hidden_dim` property says "For non-recurrent policies, return 0 or raise NotImplementedError." Should pick one for consistency. |

### 8. `registry.py` (127 lines)

**Purpose:** Policy registration and factory pattern.

**Strengths:**
- Decorator-based registration is clean and Pythonic
- Extensive docstring explaining structural check limitations vs static type checking
- `clear_registry()` provided for testing

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| RE-1 | P2 | The structural check (hasattr for required methods/properties) cannot validate signatures. A policy with wrong method signature would pass registration but fail at runtime. The docstring acknowledges this and delegates to mypy, which is reasonable. |
| RE-2 | P3 | No mechanism to unregister a single policy (only `clear_registry()`). Could cause issues if reloading modules during development. |

### 9. `types.py` (68 lines)

**Purpose:** Dataclass types for PolicyBundle return values.

**Strengths:**
- All dataclasses are `frozen=True, slots=True` (immutable, memory-efficient)
- Clear field documentation
- `op_logits` in `ActionResult` is optional with sensible default

**Concerns:**

| ID | Severity | Issue |
|----|----------|-------|
| TY-1 | P4 | `EvalResult.entropy` is `dict[str, torch.Tensor]` but docstring doesn't specify shape. Should document `[batch, seq_len]` like `log_prob`. |

---

## Cross-Cutting Integration Risks

### 1. Feature Dimension Synchronization (Medium Risk)

The feature dimension is computed by `get_feature_size(slot_config)` which depends on:
- `BASE_FEATURE_SIZE = 23`
- `SLOT_FEATURE_SIZE = 39` per slot
- Number of slots from `slot_config`

If `SLOT_FEATURE_SIZE` changes (e.g., adding new per-slot features), the network input dimension must be updated. Currently:
- `LSTMPolicyBundle.__init__` auto-computes if `feature_dim` is None (good)
- `create_policy()` passes computed `state_dim` (good)
- **Risk:** Manual instantiation with explicit `feature_dim` could drift

**Recommendation:** Add a `validate_feature_dim()` helper that checks at runtime.

### 2. Blueprint Encoding Drift (High Risk - P1)

**FE-1** identified that `_BLUEPRINT_TO_INDEX` in features.py is duplicated from leyline's `BlueprintAction`. If the enum changes:
- `features.py` would produce incorrect one-hot encodings
- The network would see garbage features for new blueprints
- Silent failure - no runtime error, just wrong behavior

**Recommendation:** Add assertion at import time:
```python
from esper.leyline import BlueprintAction
assert len(_BLUEPRINT_TO_INDEX) == len(BlueprintAction), "Blueprint index drift"
for bp in BlueprintAction:
    assert _BLUEPRINT_TO_INDEX.get(bp.to_blueprint_id()) == bp.value, f"Mismatch for {bp}"
```

### 3. Action Mask and Network Head Size Alignment (Low Risk)

Action masks are computed with dimensions from `SlotConfig` and leyline constants. Network heads use the same `get_action_head_sizes(slot_config)`. These should stay in sync because both use the same source of truth, but there's no cross-validation.

### 4. LSTM Hidden State Gradient Tracking (Low Risk)

The code correctly uses `inference_mode()` for rollout collection and expects `hidden=None` for training (network creates fresh hidden states). However:
- `initial_hidden()` returns inference-mode tensors
- If accidentally passed to `evaluate_actions()`, gradients won't flow through initial state
- **Mitigation:** The docstring warns about this extensively

### 5. Heuristic vs Neural Policy Path (Low Risk)

`HeuristicPolicyBundle` is intentionally NOT registered in the policy registry. The training loop must special-case heuristic handling. This is documented but could confuse new developers.

---

## DRL-Specific Observations

### Action Masking Correctness

The action masking implementation correctly:
1. **Masks only physical impossibility** - timing heuristics are learned via rewards
2. **Uses optimistic masking for multi-slot** - operation valid if ANY slot allows it
3. **Derives valid transitions from state machine** - `_FOSSILIZABLE_STAGES` etc. from `VALID_TRANSITIONS`
4. **Handles alpha controller state** - PRUNE blocked if alpha_mode != HOLD (unless governor override)

### Entropy Handling

`MaskedCategorical.entropy()` correctly:
1. Computes entropy only over valid actions (masked actions excluded)
2. Normalizes to [0, 1] by dividing by log(num_valid_actions)
3. Returns 0 when only one action is valid (no uncertainty)
4. Documents the coefficient calibration (0.05 appropriate for normalized entropy)

This is important for PPO's entropy bonus - without normalization, states with fewer valid actions would appear to have lower entropy even when the policy is maximally uncertain.

### Recurrent Policy Design

The LSTM integration follows best practices:
1. **Feature extraction before LSTM** - reduces dimensionality
2. **LayerNorm pre- and post-LSTM** - stabilizes training
3. **Hidden state management** - proper separation of inference vs training modes
4. **Forget gate bias = 1** - improves long-term memory (Gers et al., 2000)

### Off-Policy Support (Not Implemented)

Both `LSTMPolicyBundle` and protocol correctly:
1. Mark `supports_off_policy = False` for LSTM (needs R2D2 machinery)
2. Raise `NotImplementedError` for `get_q_values()` and `sync_from()`
3. Document that off-policy would need MLP-based policy

---

## Findings Summary

### P0 - Critical (0 findings)
None.

### P1 - Important (2 findings)

| ID | File | Issue |
|----|------|-------|
| FE-1 | features.py | Blueprint-to-index mapping duplicated without sync assertion |
| LB-1 | lstm_bundle.py | `get_value()` uses inference_mode, needs clearer documentation about gradient-free context |

### P2 - Moderate (5 findings)

| ID | File | Issue |
|----|------|-------|
| AM-4 | action_masks.py | Minor redundancy in entropy computation |
| FE-2 | features.py | Large function could be split |
| FE-3 | features.py | Nested loops in batch feature extraction |
| LB-2 | lstm_bundle.py | Dimension normalization duplicated |
| LB-3 | lstm_bundle.py | torch.compile detection fragile |
| RE-1 | registry.py | Structural check can't validate signatures |

### P3 - Minor (6 findings)

| ID | File | Issue |
|----|------|-------|
| AM-1 | action_masks.py | Function name could be clearer |
| FA-1 | factory.py | Deprecated param still has default |
| FE-4 | features.py | assert can be disabled by -O |
| FE-5 | features.py | Many .get() calls, consider typed input |
| HB-1 | heuristic_bundle.py | Non-protocol class could confuse analysis |
| LB-4 | lstm_bundle.py | Unused dropout parameter |
| PR-1 | protocol.py | Compile failure semantics undocumented |
| RE-2 | registry.py | No single-policy unregister |

### P4 - Style (3 findings)

| ID | File | Issue |
|----|------|-------|
| AM-3 | action_masks.py | Batch masks not vectorized |
| FA-2 | factory.py | state_dim vs feature_dim naming |
| IN-1 | __init__.py | Side-effect import could use comment |
| PR-2 | protocol.py | hidden_dim behavior inconsistent |
| TY-1 | types.py | entropy shape not documented |

---

## Test Coverage Assessment

The test file `tests/tamiyo/policy/test_action_masks.py` (1262 lines) provides excellent coverage:
- All lifecycle stages tested
- Boundary conditions (MIN_PRUNE_AGE, seed limit)
- Multi-slot semantics (optimistic masking)
- SlotConfig integration (2, 3, 5, 9, 25 slots)
- `MaskedCategorical` edge cases (NaN, inf, all-false mask)
- Validation toggle behavior
- FP16/BF16 numerical stability

Missing coverage:
- `features.py` extraction logic (separate test file needed)
- `LSTMPolicyBundle` end-to-end with real network
- `registry.py` error cases

---

## Recommendations

1. **Immediate (P1):** Add blueprint index sync assertion in features.py
2. **High Priority (P2):** Document `get_value()` inference context more clearly
3. **Medium Priority (P3):** Add integration test for PolicyBundle with live network forward pass
4. **Low Priority (P4):** Clean up deprecated parameters and unused kwargs

---

## Conclusion

The policy abstraction layer is **production-ready** with strong DRL fundamentals. The one critical issue (FE-1) should be addressed before adding new blueprints. The overall architecture cleanly separates:
- Policy interface (protocol.py)
- Policy implementations (lstm_bundle.py, heuristic_bundle.py)
- Feature extraction (features.py)
- Action masking (action_masks.py)
- Registration (registry.py, factory.py)

The code demonstrates expertise in both RL algorithm design and PyTorch engineering.
