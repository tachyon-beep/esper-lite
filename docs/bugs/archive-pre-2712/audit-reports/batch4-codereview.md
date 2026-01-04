# Batch 4 Code Review: Simic Agent - PPO Implementation

**Reviewer:** Claude Opus 4.5 (Python Code Quality Specialist)
**Date:** 2025-12-27
**Batch:** 4 of N
**Files Reviewed:** 7

---

## Executive Summary

The Simic agent module implements a **factored PPO algorithm** for controlling seed lifecycle in Esper's morphogenetic training system. The implementation is mature, well-documented, and demonstrates sophisticated RL engineering practices including:

- Per-head causal masking for advantage computation
- Proper LSTM hidden state management for recurrent policies
- Comprehensive telemetry and anomaly detection
- Robust checkpoint save/load with version validation

**Overall Assessment:** HIGH QUALITY - Production-ready with minor improvements possible.

| Severity | Count | Summary |
|----------|-------|---------|
| P0 (Critical) | 0 | None |
| P1 (Correctness) | 1 | SET_ALPHA_TARGET missing from advantages.py causal documentation |
| P2 (Performance) | 2 | GAE loop optimization opportunity, dead code in entropy floor |
| P3 (Code Quality) | 4 | Documentation sync, type annotations, code organization |
| P4 (Style) | 3 | Minor naming, comment cleanup |

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/simic/agent/advantages.py`

**Purpose:** Per-head advantage computation with causal masking for factored action space.

**Strengths:**
- Excellent documentation of causal structure (decision tree)
- Clean implementation of per-head masking
- Properly handles all LifecycleOp variants
- M8 comment explains intentional lack of clone for op_advantages

**Concerns:**

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| ADV-1 | P1 | 20-34 | SET_ALPHA_TARGET documentation in decision tree is correct, but the test file `test_advantages.py` has no test case for it. Missing test coverage for a key operation. |
| ADV-2 | P3 | 61-64 | Variable `is_fossilize` and `is_advance` are NOT declared but would improve readability. Currently implicit in `~is_wait` for slot_mask. While correct, explicit is better. |

**Code Snippet (ADV-2):**
```python
# Current (implicit)
slot_mask = ~is_wait

# Clearer (explicit - not a bug, just documentation)
# slot: GERMINATE, FOSSILIZE, PRUNE, ADVANCE (all non-WAIT ops)
slot_mask = ~is_wait  # Intentionally covers FOSSILIZE and ADVANCE
```

---

### 2. `/home/john/esper-lite/src/esper/simic/agent/__init__.py`

**Purpose:** Package exports for agent submodule.

**Strengths:**
- Clean, organized exports
- Groups exports by function (Advantages, Buffer, PPO Agent, Types)
- `__all__` matches imports exactly

**Concerns:** None identified.

---

### 3. `/home/john/esper-lite/src/esper/simic/agent/ppo.py`

**Purpose:** Core PPOAgent implementation with factored recurrent actor-critic.

**Strengths:**
- Comprehensive hyperparameter configuration with sensible defaults from leyline
- Proper weight decay separation (actor/shared/critic) following RL best practices
- Robust early stopping with KL divergence check BEFORE optimizer.step() (BUG-003 fix)
- Per-head entropy tracking with causal masking
- Proper LSTM hidden state staleness warnings
- Checkpoint versioning with strict validation (no backwards compatibility shims)
- Excellent GPU memory optimization (batched GPU->CPU transfers)

**Concerns:**

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PPO-1 | P2 | 423-461 | `get_entropy_floor()` has a TODO noting it may be dead code - `action_mask` is not threaded through callers. Either implement the threading or remove the adaptive floor feature. |
| PPO-2 | P3 | 100-117 | `signals_to_features` builds obs dict with many fields, but some fields (`grad_norm_host`) are mentioned in test mocks but not in the actual implementation. |
| PPO-3 | P3 | 746 | `entropy_coef_per_head.get(key, 1.0)` uses `.get()` - verify this is legitimate optional field handling, not defensive programming hiding a bug. **AUTHORIZED:** This is legitimate because entropy_coef_per_head is user-configurable and may not include all heads. |
| PPO-4 | P4 | 541 | `head_grad_norm_history` type annotation could be `dict[str, list[float]]` explicitly in docstring. |
| PPO-5 | P3 | 989-990 | `config.get('compile_mode', 'off')` in load() - this is defensive for old checkpoints. Per No Legacy Code Policy, consider failing if missing instead of defaulting. |

**Code Snippet (PPO-5):**
```python
# Line 989-990: Defensive default for compile_mode
compile_mode = config.get('compile_mode', 'off')

# Per No Legacy Code Policy, could be:
if 'compile_mode' not in config:
    raise RuntimeError("Incompatible checkpoint: config.compile_mode is required")
compile_mode = config['compile_mode']
```

However, noting that test `test_compile_mode_defaults_to_off_for_old_checkpoints` explicitly tests this default behavior, suggesting it's intentional for some migration path. This may warrant discussion.

---

### 4. `/home/john/esper-lite/src/esper/simic/agent/rollout_buffer.py`

**Purpose:** Pre-allocated tensor buffer for factored recurrent PPO rollouts.

**Strengths:**
- Excellent pre-allocation strategy (fixed episode length, no GC pressure)
- Proper per-environment GAE computation (P0 bug fix referenced)
- `@torch.compiler.disable` decorator on GAE loop (correctly identified as incompatible with compile)
- Proper `detach()` calls to prevent gradient graph memory leak
- Non-blocking CPU->GPU transfers with `non_blocking=True`
- Clear documentation of design rationale

**Concerns:**

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| BUF-1 | P2 | 347-416 | `compute_advantages_and_returns()` has a TODO for vectorization. The current O(envs * steps) loop is correct but could be ~4x faster with tensor operations for fixed-length rollouts. Low priority since this runs once per rollout. |
| BUF-2 | P4 | 113 | `lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM` - consider renaming to match common convention `hidden_dim` for consistency with PolicyBundle. |
| BUF-3 | P3 | 435 | `std = all_adv.std(correction=0)` - comment explains why (Bessel correction undefined for n=1), but could add assertion that len(all_adv) > 0 before this line (currently guaranteed by the `if not all_advantages: return` check). |

---

### 5. `/home/john/esper-lite/src/esper/simic/agent/types.py`

**Purpose:** TypedDict definitions for PPO metrics and action types.

**Strengths:**
- Proper use of TypedDict for structured return types
- `total=False` on PPOUpdateMetrics correctly marks all keys as optional
- Good documentation of aggregation behavior in docstrings
- HeadGradientNorms includes all 9 heads (8 action + value)

**Concerns:**

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| TYP-1 | P4 | 39 | `PPOUpdateMetrics` has many fields but no validation that returned dicts match. Consider adding a factory function or dataclass instead of TypedDict for runtime validation. |
| TYP-2 | P3 | 65-66 | `head_entropies` and `head_grad_norms` are typed as `dict[str, list[float]]` but should include all HEAD_NAMES keys. Could use TypedDict for inner structure. |

---

### 6. `/home/john/esper-lite/src/esper/simic/contracts.py`

**Purpose:** Protocol definitions for decoupling Simic from Kasmina.

**Strengths:**
- Clean protocol definitions following Python typing best practices
- `@runtime_checkable` on SeedSlotProtocol enables isinstance checks
- SeedStateProtocol captures the minimal interface needed
- SlottedHostProtocol is comprehensive for host model operations

**Concerns:**

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| CON-1 | P3 | 25-28 | `SeedStateProtocol.metrics`, `alpha_controller`, `alpha_algorithm` typed as `Any` - could use more specific Protocol types for better type safety. |
| CON-2 | P4 | 10 | Imports `contextmanager` and `Iterator` but only used in one method signature. Clean, just noting the typing dependency. |

---

### 7. `/home/john/esper-lite/src/esper/simic/__init__.py`

**Purpose:** Package-level exports for the Simic subsystem.

**Strengths:**
- Organized imports by category (Control, Rewards, Telemetry, Training)
- Lazy imports for heavy modules (PPOAgent, train_ppo_vectorized) noted in comments
- Re-exports from subpackages (rewards, telemetry) for convenient access

**Concerns:**

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| SIM-1 | P4 | 30 | `from esper.tamiyo.policy.features import safe, TaskConfig` - cross-domain import. Simic depends on Tamiyo for features. This coupling is documented but worth noting. |
| SIM-2 | P3 | 93 | `from esper.simic.training import ParallelEnvState` - this is the only import from training at package level. Ensure this doesn't cause circular import issues. |

---

## Cross-Cutting Integration Risks

### 1. Causal Masking Consistency (Medium Risk)

**Files:** `advantages.py`, `ppo.py`

The causal masking logic is duplicated between:
- `compute_per_head_advantages()` in advantages.py (lines 60-94)
- PPO update loop in ppo.py (lines 616-629)

Both use the same masks, but if LifecycleOp is extended, both locations must be updated. Consider extracting a shared `get_causal_masks(op_actions)` helper.

### 2. Slot Configuration Contract (Low Risk)

**Files:** `ppo.py`, `rollout_buffer.py`

Both files depend on `SlotConfig` from leyline. The slot_config assertion at line 340 of ppo.py (`buffer.slot_config.slot_ids == policy.slot_config.slot_ids`) is critical for preventing silent training corruption. This is well-guarded.

### 3. Test Coverage for SET_ALPHA_TARGET (Medium Risk)

**Files:** `advantages.py`, `tests/simic/test_advantages.py`

The test file covers WAIT, GERMINATE, FOSSILIZE, PRUNE, ADVANCE but does NOT have a dedicated test for SET_ALPHA_TARGET. While SET_ALPHA_TARGET is partially covered by `test_mixed_ops_correct_masking`, it deserves explicit verification of:
- slot_mask active
- style_mask active
- alpha_target_mask active
- alpha_speed_mask active
- alpha_curve_mask active
- blueprint_mask masked (zero)
- tempo_mask masked (zero)

### 4. Checkpoint Compatibility Policy Tension (Low Risk)

**Files:** `ppo.py` (lines 989-990)

The `config.get('compile_mode', 'off')` default conflicts with the No Legacy Code Policy stated in CLAUDE.md. However, there's an explicit test for this behavior. Recommend either:
- Remove the default and require all checkpoints have compile_mode
- Document this as an intentional exception to the policy

---

## Severity-Tagged Findings Summary

### P0 (Critical) - Must Fix Before Merge
*None identified*

### P1 (Correctness) - Should Fix
| ID | File | Line | Issue |
|----|------|------|-------|
| ADV-1 | advantages.py | N/A | Missing test coverage for SET_ALPHA_TARGET |

### P2 (Performance) - Should Consider
| ID | File | Line | Issue |
|----|------|------|-------|
| PPO-1 | ppo.py | 423-461 | Dead code: get_entropy_floor adaptive floor not threaded |
| BUF-1 | rollout_buffer.py | 347-416 | GAE loop vectorization opportunity |

### P3 (Code Quality) - Nice to Have
| ID | File | Line | Issue |
|----|------|------|-------|
| ADV-2 | advantages.py | 61-64 | Implicit FOSSILIZE/ADVANCE in slot_mask |
| PPO-2 | ppo.py | 100-117 | Unused fields in obs dict vs test mocks |
| PPO-5 | ppo.py | 989-990 | compile_mode default vs No Legacy Code Policy |
| TYP-2 | types.py | 65-66 | head_entropies/head_grad_norms typing |
| CON-1 | contracts.py | 25-28 | Protocol fields typed as Any |
| SIM-2 | __init__.py | 93 | ParallelEnvState import from training |
| BUF-3 | rollout_buffer.py | 435 | std computation edge case |

### P4 (Style) - Minor
| ID | File | Line | Issue |
|----|------|------|-------|
| PPO-4 | ppo.py | 541 | head_grad_norm_history type annotation |
| BUF-2 | rollout_buffer.py | 113 | lstm_hidden_dim naming |
| TYP-1 | types.py | 39 | PPOUpdateMetrics validation |
| CON-2 | contracts.py | 10 | Import usage |
| SIM-1 | __init__.py | 30 | Cross-domain import |

---

## Recommendations

### Immediate Actions

1. **Add SET_ALPHA_TARGET test** (ADV-1): Add explicit test case in `test_advantages.py`:
```python
def test_set_alpha_target_correct_masking(self):
    """When op=SET_ALPHA_TARGET, slot/style/alpha heads should get advantage."""
    op_actions = torch.tensor([LifecycleOp.SET_ALPHA_TARGET])
    base_advantages = torch.tensor([2.0])

    per_head = compute_per_head_advantages(base_advantages, op_actions)

    assert torch.allclose(per_head["op"], base_advantages)
    assert torch.allclose(per_head["slot"], base_advantages)
    assert torch.allclose(per_head["style"], base_advantages)
    assert torch.allclose(per_head["alpha_target"], base_advantages)
    assert torch.allclose(per_head["alpha_speed"], base_advantages)
    assert torch.allclose(per_head["alpha_curve"], base_advantages)
    assert torch.allclose(per_head["blueprint"], torch.zeros(1))
    assert torch.allclose(per_head["tempo"], torch.zeros(1))
```

2. **Resolve entropy floor dead code** (PPO-1): Either wire `action_mask` through `get_entropy_coef()` callers or remove the adaptive floor feature with a TODO comment explaining when it might be useful.

### Future Considerations

1. Extract `get_causal_masks()` helper to deduplicate masking logic between advantages.py and ppo.py
2. Consider stricter checkpoint validation (PPO-5) per No Legacy Code Policy
3. Vectorize GAE computation if profiling shows it as a bottleneck

---

## Test Coverage Assessment

| File | Unit Tests | Integration Tests | Coverage |
|------|------------|-------------------|----------|
| advantages.py | test_advantages.py | - | Good (missing SET_ALPHA_TARGET) |
| ppo.py | test_ppo.py | test_ppo_integration.py | Excellent |
| rollout_buffer.py | test_ppo.py (indirect) | - | Adequate |
| types.py | - | - | N/A (type definitions) |
| contracts.py | - | - | N/A (protocols) |

---

## Conclusion

The Simic agent implementation is **production-ready** with well-considered design decisions and comprehensive test coverage. The PPO implementation correctly handles:

- Factored action spaces with 8 heads
- Recurrent LSTM policies with proper hidden state management
- Per-head causal masking for clean gradient attribution
- Robust checkpointing with version validation

The identified issues are minor and do not block deployment. The P1 finding (missing test for SET_ALPHA_TARGET) should be addressed to ensure complete causal masking coverage, but the implementation itself is correct.

**Recommendation:** Approve for merge after adding SET_ALPHA_TARGET test case.
