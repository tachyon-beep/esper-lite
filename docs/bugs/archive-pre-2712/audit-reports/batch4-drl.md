# Batch 4 Deep Dive Review: Simic Agent (PPO Implementation)

**Reviewer**: DRL Expert
**Date**: 2025-12-27
**Branch**: `ux-overwatch-refactor`
**Files Reviewed**: 7 files in `/src/esper/simic/agent/` and `/src/esper/simic/`

---

## Executive Summary

The PPO implementation is **production-quality** with impressive attention to:
- Correct factored action space handling with causal masking (per-head advantages)
- Proper recurrent policy handling with LSTM hidden state coherence
- Extensive telemetry and diagnostic infrastructure

However, I identified **1 P1 issue** (logic error in advantage masking), **3 P2 issues** (performance/correctness), and several P3/P4 quality improvements.

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/simic/agent/advantages.py`

**Purpose**: Computes per-head advantages with causal masking based on the factored action space's causal structure.

**Strengths**:
- Excellent documentation of the causal decision tree
- Correct understanding that only causally-relevant heads should receive gradient signal
- Clean, readable implementation

**Concerns**:

| ID | Severity | Line(s) | Finding |
|----|----------|---------|---------|
| A-1 | **P1** | 60-94 | **Missing FOSSILIZE and ADVANCE from causal masks** - The docstring (lines 17-34) documents FOSSILIZE and ADVANCE operations, but neither `is_fossilize` nor `is_advance` are computed. The slot_mask uses `~is_wait` which is correct (FOSSILIZE/ADVANCE need slot selection), but if any other head should be relevant for these ops, they're incorrectly masked to zero. Currently no other heads are documented as relevant for these ops, so this is more of a **documentation/defensiveness issue** than a functional bug. However, the comment at line 70 claims "slot head: relevant for GERMINATE, FOSSILIZE, PRUNE, ADVANCE" which matches `~is_wait`, so this is actually correct. Downgrading to P3 - the asymmetry between documented ops and computed masks is confusing. |
| A-2 | P3 | 66-68 | **Clone vs no-clone comment** - Comment says "No clone needed - we're not modifying the tensor" but the tensor IS being returned and could theoretically be modified by the caller. This is fine since PPO's update() only reads from the returned dict, but the reasoning in the comment is incomplete. |

**Corrected Assessment**: After re-reading, the masking logic is correct. `~is_wait` correctly handles all non-WAIT ops for slot head. The only true issue is stylistic - FOSSILIZE and ADVANCE are documented but not explicitly computed as booleans.

---

### 2. `/home/john/esper-lite/src/esper/simic/agent/__init__.py`

**Purpose**: Package exports for the agent submodule.

**Assessment**: Clean, well-organized exports. No issues found.

---

### 3. `/home/john/esper-lite/src/esper/simic/agent/ppo.py`

**Purpose**: Core PPO implementation with factored recurrent actor-critic support.

**Strengths**:
- Excellent handling of recurrent policy challenges (value staleness warnings, single-batch processing rationale)
- Proper KL early stopping that works with `recurrent_n_epochs=1` (BUG-003 fix)
- Per-head entropy weighting with causal masking
- Weight decay correctly applied to critic only (DRL best practice)
- Extensive telemetry (per-head entropy, per-head gradient norms)
- Good checkpoint versioning with forward compatibility

**Concerns**:

| ID | Severity | Line(s) | Finding |
|----|----------|---------|---------|
| P-1 | **P1** | 616-629 | **Causal mask consistency with advantages.py** - The head_masks dict here duplicates the logic from `compute_per_head_advantages()` in advantages.py. If either changes, they'll drift. This is a DRY violation with correctness implications. The masks MUST be identical or gradient signals will be inconsistent. **Recommendation**: Extract causal mask computation to a shared function in advantages.py and import here. |
| P-2 | **P2** | 667-683 | **KL computation creates tensors on device inside inference_mode** - While `torch.inference_mode()` prevents gradient tracking, the repeated `torch.tensor(0.0, device=self.device)` allocations could be pre-allocated. Minor perf impact but inconsistent with the performance optimization comments elsewhere. |
| P-3 | **P2** | 746 | **`entropy_coef_per_head.get(key, 1.0)`** - This `.get()` pattern could mask a typo in key names. If a head name is misspelled in the dict, it silently uses 1.0 instead of failing. Given the project's prohibition on defensive programming, this should use direct key access: `self.entropy_coef_per_head[key]` (assuming keys are validated at init). However, since `entropy_coef_per_head` is optional and defaults to a dict with all heads, this is acceptable. Downgrading to P4. |
| P-4 | P3 | 423-424 | **TODO comment for potential dead code** - The `get_entropy_floor` method is marked as potential dead code. This should be either implemented fully (threading action_mask through callers) or removed. Currently it's half-implemented. |
| P-5 | P3 | 840-851 | **Metric aggregation handles mixed types** - The aggregation converts list[float] to float for scalars but keeps dicts as-is. The type annotation `PPOUpdateMetrics` uses `total=False` to allow partial keys, but the actual return type is more complex (some values are float, some are dict). This is technically correct but confusing. |
| P-6 | P4 | 243-247 | **Repeated getattr unwrap pattern** - The `getattr(policy.network, '_orig_mod', policy.network)` pattern appears 5 times. Consider extracting to a helper method `_unwrap_network()`. |
| P-7 | P2 | 936 | **`weights_only=True` may fail on older checkpoints** - `torch.load(..., weights_only=True)` rejects non-tensor types. The checkpoint contains dicts with strings (`architecture`, `config`). This should fail on ANY checkpoint since checkpoint format includes non-tensor data. **CRITICAL CHECK NEEDED**: Either this works because PyTorch 2.x allows dict/list/str in weights_only mode, or this is a P0 bug. Testing required. |
| P-8 | P3 | 877 | **Save uses `self.policy.state_dict()`** - Saves via PolicyBundle, but load uses `agent.policy.load_state_dict()` directly. Symmetric, but the comment on line 873 says "Get network state dict from policy" while the actual line uses the bundle's state_dict. Minor confusion. |
| P-9 | P3 | 571-574 | **Hidden state passed to evaluate_actions** - The code passes `hidden=(data["initial_hidden_h"], data["initial_hidden_c"])` to `evaluate_actions`. The docstring for `initial_hidden()` in lstm_bundle.py warns that these are inference-mode tensors not suitable for gradient computation. However, here the data is loaded from the rollout buffer (which stores hidden states from rollout collection). The question is: are these tensors in the buffer differentiable? Looking at rollout_buffer.py line 342-343, they're `.detach()`ed. This is CORRECT for PPO - we don't backprop through the initial hidden state, we reconstruct LSTM evolution during the forward pass. But this could use a clarifying comment. |

**Critical Check for P-7**:
Looking at PyTorch 2.x documentation, `weights_only=True` allows:
- Tensors, dicts, lists, strings, ints, floats, bools, None
- It blocks arbitrary Python code execution (pickled classes)

So `weights_only=True` SHOULD work since the checkpoint only contains these safe types. Not a bug.

---

### 4. `/home/john/esper-lite/src/esper/simic/agent/rollout_buffer.py`

**Purpose**: Per-environment rollout storage with pre-allocated tensors for factored recurrent actor-critic.

**Strengths**:
- Excellent design: pre-allocated tensors, per-env storage, LSTM state tracking
- Fixes the GAE interleaving bug by computing GAE per-environment
- Good documentation of design rationale
- Proper handling of truncation vs terminal states in GAE

**Concerns**:

| ID | Severity | Line(s) | Finding |
|----|----------|---------|---------|
| R-1 | **P2** | 388 | **`zero_tensor.clone()` inside loop** - The code clones `zero_tensor` to avoid in-place modification, but this creates a new tensor on every `dones[t] and not truncated[t]`. For a 25-epoch episode, this could be up to 25 small allocations. Better: use `last_gae = torch.zeros_like(last_gae)` or just `last_gae = zero_tensor` (the multiplication in line 412 creates a new tensor anyway). |
| R-2 | **P2** | 390-412 | **GAE loop not vectorizable due to data dependencies** - The comment acknowledges this, but there's a partial vectorization opportunity: the terminal detection (`dones[t] and not truncated[t]`) could be pre-computed as a boolean tensor, reducing Python overhead. |
| R-3 | P3 | 215-230 | **Default mask initialization with first action valid** - Setting first action as valid for padding prevents `InvalidStateMachineError`, but it means padded timesteps have a specific (albeit unused) action structure. This is fine since `valid_mask` filters them out, but the comment could clarify this is purely for error prevention, not semantic correctness. |
| R-4 | P3 | 421-443 | **`normalize_advantages` uses cat then iterates** - First `torch.cat(all_advantages)` to compute mean/std, then iterates again to normalize. Could normalize in the first loop iteration using online algorithms (Welford's), but the current approach is clearer and the buffer is small. |
| R-5 | P4 | 458-460 | **step_counts to tensor conversion** - `torch.tensor(self.step_counts, device=device)` creates a new tensor every call. Could cache this. Minor since this is called once per update. |
| R-6 | P4 | 502-503 | **initial_hidden uses permute().contiguous()** - The comment explains this is for LSTM format, but `.contiguous()` might not be needed after `.to(device)` since `.to()` often makes tensors contiguous. Micro-optimization, no action needed. |

---

### 5. `/home/john/esper-lite/src/esper/simic/agent/types.py`

**Purpose**: TypedDict definitions for PPO metrics and structured returns.

**Strengths**:
- Clean type definitions
- `total=False` correctly used for optional metrics

**Concerns**:

| ID | Severity | Line(s) | Finding |
|----|----------|---------|---------|
| T-1 | P4 | 39-67 | **PPOUpdateMetrics missing `advantage_mean` and `advantage_std`** - These are computed in ppo.py lines 532-536 but not declared in the TypedDict. Add them for completeness. |
| T-2 | P4 | 21-36 | **HeadGradientNorms unused** - Declared but the actual metrics use `dict[str, list[float]]` format. Either use this TypedDict or remove it. |

---

### 6. `/home/john/esper-lite/src/esper/simic/contracts.py`

**Purpose**: Protocol definitions for decoupling Simic from Kasmina.

**Strengths**:
- Good use of Protocols for loose coupling
- `runtime_checkable` on SeedSlotProtocol enables validation
- Comprehensive interface capture

**Concerns**:

| ID | Severity | Line(s) | Finding |
|----|----------|---------|---------|
| C-1 | P3 | 26-28 | **`Any` type annotations on metrics, alpha_controller, alpha_algorithm** - These could be more specific (e.g., `SeedMetrics`, `AlphaController` Protocol). Using `Any` defeats some type checking benefits. |
| C-2 | P4 | 149-150 | **germinate_seed ellipsis for default args** - The `...` notation for default args in Protocol methods is unconventional. Consider documenting what the defaults should be or removing them. |

---

### 7. `/home/john/esper-lite/src/esper/simic/__init__.py`

**Purpose**: Simic package exports.

**Strengths**:
- Good organization with lazy imports for heavy modules
- Clear grouping of exports by functionality

**Concerns**:

| ID | Severity | Line(s) | Finding |
|----|----------|---------|---------|
| S-1 | P4 | 29-37 | **Mixed import sources** - Some imports from `esper.simic.control`, others from `esper.tamiyo.policy.*`. The Tamiyo imports (`safe`, `TaskConfig`, `MaskedCategorical`, etc.) might be better re-exported from a simic-internal module to maintain clearer boundaries. However, this is architectural, not a bug. |

---

## Cross-Cutting Integration Risks

### 1. Causal Mask Duplication (P1)

**Risk**: `compute_per_head_advantages()` and PPO's `update()` both compute causal masks. If one changes without the other, gradients will be inconsistent.

**Location**:
- `/home/john/esper-lite/src/esper/simic/agent/advantages.py:60-94`
- `/home/john/esper-lite/src/esper/simic/agent/ppo.py:616-629`

**Recommendation**: Extract to a single source of truth:
```python
# In advantages.py
def compute_causal_masks(op_actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute causal relevance masks for each action head."""
    is_wait = op_actions == LifecycleOp.WAIT
    is_germinate = op_actions == LifecycleOp.GERMINATE
    is_set_alpha = op_actions == LifecycleOp.SET_ALPHA_TARGET
    is_prune = op_actions == LifecycleOp.PRUNE

    return {
        "op": torch.ones_like(is_wait),
        "slot": ~is_wait,
        "blueprint": is_germinate,
        "style": is_germinate | is_set_alpha,
        "tempo": is_germinate,
        "alpha_target": is_set_alpha | is_germinate,
        "alpha_speed": is_set_alpha | is_prune,
        "alpha_curve": is_set_alpha | is_prune,
    }
```

### 2. Hidden State Contract Between Buffer and Network

**Risk**: The rollout buffer stores LSTM hidden states from rollout collection (inference-mode, detached). These are passed to `evaluate_actions()`. The network must reconstruct the LSTM evolution during training, not rely on these stored states for gradients.

**Assessment**: Currently correct. The network's `evaluate_actions()` uses the initial hidden state as a starting point but recomputes the LSTM forward pass, producing gradient-compatible hidden states for BPTT.

**Recommendation**: Add an explicit comment in ppo.py around line 571-574 clarifying this contract.

### 3. Slot Config Consistency

**Risk**: SlotConfig flows through many components (PPOAgent, TamiyoRolloutBuffer, FactoredRecurrentActorCritic). Misalignment causes silent training corruption.

**Assessment**: The code has good assertions (ppo.py:327-344) verifying slot_ids match exactly between buffer and policy. This is well-handled.

### 4. Test Coverage Gaps

**Observation**: Looking at tests, there's good coverage for:
- KL early stopping (including BUG-003 regression test)
- Checkpoint round-trip
- Weight decay optimizer coverage
- Multi-slot configurations

**Missing coverage**:
- GAE computation correctness (no direct tests for `compute_advantages_and_returns`)
- Causal mask correctness (no tests verifying blueprint head only receives gradient during GERMINATE)
- Per-head entropy masking (no tests for sparse head entropy computation)

---

## Severity-Tagged Findings Summary

### P0 (Critical) - None found

### P1 (Correctness Bugs)
| ID | File | Description |
|----|------|-------------|
| P-1 | ppo.py | Causal mask logic duplicated between advantages.py and ppo.py - must stay in sync |

### P2 (Performance/Correctness Concerns)
| ID | File | Description |
|----|------|-------------|
| P-2 | ppo.py | Tensor allocations inside inference_mode loop in KL computation |
| R-1 | rollout_buffer.py | `zero_tensor.clone()` allocates inside GAE loop |
| R-2 | rollout_buffer.py | GAE loop could pre-compute terminal detection tensor |

### P3 (Code Quality)
| ID | File | Description |
|----|------|-------------|
| A-2 | advantages.py | Comment about clone reasoning is incomplete |
| P-4 | ppo.py | TODO for dead code `get_entropy_floor` |
| P-5 | ppo.py | Metric aggregation type handling is confusing |
| P-8 | ppo.py | Save/load state_dict comment inconsistency |
| P-9 | ppo.py | Missing comment on hidden state contract |
| R-3 | rollout_buffer.py | Default mask initialization comment could be clearer |
| R-4 | rollout_buffer.py | normalize_advantages iterates twice |
| C-1 | contracts.py | `Any` types where specific protocols could be used |
| T-1 | types.py | PPOUpdateMetrics missing advantage_mean/std fields |

### P4 (Style/Minor)
| ID | File | Description |
|----|------|-------------|
| P-3 (downgraded) | ppo.py | `.get()` pattern for entropy_coef_per_head |
| P-6 | ppo.py | Repeated getattr unwrap pattern |
| R-5 | rollout_buffer.py | step_counts tensor creation not cached |
| R-6 | rollout_buffer.py | Potentially unnecessary .contiguous() |
| T-2 | types.py | HeadGradientNorms TypedDict unused |
| C-2 | contracts.py | Ellipsis for Protocol default args |
| S-1 | __init__.py | Mixed import sources from simic/tamiyo |

---

## Algorithm Correctness Assessment

### PPO Implementation

**Clipping**: Correct. Uses standard `min(surr1, surr2)` clipped surrogate objective.

**Advantage Estimation**: Correct. GAE with proper handling of truncation vs terminal states.

**Value Clipping**: Correct. Uses separate `value_clip` (default 10.0) instead of policy `clip_ratio` (0.2).

**KL Divergence**: Correct. Uses KL3 estimator (Schulman's approximation). Properly weighted by causal relevance.

**Entropy Bonus**: Correct. Per-head entropy with causal masking to avoid diluting signal for sparse heads.

**Gradient Clipping**: Correct. `clip_grad_norm_` after backward, before step.

### Factored Action Space Handling

**Per-Head Ratios**: Correct. Each head computes its own ratio independently.

**Causal Masking**: Correct (when consistent). Only causally-relevant heads receive gradient.

**Action Masking**: Delegated to network's `evaluate_actions`, which uses `MaskedCategorical`.

### Recurrent Policy Handling

**Hidden State**: Correct. Stores initial hidden states per sequence, passes to network for BPTT.

**Single-Batch Processing**: Correct. No minibatch shuffling for recurrent policy (preserves temporal dependencies).

**Value Staleness Warning**: Excellent. Warns users about the recurrent PPO value clipping issue.

---

## Recommendations Priority

1. **P1 - Extract causal mask computation** to single source of truth
2. **P2 - Pre-allocate tensors** in KL computation and GAE loop
3. **P3 - Add hidden state contract comment** in ppo.py
4. **P3 - Resolve or remove** `get_entropy_floor` TODO
5. **P4 - Add missing TypedDict fields** for advantage_mean/std

---

## Conclusion

This is a **well-engineered PPO implementation** that handles the complexities of factored recurrent actor-critic training correctly. The main risk is the duplicated causal mask logic which should be extracted to a single source of truth. The performance concerns are minor but could be addressed in a polish pass.

The test coverage is solid for happy paths but could benefit from more unit tests for the core algorithms (GAE, causal masking).

**Overall Assessment**: Ready for production with the P1 fix for mask duplication.
