# Batch 9 Code Review: Tamiyo Core + Networks (PyTorch Engineering)

**Reviewer Specialization:** PyTorch Engineering
**Date:** 2025-12-27
**Files Reviewed:**
1. `/home/john/esper-lite/src/esper/tamiyo/decisions.py`
2. `/home/john/esper-lite/src/esper/tamiyo/heuristic.py`
3. `/home/john/esper-lite/src/esper/tamiyo/__init__.py`
4. `/home/john/esper-lite/src/esper/tamiyo/tracker.py`
5. `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py`
6. `/home/john/esper-lite/src/esper/tamiyo/networks/__init__.py`

---

## Executive Summary

The Tamiyo domain implements the "brain" of Esper - a strategic decision-making system for seed lifecycle management. From a PyTorch engineering perspective, the code demonstrates solid understanding of recurrent network design, proper LSTM initialization patterns, and numerical stability for mixed-precision training. The factored LSTM architecture is well-designed with appropriate LayerNorm placement for training stability.

**Key Strengths:**
- Correct LSTM hidden state shape handling (num_layers, batch, hidden_dim)
- Forget gate bias initialization following best practices (Gers et al., 2000)
- FP16-safe masking with MASKED_LOGIT_VALUE = -1e4 (avoids overflow)
- Proper gradient isolation via detach() at episode boundaries
- Well-documented hidden state memory management warnings

**Primary Concerns:**
- One backwards compatibility comment violates project policy (P3)
- Minor test coverage gap for edge cases

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/tamiyo/decisions.py`

**Purpose:** Defines `TamiyoDecision` dataclass representing strategic decisions with action, target seed, reason, and confidence.

**Lines:** 54

**PyTorch Relevance:** Minimal - pure Python dataclass with no tensor operations.

**Analysis:**
- Clean, minimal implementation
- Helper functions `_is_germinate_action()` and `_get_blueprint_from_action()` use string parsing on enum names - functional but brittle if naming conventions change
- The `confidence` field (default 1.0) is never used in torch operations - could be useful for weighted loss or exploration

**Findings:** None

---

### 2. `/home/john/esper-lite/src/esper/tamiyo/heuristic.py`

**Purpose:** Rule-based heuristic policy for seed lifecycle management. Implements the `TamiyoPolicy` protocol for baseline comparisons.

**Lines:** 358

**PyTorch Relevance:** Low - uses dynamic enum from `build_action_enum()` but no tensor operations.

**Analysis:**

The heuristic policy is clean Python code without PyTorch dependencies. Key observations:

1. **Blueprint Validation at Init (lines 102-115):** Good defensive check that validates blueprint_rotation against available actions early, preventing runtime AttributeError.

2. **Ransomware Detection (lines 256-266):** Novel pattern detection for seeds that create dependencies without adding value. Well-documented with clear thresholds.

3. **Blueprint Penalty Decay (lines 126-131, 311-316):** Per-epoch decay (not per-decision) with configurable decay rate. The 0.1 threshold for penalty deletion is hardcoded - could be a config option.

4. **Authorized getattr (line 183):** The dynamic enum lookup with `getattr(Action, f"GERMINATE_{blueprint_id.upper()}")` is properly documented with authorization comment.

**Findings:** None

---

### 3. `/home/john/esper-lite/src/esper/tamiyo/__init__.py`

**Purpose:** Package initialization, exports core components and policy registry.

**Lines:** 55

**PyTorch Relevance:** None - pure import/export.

**Analysis:**

**[P3] Line 41: Backwards Compatibility Comment Violates Policy**

```python
# Legacy heuristic (kept for backwards compatibility)
"TamiyoPolicy",
"HeuristicPolicyConfig",
"HeuristicTamiyo",
```

The comment states these are "kept for backwards compatibility" which directly violates the project's No Legacy Code Policy. Per CLAUDE.md:

> **STRICT REQUIREMENT:** Legacy code, backwards compatibility, and compatibility shims are strictly forbidden.

Either:
1. These exports are still actively used (remove the misleading comment), OR
2. They are truly legacy and should be removed

**Recommendation:** Grep for usages and either remove the comment or remove the exports.

---

### 4. `/home/john/esper-lite/src/esper/tamiyo/tracker.py`

**Purpose:** `SignalTracker` maintains running statistics for Tamiyo's decision-making, tracking loss/accuracy history, plateau detection, and host stabilization.

**Lines:** 348

**PyTorch Relevance:** Low - uses deque for history, no tensor operations. Interacts with Nissa telemetry hub.

**Analysis:**

1. **Stabilization Latch (lines 124-163):** Well-designed latch behavior that stays True once set. The P1-A fix (line 132-134) correctly requires `loss_delta >= 0` to prevent regression epochs counting as stable.

2. **Accuracy Scale Validation (lines 104-116):** Good warning for incorrect scale (expects 0-100, warns on 0-1). Could be stricter with an exception for obviously wrong values.

3. **Seed Summary Selection (lines 196-213):** Multi-slot summary uses a deterministic tie-break key:
   - Highest stage
   - Highest alpha
   - Most negative counterfactual (safety)
   - seed_id for determinism

   This is well-documented and correct.

4. **`peek()` Method (lines 241-322):** Read-only version of `update()` for bootstrap value computation. Duplicates significant logic from `update()` - could benefit from internal factoring to reduce drift risk.

**Findings:** None (P3 items noted but not actionable without broader refactor)

---

### 5. `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py`

**Purpose:** Core recurrent actor-critic network with factored action heads for multi-dimensional action space (slot, blueprint, style, tempo, alpha params, lifecycle op).

**Lines:** 636

**PyTorch Relevance:** **HIGH** - This is the primary neural network implementation.

**Architecture:**
```
state -> feature_net (Linear+LN+ReLU) -> LSTM -> lstm_ln -> 8 parallel heads
                                                        -> value_head
```

**Detailed Analysis:**

#### LSTM Implementation Correctness

1. **Hidden State Shapes (line 285-287):**
   ```python
   h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
   c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
   ```
   Correct shape: `[num_layers, batch, hidden_dim]`

2. **Forget Gate Bias Initialization (lines 246-257):**
   ```python
   n = param.size(0)
   param.data[n // 4 : n // 2].fill_(1.0)
   ```
   Correctly sets forget gate bias to 1.0 per Gers et al. (2000). The comment explains PyTorch's gate order: [input, forget, cell, output].

3. **LSTM num_layers=1 Default:** Single layer is appropriate for the 25-epoch episode length. Multi-layer LSTMs would be overkill.

#### LayerNorm Placement

1. **Pre-LSTM LayerNorm (line 130):** Inside feature_net before LSTM input. Stabilizes LSTM gate activations.

2. **Post-LSTM LayerNorm (line 150):** Critical for preventing hidden state magnitude drift in long sequences. The comment (lines 142-149) correctly cites Ba et al., 2016 and explains the "LN everywhere" pattern.

Both LayerNorms are correctly placed and documented.

#### Action Masking Implementation

1. **MASKED_LOGIT_VALUE = -1e4 (from leyline):** FP16-safe value that produces near-zero probability after softmax without overflow. Tested across dtypes in test suite.

2. **Mask Application (lines 341-356):**
   ```python
   slot_logits = slot_logits.masked_fill(~slot_mask, MASKED_LOGIT_VALUE)
   ```
   Uses `masked_fill` which is efficient and compile-friendly.

#### Hidden State Memory Management

**Documentation (lines 266-282):** Excellent warning about gradient graph retention:

```python
"""
MEMORY MANAGEMENT - Hidden State Detachment:
--------------------------------------------
LSTM hidden states carry gradient graphs. To prevent memory leaks during
training, callers MUST detach hidden states at episode boundaries:

    hidden = (h.detach(), c.detach())  # Break gradient graph
...
"""
```

**Integration Verification:** Checked `/home/john/esper-lite/src/esper/simic/training/vectorized.py`:
- Lines 2934-2935: `hidden_h.detach()`, `hidden_c.detach()` in buffer storage
- Lines 2944-2954: Episode reset creates fresh hidden states via `initial_hidden(1)`

The detachment is properly handled at episode boundaries.

#### Inference Mode Usage

**`get_action()` (lines 436-531):**
```python
with torch.inference_mode():
    output = self.forward(...)
```

Correct use of `inference_mode()` for action sampling during rollout collection. The docstring (lines 391-397) correctly warns that returned log_probs are NOT differentiable.

**`evaluate_actions()` (lines 533-632):**
No inference_mode - correct, as this needs gradient computation for PPO updates.

#### torch.compile Compatibility

1. **No graph breaks detected:** The forward pass is straightforward Linear/LSTM/masking without Python control flow that would cause graph breaks.

2. **MaskedCategorical validation isolated:** In `action_masks.py`, validation functions use `@torch.compiler.disable` to prevent graph breaks from `.any()` calls.

3. **Style mask override (lines 502-508):** Avoids `.any()` call by unconditional cloning and masking - good for compile compatibility.

#### Head Architecture

All 8 heads share the same pattern:
```python
nn.Sequential(
    nn.Linear(lstm_hidden_dim, head_hidden),  # head_hidden = lstm_hidden_dim // 2
    nn.ReLU(),
    nn.Linear(head_hidden, num_actions),
)
```

This is reasonable for the action space sizes. The smaller hidden dimension (64 for default 128 LSTM) reduces parameter count while maintaining capacity.

#### Initialization

1. **Orthogonal init (lines 208-213):** For all Linear layers with gain=sqrt(2) (appropriate for ReLU)
2. **Small output init (lines 216-229):** Policy heads use gain=0.01 for stable initial policies
3. **Value head init (lines 230-233):** Uses gain=1.0 (standard for value function)

This follows PPO best practices.

#### Style Head Conditional Logic

**In `get_action()` (lines 498-508):**
```python
style_irrelevant = (actions["op"] != LifecycleOp.GERMINATE) & (
    actions["op"] != LifecycleOp.SET_ALPHA_TARGET
)
style_mask_override[style_irrelevant] = False
style_mask_override[style_irrelevant, int(GerminationStyle.SIGMOID_ADD)] = True
```

Forces style to default when irrelevant. This is repeated in `evaluate_actions()` (lines 613-625) for consistency.

**[P4] Minor Code Duplication:** The style-irrelevance logic appears in both `get_action()` and `evaluate_actions()`. Could be factored into a helper, but low priority given the small code size.

**Findings:**

| Severity | Location | Issue |
|----------|----------|-------|
| P4 | Lines 498-508, 613-625 | Style mask override logic duplicated between get_action and evaluate_actions |

---

### 6. `/home/john/esper-lite/src/esper/tamiyo/networks/__init__.py`

**Purpose:** Re-exports `FactoredRecurrentActorCritic` and `GetActionResult`.

**Lines:** 7

**PyTorch Relevance:** None - pure re-export.

**Findings:** None

---

## Cross-Cutting Integration Risks

### 1. Hidden State Shape Consistency

**Risk:** Shape mismatch between network output and buffer storage.

**Verification:**
- Network returns `[num_layers, batch, hidden_dim]`
- Buffer stores with `squeeze(1)` to get `[num_layers, hidden_dim]` per env (line 342-343 in rollout_buffer.py)
- Restoration unsqueezes correctly

**Status:** No issue - shapes are consistent.

### 2. Mask Shape Normalization

**Risk:** 2D vs 3D mask shape inconsistency between get_action (inference) and evaluate_actions (training).

**Verification:**
- `get_action()` (lines 419-434): Converts 2D masks to 3D with unsqueeze(1)
- `forward()` expects 3D masks `[batch, seq_len, action_dim]`
- `evaluate_actions()` receives 3D masks from buffer

**Status:** No issue - shape normalization is handled.

### 3. dtype Consistency

**Risk:** Mixed precision training could cause dtype mismatches.

**Verification:**
- MASKED_LOGIT_VALUE = -1e4 is safe for FP16 (tested in test_tamiyo_network.py)
- Test explicitly checks softmax doesn't produce NaN/Inf in FP16/BF16

**Status:** No issue - explicitly tested.

### 4. Device Placement

**Risk:** Tensors created on wrong device.

**Verification:**
- `get_initial_hidden()` takes explicit device parameter
- All masks created with explicit `device=device`
- Network parameters determine device via `next(self._network.parameters()).device`

**Status:** No issue.

---

## Severity-Tagged Findings Summary

| Severity | File | Line(s) | Issue | Recommendation |
|----------|------|---------|-------|----------------|
| **P3** | `__init__.py` | 41 | Comment claims "backwards compatibility" which violates project policy | Remove misleading comment or remove truly legacy exports |
| **P4** | `factored_lstm.py` | 498-508, 613-625 | Style mask override logic duplicated | Consider factoring into helper (low priority) |

---

## Test Coverage Assessment

**Examined Tests:**
- `/home/john/esper-lite/tests/simic/test_tamiyo_network.py` (419 lines)
- `/home/john/esper-lite/tests/tamiyo/policy/test_lstm_bundle.py` (209 lines)

**Coverage Highlights:**
1. Forward pass shape verification
2. Hidden state propagation
3. Mask application (including FP16/BF16)
4. Per-head log probs and entropy normalization
5. Single-action edge case (num_slots=1, log(1)=0 division handling)
6. Deterministic mode
7. InvalidStateMachineError for all-false masks

**Potential Gaps:**
- No test for multi-layer LSTM (lstm_layers > 1)
- No explicit torch.compile test in this batch (may exist elsewhere)
- No gradient flow test through LSTM hidden state during training

---

## Recommendations

### Immediate (This PR)

1. **Remove backwards compatibility comment in `__init__.py`** (P3)

### Future Considerations

1. **Add torch.compile smoke test** for factored_lstm.py to catch graph breaks early
2. **Factor style mask override logic** into a shared helper to reduce duplication
3. **Consider adding gradient flow test** that verifies gradients propagate through LSTM hidden state during evaluate_actions()

---

## Conclusion

The Tamiyo networks implementation demonstrates strong PyTorch engineering:
- Correct LSTM patterns (shapes, initialization, hidden state management)
- Proper LayerNorm placement for training stability
- FP16-safe masking implementation
- Clean separation between inference (get_action) and training (evaluate_actions) paths
- Well-documented memory management requirements

The single P3 finding (backwards compatibility comment) is a documentation issue, not a code defect. The network implementation is production-ready for the Esper morphogenetic training system.
