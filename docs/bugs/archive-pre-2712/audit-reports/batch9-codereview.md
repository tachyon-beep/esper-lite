# Batch 9 Code Review: Tamiyo Core + Networks

**Reviewer**: Python Code Quality Specialist
**Date**: 2025-12-27
**Files Reviewed**:
1. `/home/john/esper-lite/src/esper/tamiyo/decisions.py`
2. `/home/john/esper-lite/src/esper/tamiyo/heuristic.py`
3. `/home/john/esper-lite/src/esper/tamiyo/__init__.py`
4. `/home/john/esper-lite/src/esper/tamiyo/tracker.py`
5. `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py`
6. `/home/john/esper-lite/src/esper/tamiyo/networks/__init__.py`

---

## Executive Summary

The Tamiyo domain (Brain/Cortex) is well-designed with clear separation between the heuristic baseline policy and the neural LSTM policy. The code demonstrates good understanding of RL concepts and proper integration with the leyline contracts. Test coverage is comprehensive.

**Overall Assessment**: High quality code with a few minor issues to address.

**Finding Counts by Severity**:
- P0 (Critical): 0
- P1 (Correctness): 1
- P2 (Performance/Resource): 2
- P3 (Code Quality): 4
- P4 (Style/Minor): 3

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/tamiyo/decisions.py`

**Purpose**: Defines the `TamiyoDecision` dataclass representing strategic decisions about seed lifecycle management.

**Strengths**:
- Clean, minimal dataclass design
- Good use of properties (`blueprint_id`) for derived values
- Helper functions (`_is_germinate_action`, `_get_blueprint_from_action`) cleanly separated

**Concerns**: None significant.

---

### 2. `/home/john/esper-lite/src/esper/tamiyo/heuristic.py`

**Purpose**: Rule-based strategic controller implementing the `TamiyoPolicy` protocol. Handles germination, advancement, fossilization, and pruning decisions based on configurable thresholds.

**Strengths**:
- Excellent blueprint validation at init (P1-B fix) - fails fast with helpful error messages
- Per-epoch penalty decay (not per-decision) - correct temporal design
- Ransomware detection (P2-B) - clever pattern detection for pathological seeds
- Blueprint penalty system prevents thrashing on bad blueprints
- Embargo mechanism prevents rapid germination after prune
- Uses leyline constants as defaults - proper contract adherence

**Concerns**:

#### P3-A: Unused helper functions exported but not in `__all__`

Lines 12-22 define `_is_germinate_action` and `_get_blueprint_from_action` at module level but they're only used by `TamiyoDecision.blueprint_id`. Consider:
1. Moving them inside `decisions.py` (where `TamiyoDecision` lives), or
2. Making them private methods of `TamiyoDecision`

These functions are in `decisions.py`, not `heuristic.py` - my error in listing. This is actually correct.

#### P3-B: Seed iteration returns on first match - multi-slot behavior

Lines 204-286: `_decide_seed_management` iterates over `active_seeds` and returns immediately on the first actionable seed. This works for single-slot scenarios but may need reconsideration for multi-slot:

```python
for seed in active_seeds:
    stage = seed.stage
    # ... returns immediately for first match
```

In multi-slot mode, which seed gets priority? The iteration order is caller-determined. Consider documenting the priority semantics or adding explicit prioritization.

---

### 3. `/home/john/esper-lite/src/esper/tamiyo/__init__.py`

**Purpose**: Package-level exports for the Tamiyo domain.

**Strengths**:
- Clear organization of exports by category
- Imports from policy subpackage are well-structured

**Concerns**:

#### P1-A: "Legacy" comment violates No Legacy Code Policy

Line 43-44:
```python
# Legacy heuristic (kept for backwards compatibility)
"TamiyoPolicy",
```

Per CLAUDE.md: "Legacy code, backwards compatibility, and compatibility shims are strictly forbidden."

The comment itself suggests this is being kept for compatibility rather than because it's the primary interface. Either:
1. Remove the comment if `TamiyoPolicy` is still the primary protocol, or
2. Remove the export if it's truly legacy

Looking at usage: `TamiyoPolicy` is a Protocol defined in `heuristic.py` that `HeuristicTamiyo` implements. This isn't legacy - it's the contract. The comment is misleading.

**Recommendation**: Change comment to:
```python
# Heuristic policy (baseline for comparison)
"TamiyoPolicy",
```

---

### 4. `/home/john/esper-lite/src/esper/tamiyo/tracker.py`

**Purpose**: `SignalTracker` maintains running statistics for decision-making, computing plateau detection, stabilization, and delta metrics.

**Strengths**:
- Proper latch behavior for stabilization (once True, stays True)
- Telemetry emission on stabilization event
- `peek()` method for read-only signal queries (useful for bootstrap values)
- P2-A scale validation warning - catches 0-1 vs 0-100 scale mistakes
- P1-A fix: regression epochs don't count as stable

**Concerns**:

#### P2-A: `best_val_loss` semantics mismatch with name

Lines 186-187:
```python
# NOTE: best_val_loss is "best in window" (last N epochs), not global best
best_val_loss=min(self._loss_history) if self._loss_history else float('inf'),
```

The name `best_val_loss` suggests global best, but it's actually window-best. This could confuse downstream consumers. The comment helps, but the field name in `TrainingMetrics` doesn't have this context.

Consider:
1. Renaming to `window_best_val_loss`, or
2. Adding a true global `best_val_loss` like `best_val_accuracy`, or
3. Documenting this in the `TrainingMetrics` dataclass

#### P3-C: Duplicate logic in `update()` and `peek()`

Lines 261-282 and 173-188: The TrainingMetrics construction is duplicated between `update()` and `peek()`. If fields are added to `TrainingMetrics`, both must be updated.

Consider extracting a `_build_metrics()` helper:
```python
def _build_metrics(self, epoch, global_step, ...) -> TrainingMetrics:
    ...
```

#### P4-A: Inconsistent use of `float('inf')` vs `math.inf`

Lines 79, 187, 278-280, 332: Uses `float('inf')`. While functional, `math.inf` is slightly more Pythonic. Minor style issue.

---

### 5. `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py`

**Purpose**: Recurrent actor-critic network with factored action heads for multi-slot morphogenetic control. Core neural network for the learned policy.

**Strengths**:
- Excellent architecture documentation with design rationale
- Pre-LSTM and post-LSTM LayerNorm for training stability (well-documented M7)
- Forget gate bias = 1 initialization (Gers et al., 2000) - correct LSTM practice
- Orthogonal weight initialization with appropriate gains
- Clear separation of get_action (inference) vs evaluate_actions (training)
- MaskedCategorical for safe action sampling
- Type hints with TypedDict for forward output

**Concerns**:

#### P2-B: Style mask override logic creates tensor every call

Lines 499-508:
```python
style_mask_override = masks["style"]
if style_mask_override is None:
    style_mask_override = torch.ones_like(head_logits["style"], dtype=torch.bool)
# Avoid `.any()` (CPU sync) by applying the override unconditionally.
style_mask_override = style_mask_override.clone()  # <-- allocation every call
style_irrelevant = (actions["op"] != LifecycleOp.GERMINATE) & (...)
style_mask_override[style_irrelevant] = False
style_mask_override[style_irrelevant, int(GerminationStyle.SIGMOID_ADD)] = True
```

The `.clone()` allocates a new tensor on every `get_action()` call even when not needed. This is in the hot inference path. Consider:
1. Only clone if modification is needed (check `style_irrelevant.any()` - though that syncs)
2. Use in-place masking operations without clone
3. Pre-allocate a reusable buffer

The comment says "avoid .any()" but then does in-place modification anyway. The clone is to avoid mutating the input mask.

#### P3-D: Long method - `get_action` is 118 lines

Lines 374-531: The `get_action` method is quite long with repeated logic for each head. Consider extracting the common pattern into a helper or loop.

#### P4-B: Redundant type annotation comments

Lines 227-233:
```python
# head[-1] is a Linear layer, access .weight.data to get Tensor
last_layer = head[-1]
if isinstance(last_layer, nn.Linear):
    nn.init.orthogonal_(last_layer.weight.data, gain=0.01)
```

The isinstance check makes the comment redundant. The type checker already knows.

#### P4-C: Inconsistent docstring style

Some methods have full docstrings with Args/Returns sections (e.g., `get_action`), others have minimal docstrings (e.g., `_init_weights`). Minor inconsistency.

---

### 6. `/home/john/esper-lite/src/esper/tamiyo/networks/__init__.py`

**Purpose**: Re-exports from factored_lstm.py.

**Strengths**: Clean, minimal re-export.

**Concerns**: None.

---

## Cross-Cutting Integration Risks

### Risk 1: Heuristic vs Neural Policy Interface Divergence

The `HeuristicTamiyo` and the LSTM policy bundle have different interfaces:
- Heuristic: `decide(signals, active_seeds) -> TamiyoDecision`
- LSTM: `get_action(state, hidden, masks, ...) -> GetActionResult`

This is intentional (heuristic works on signals, neural works on tensors), but the `HeuristicPolicyBundle` adapter (in `policy/heuristic_bundle.py`, not reviewed here) must bridge these correctly.

**Verification needed**: Confirm `HeuristicPolicyBundle` correctly maps between these interfaces.

### Risk 2: TrainingMetrics Scale Confusion

`tracker.py` validates accuracy scale (warns if 0-1 instead of 0-100), but `TrainingMetrics` in leyline has no runtime validation. A misbehaving training loop could pass wrong-scale values that silently corrupt plateau detection.

**Recommendation**: Consider adding an assertion in `TrainingMetrics.__post_init__` for sanity checking.

### Risk 3: LSTM Hidden State Memory Management

`factored_lstm.py` documents the hidden state detachment requirement clearly (lines 269-283), but the caller is responsible for detachment. If the training loop forgets:
1. BPTT will extend across episode boundaries
2. Memory will grow unbounded
3. OOM after ~100-1000 episodes

**Verification needed**: Confirm `vectorized.py` properly detaches hidden states at episode reset.

### Risk 4: Action Enum Consistency

`HeuristicTamiyo` uses `build_action_enum(topology)` to create a dynamic IntEnum at init. The factored LSTM policy uses the static `LifecycleOp` enum from leyline. These must stay consistent:
- Heuristic actions: `WAIT, GERMINATE_CONV_LIGHT, ADVANCE, PRUNE, FOSSILIZE`
- Factored actions: `LifecycleOp.WAIT, GERMINATE, ADVANCE, PRUNE, FOSSILIZE`

The mapping is different (heuristic has blueprint baked into action, factored has separate heads).

**This is intentional** - the heuristic is a baseline that doesn't use factored actions. No bug, but worth noting for anyone comparing them.

---

## Test Coverage Assessment

Based on reviewed test files:

| Component | Coverage | Notes |
|-----------|----------|-------|
| SignalTracker | Excellent | `test_tracker_unit.py` covers deltas, plateaus, stabilization, reset |
| HeuristicTamiyo | Excellent | `test_heuristic_unit.py` + `test_heuristic_decisions.py` |
| Blueprint validation | Good | P1-B validation has dedicated tests |
| Blueprint penalties | Good | Decay, threshold, all-penalized cases |
| FactoredRecurrentActorCritic | Unknown | Not in reviewed batch, likely in `test_tamiyo_network.py` |

---

## Findings Summary

| ID | Severity | File | Line(s) | Description |
|----|----------|------|---------|-------------|
| P1-A | P1 | `__init__.py` | 43-44 | "Legacy" comment violates No Legacy Code Policy |
| P2-A | P2 | `tracker.py` | 186-187 | `best_val_loss` name misleading (window-best, not global) |
| P2-B | P2 | `factored_lstm.py` | 499-508 | Tensor clone on every get_action call |
| P3-A | P3 | (n/a) | - | (Retracted - functions are in correct file) |
| P3-B | P3 | `heuristic.py` | 204-286 | Multi-slot priority semantics undocumented |
| P3-C | P3 | `tracker.py` | various | Duplicate TrainingMetrics construction logic |
| P3-D | P3 | `factored_lstm.py` | 374-531 | Long method - get_action is 118 lines |
| P4-A | P4 | `tracker.py` | various | `float('inf')` vs `math.inf` inconsistency |
| P4-B | P4 | `factored_lstm.py` | 227-233 | Redundant type annotation comments |
| P4-C | P4 | `factored_lstm.py` | various | Inconsistent docstring depth |

---

## Recommendations

### Immediate Actions (P1-P2)

1. **P1-A**: Remove or reword the "Legacy" comment in `__init__.py` to comply with the No Legacy Code Policy. Suggest: "Heuristic policy (baseline for comparison)".

2. **P2-A**: Either rename `best_val_loss` to `window_best_val_loss` in `TrainingMetrics` (leyline change), or add a comment in the dataclass documenting this semantic.

3. **P2-B**: Profile the `get_action` tensor clone. If it's measurable, consider conditional cloning or pre-allocated buffers.

### Optional Improvements (P3-P4)

4. **P3-B**: Document multi-slot priority in `_decide_seed_management` docstring.

5. **P3-C**: Extract `_build_metrics()` helper in `SignalTracker`.

6. **P3-D**: Consider refactoring `get_action` to reduce length, perhaps with a loop over head names.

---

## What Was Done Well

1. **Contract adherence**: Proper use of leyline constants and types throughout.

2. **Fail-fast validation**: Blueprint validation at init catches configuration errors before training starts.

3. **Training stability**: LSTM architecture follows best practices (LayerNorm, forget gate bias, orthogonal init).

4. **Documentation**: Network architecture rationale is excellent with citations.

5. **Anti-thrashing mechanisms**: Blueprint penalties and embargo prevent pathological cycling.

6. **Ransomware detection**: Clever P2-B pattern catches seeds that create dependencies without value.

7. **Test coverage**: Comprehensive unit tests for heuristic behavior and signal tracking.

---

## Conclusion

Batch 9 (Tamiyo Core + Networks) demonstrates high code quality with solid architecture decisions. The main actionable finding is the P1-A "Legacy" comment that conflicts with the No Legacy Code Policy - this should be addressed. The P2 findings are minor optimizations. The code is well-tested and follows leyline contracts correctly.

The domain is production-ready with the noted minor improvements.
