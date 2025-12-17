# Deep Code Review: PPO Agent (`simic/agent/ppo.py`)

**Reviewer**: PyTorch/DRL Specialist
**Date**: 2025-12-17
**File**: `/home/john/esper-lite/src/esper/simic/agent/ppo.py` (808 lines)
**PyTorch Version Baseline**: 2.9.0

---

## Executive Summary

The PPO agent implementation is **well-architected** and demonstrates strong understanding of both PPO best practices and PyTorch 2.x idioms. The codebase correctly handles:

- Factored action spaces with per-head credit assignment
- Recurrent policy with LSTM hidden state management
- torch.compile integration with appropriate escape hatches
- Numerical stability (using -1e4 mask values, proper clipping)

**Overall Assessment**: Production-ready with minor improvements possible.

**Critical Issues**: 0
**High Priority Issues**: 2
**Medium Priority Issues**: 4
**Low Priority Suggestions**: 5

---

## Critical Issues (Bugs, Correctness)

**None found.**

The PPO implementation correctly handles:
- GAE computation with proper truncation/terminal handling (delegated to buffer)
- KL divergence approximation using the Schulman KL3 estimator
- Per-head ratio computation and clipping
- Early stopping on KL divergence (checked before optimizer step, not after)

---

## High Priority Issues (Performance, Stability)

### H1. Per-Head Gradient Norm Computation Uses Compiled Network Attributes

**Location**: Lines 616-630
**Severity**: High (runtime error under torch.compile)

```python
for head_name, head_module in [
    ("slot", self.network.slot_head),
    ("blueprint", self.network.blueprint_head),
    ("blend", self.network.blend_head),
    ("op", self.network.op_head),
    ("value", self.network.value_head),
]:
```

**Problem**: When `compile_network=True`, `self.network` is an `OptimizedModule`. Accessing `.slot_head` etc. directly works but returns compiled submodules with potentially different iteration behavior for `.parameters()`.

**Risk**: The `_base_network` property exists (line 341-353) but is not used here. If torch.compile wraps submodules differently in future versions, this could break or return incorrect gradient statistics.

**Recommendation**: Use `self._base_network` consistently:

```python
for head_name, head_module in [
    ("slot", self._base_network.slot_head),
    ("blueprint", self._base_network.blueprint_head),
    # ...
]:
```

---

### H2. Entropy Coefficient Computed Without Action Mask Context

**Location**: Line 606
**Severity**: High (ineffective adaptive entropy floor)

```python
entropy_coef = self.get_entropy_coef()
```

**Problem**: `get_entropy_coef()` supports an optional `action_mask` parameter for adaptive entropy floor scaling (lines 355-415), but it's called without any mask during the update loop. The adaptive floor is designed to scale exploration incentive when few actions are valid, but this feature is never activated in practice.

**Context**: The method signature shows the feature was intended:
```python
def get_entropy_coef(self, action_mask: torch.Tensor | None = None) -> float:
```

**Impact**: When `adaptive_entropy_floor=True`, the floor scaling is never applied because no mask is passed.

**Recommendation**: Either:
1. Remove the adaptive floor feature if not used (dead code)
2. Pass a representative mask (e.g., mean of op_masks across batch) if the feature should be active

---

## Medium Priority Issues (Best Practices)

### M1. Redundant `.item()` Calls Inside Training Loop

**Location**: Lines 637-642
**Severity**: Medium (unnecessary CPU syncs)

```python
metrics["ratio_mean"].append(joint_ratio.mean().item())
metrics["ratio_max"].append(joint_ratio.max().item())
metrics["ratio_min"].append(joint_ratio.min().item())
```

**Problem**: Three separate `.item()` calls force three GPU-CPU synchronization points. With small tensors this is negligible, but it's a pattern that can accumulate.

**Recommendation**: Batch into single sync:
```python
ratio_stats = torch.stack([joint_ratio.mean(), joint_ratio.max(), joint_ratio.min()]).tolist()
metrics["ratio_mean"].append(ratio_stats[0])
metrics["ratio_max"].append(ratio_stats[1])
metrics["ratio_min"].append(ratio_stats[2])
```

---

### M2. Value Loss Clipping Uses Different Range Than Policy Clipping

**Location**: Lines 583-591
**Severity**: Medium (intentional but undocumented trade-off)

```python
if self.clip_value:
    # Use separate value_clip (not policy clip_ratio) since value scale differs
    values_clipped = valid_old_values + torch.clamp(
        values - valid_old_values, -self.value_clip, self.value_clip
    )
```

**Context**: `DEFAULT_VALUE_CLIP = 10.0` vs `DEFAULT_CLIP_RATIO = 0.2`. This is actually good practice (value predictions can range wider than policy ratios), but the comment at line 583 is the only documentation.

**Recommendation**: Add hyperparameter documentation in `__init__` docstring or consider removing value clipping entirely. Research by Engstrom et al. (2020) "Implementation Matters" found value clipping often hurts performance. The existing comment at line 208 already notes this.

---

### M3. Mixed Type Annotations for `policy_loss`

**Location**: Lines 565, 579
**Severity**: Medium (type checker confusion)

```python
policy_loss = 0.0  # float
# ...
policy_loss = policy_loss + head_loss  # now Tensor
```

**Problem**: `policy_loss` starts as `float` and becomes `Tensor` after the first addition. This works but confuses type checkers and can cause issues if someone adds type hints later.

**Recommendation**: Initialize as tensor:
```python
policy_loss = torch.tensor(0.0, device=self.device)
```

Or use torch.zeros with proper device placement.

---

### M4. Checkpoint Loading Warns on Legacy Version but Doesn't Upgrade

**Location**: Lines 758-766
**Severity**: Medium (warning without action)

```python
if version == 0:
    warnings.warn(
        f"Loading legacy checkpoint (version 0) from {path}. "
        "Slot configuration will default to 3 slots. "
        "Re-save checkpoint to upgrade format.",
        DeprecationWarning,
        stacklevel=2
    )
```

**Problem**: The warning says "Re-save checkpoint to upgrade format" but there's no automated upgrade path or documentation on how to do this.

**Recommendation**: Either:
1. Add an `upgrade_checkpoint()` method or CLI command
2. Auto-save upgraded checkpoint with `.v1` suffix
3. Update warning to explain manual process

---

## Low Priority Suggestions

### L1. `signals_to_features` Has Complex Parameter List

**Location**: Lines 59-70
**Severity**: Low (readability)

```python
def signals_to_features(
    signals,
    *,
    slot_reports: dict[str, "SeedStateReport"],
    use_telemetry: bool = True,
    max_epochs: int = 200,
    slots: list[str] | None = None,
    total_params: int = 0,
    total_seeds: int = 0,
    max_seeds: int = 0,
    slot_config: "SlotConfig | None" = None,
) -> list[float]:
```

**Suggestion**: Consider a `FeatureExtractionContext` dataclass to bundle related parameters.

---

### L2. Unused `max_epochs` Parameter in `signals_to_features`

**Location**: Line 64
**Severity**: Low (potential dead code)

```python
max_epochs: int = 200,
```

**Observation**: This parameter is never used in the function body. It may be vestigial from removed normalization code.

**Suggestion**: Remove if not needed, or add usage for epoch-based normalization.

---

### L3. `recurrent_n_epochs` Default of 1 May Be Too Conservative

**Location**: Line 242
**Severity**: Low (tuning opportunity)

```python
self.recurrent_n_epochs = recurrent_n_epochs if recurrent_n_epochs is not None else 1
```

**Context**: The comment at line 240 explains why (hidden state staleness), but 1 epoch is quite conservative. Recent work on recurrent PPO (e.g., PureJaxRL) shows 2-4 epochs work well with proper hidden state handling.

**Suggestion**: Consider 2-3 epochs as default with appropriate KL early stopping.

---

### L4. Ratio Diagnostic Only Captures Op Head Ratios

**Location**: Lines 647-655
**Severity**: Low (incomplete diagnostics)

```python
diag = RatioExplosionDiagnostic.from_batch(
    ratio=joint_ratio.flatten(),
    old_log_probs=old_log_probs["op"].flatten(),
    new_log_probs=log_probs["op"].flatten(),
    actions=valid_op_actions.flatten(),
    # ...
)
```

**Observation**: When ratio explosion is detected, only the `op` head is diagnosed. If the explosion originated in `slot`, `blueprint`, or `blend` heads, this diagnostic would miss the root cause.

**Suggestion**: Consider capturing diagnostics for the head with the worst ratio, not just `op`.

---

### L5. `train_steps` Only Incremented Once Per Update

**Location**: Line 657
**Severity**: Low (semantic clarity)

```python
self.train_steps += 1
```

**Observation**: `train_steps` is incremented per `update()` call, not per PPO epoch within the update. This is fine but the variable name `train_steps` could be confused with `ppo_epochs_total`.

**Suggestion**: Rename to `update_count` or `rollout_count` for clarity, or add docstring clarification.

---

## Code Quality Observations

### Strengths

1. **Correct KL Early Stopping** (lines 528-561): The KL check happens BEFORE `optimizer.step()`, not after. This is the correct implementation - the comment at line 529 explicitly documents this as a bug fix (BUG-003).

2. **Per-Head Causal Masking** (lines 507-514): The `head_masks` dict correctly implements the causal structure where blueprint/blend only matter for GERMINATE actions. This reduces gradient noise significantly.

3. **Masked Mean for Policy Loss** (lines 576-578): Using masked mean instead of regular mean prevents bias from zero-padded positions:
   ```python
   n_valid = mask.sum().clamp(min=1)
   head_loss = -(clipped_surr * mask.float()).sum() / n_valid
   ```

4. **Proper torch.compile Integration** (lines 296-300): Using `mode="default"` is appropriate for networks with complex control flow. The `MaskedCategorical._validate_action_mask` is correctly decorated with `@torch.compiler.disable`.

5. **Weight Decay Actor/Critic Separation** (lines 311-334): Correctly applies weight decay only to critic, not actor or shared layers. This is critical for PPO - weight decay on actor heads kills exploration.

6. **Fused/Foreach Optimizer** (lines 302-308): Correctly uses `fused=True` for CUDA and `foreach=True` for CPU. This is PyTorch 2.x best practice for AdamW.

### Areas of Excellence

- **Numerical stability**: Using -1e4 for mask values (not -inf or dtype.min)
- **Memory efficiency**: Pre-allocated rollout buffer with direct indexing
- **Telemetry integration**: RatioExplosionDiagnostic captures actionable debug info
- **Checkpoint versioning**: Forward-compatible checkpoint format with version field

---

## Integration Notes

The PPO agent integrates cleanly with:

- **`rollout_buffer.py`**: Buffer correctly handles per-environment GAE computation with `compute_advantages_and_returns()`. The `@torch.compiler.disable` decorator on this method is appropriate since it uses Python loops.

- **`network.py`**: The `FactoredRecurrentActorCritic` architecture is well-designed. The mask value of -1e4 (defined as `_MASK_VALUE`) is consistent with HuggingFace Transformers conventions.

- **`advantages.py`**: The `compute_per_head_advantages()` function correctly implements causal masking for the factored action space.

- **`action_masks.py`** (in tamiyo.policy): The `MaskedCategorical` class correctly handles entropy normalization and has proper validation with `@torch.compiler.disable` escape hatches.

---

## Recommendations Summary

| Priority | Issue | Action |
|----------|-------|--------|
| H1 | Compiled network attribute access | Use `_base_network` in gradient norm loop |
| H2 | Unused adaptive entropy floor | Pass mask or remove feature |
| M1 | Multiple `.item()` syncs | Batch ratio stats |
| M2 | Value clip documentation | Document or reconsider feature |
| M3 | Mixed type for policy_loss | Initialize as tensor |
| M4 | Legacy checkpoint warning | Add upgrade path |

---

## Files Reviewed

- `/home/john/esper-lite/src/esper/simic/agent/ppo.py` (primary)
- `/home/john/esper-lite/src/esper/simic/agent/rollout_buffer.py`
- `/home/john/esper-lite/src/esper/simic/agent/network.py`
- `/home/john/esper-lite/src/esper/simic/agent/advantages.py`
- `/home/john/esper-lite/src/esper/simic/agent/types.py`
- `/home/john/esper-lite/src/esper/simic/telemetry/debug_telemetry.py`
- `/home/john/esper-lite/src/esper/tamiyo/policy/action_masks.py`
- `/home/john/esper-lite/src/esper/leyline/__init__.py` (constants)
