# CRITICAL BUG: Op Value/Action Mismatch in Q(s, op) Implementation

**Status**: CRITICAL - Blocks Phase 7
**Discovered**: 2025-12-31 (post-Phase 6 release review)
**Severity**: Undermines Q(s, op) design, causes biased advantages

---

## Executive Summary

The Q(s, op) value head implementation in `FactoredRecurrentActorCritic.get_action()` computes the value using a **different op** than the op action stored in the buffer. This creates systematic bias in PPO advantage estimates.

**Root Cause**: `forward()` samples the op for value computation, but `get_action()` independently samples (or argmaxes) the op for the action. These two ops frequently diverge.

---

## Detailed Bug Description

### Code Flow

1. **`get_action()` calls `forward()`** (line 528-541)
2. **`forward()` samples op STOCHASTICALLY** (line 467):
   ```python
   op_dist = MaskedCategorical(logits=op_logits_flat, mask=op_mask_flat)
   sampled_op_flat = op_dist.sample()  # ALWAYS samples (never argmax)
   ```

3. **`forward()` computes value with sampled op** (line 471):
   ```python
   value = self._compute_value(lstm_out, sampled_op)  # Q(s, sampled_op)
   ```

4. **`get_action()` samples op INDEPENDENTLY** (line 618):
   ```python
   _sample_head("op")  # Samples or argmaxes INDEPENDENTLY
   ```

5. **`_sample_head` implementation** (line 609-616):
   ```python
   if deterministic:
       action = dist.masked_logits.argmax(dim=-1)  # Argmax
   else:
       action = dist.sample()  # Independent sample
   actions[key] = action
   ```

6. **`get_action()` returns value from `forward()`** (line 640):
   ```python
   value = output["value"][:, 0]  # Q(s, sampled_op_from_forward)
   ```

### The Mismatch

| Context | Op for Value Computation | Op Stored as Action | Result |
|---------|-------------------------|---------------------|--------|
| **Rollout** (deterministic=False) | `forward()` sample #1 | `get_action()` sample #2 | ❌ Two independent random draws |
| **Bootstrap** (deterministic=True) | `forward()` sample | `get_action()` argmax | ❌ Sampled op ≠ argmax op |

**The stored buffer entry**:
```python
buffer.add(
    value=Q(s, op_from_forward_sample),  # Line 640
    op_action=op_from_get_action,        # Line 618
    ...
)
```

These two ops are often **different**, violating the Q(s, op) design.

---

## Impact on Training

### 1. Biased Advantage Estimates

**During PPO update** (`ppo.py` line 505-511):
```python
result = self.policy.evaluate_actions(
    states,
    blueprint_indices,
    actions,  # actions["op"] = op_from_get_action
    masks,
    hidden,
)
# Returns Q(s, actions["op"]) = Q(s, op_from_get_action)
```

**But the buffer contains**:
- `value = Q(s, op_from_forward_sample)`  ← Different op!
- `action["op"] = op_from_get_action`

**Advantage computation**:
```python
# In rollout_buffer.py compute_advantages():
delta = rewards[t] + gamma * next_value - values[t]
#                                          ^^^^^^^^
#                                          Q(s, op_from_forward_sample)
```

But we're training the policy to maximize Q(s, op_from_get_action), not the op used in the value baseline.

### 2. Bootstrap Value Corruption

**Bootstrap computation** (`vectorized.py` line 3231-3237):
```python
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    blueprint_indices=post_action_bp_indices,
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=True,  # ← Uses argmax for action, sample for value
)
bootstrap_values = bootstrap_result.value.cpu().tolist()
```

**What this returns**:
- `value = Q(s', op_from_forward_sample)` ← Random sample
- `action["op"] = argmax_op` ← Deterministic argmax

The bootstrap value should approximate:
```
V(s') = E_a[Q(s', a)] ≈ Q(s', argmax_a Q(s', a))
```

But we're getting `Q(s', random_sample)` instead, which:
1. Has higher variance (not the expected value)
2. Is biased (random sample ≠ argmax)
3. Corrupts the final GAE advantage estimate

### 3. Credit Assignment Failure

The policy is optimized to select `op_from_get_action`, but the advantage signal is computed using `Q(s, op_from_forward_sample)`.

When these ops differ:
- The policy gradient pushes toward increasing log_prob(op_from_get_action)
- But the advantage measures the value of op_from_forward_sample
- **Result**: The policy learns to select actions based on unrelated value estimates

---

## Why This Wasn't Caught

### DRL Expert Review Statements

The DRL expert review validated:
> "**Rollout**: forward() samples op, computes Q(s, sampled_op) ✅"
> "**PPO Update**: evaluate_actions() uses stored op, computes Q(s, stored_op) ✅"
> "**Bootstrap**: get_action() samples op, computes Q(s, sampled_op) ✅"

**The Missed Detail**:
- The review assumed `get_action()` samples the op **once**
- But `get_action()` actually samples the op **twice**:
  1. Inside `forward()` for value computation
  2. Inside `_sample_head("op")` for the action

The review verified that each *individual* path was self-consistent, but missed that the **op used for the value differs from the op stored as the action**.

### PyTorch Expert Review

The PyTorch expert verified tensor shapes and op-conditioning implementation but didn't trace the control flow showing two independent samples.

### Code Reviewer

The code reviewer verified that `forward()` and `evaluate_actions()` both use op-conditioning correctly but didn't identify the double-sampling in `get_action()`.

---

## Reproduction

### Minimal Example

```python
import torch
from esper.tamiyo.networks import FactoredRecurrentActorCritic

net = FactoredRecurrentActorCritic(state_dim=126, num_slots=3)
state = torch.randn(1, 1, 126)
bp_idx = torch.randint(0, 13, (1, 1, 3))
masks = {k: None for k in HEAD_NAMES}

# Call get_action with deterministic=False
result1 = net.get_action(state[:, 0, :], bp_idx[:, 0, :], masks, deterministic=False)

# The value is conditioned on output["sampled_op"] from forward()
# But the action["op"] is from _sample_head("op")
# These are independent samples - print to verify:
print(f"Op used for value (from forward): {result1.sampled_op}")
print(f"Op stored as action: {result1.actions['op']}")
# These will often differ!

# With deterministic=True (bootstrap case):
result2 = net.get_action(state[:, 0, :], bp_idx[:, 0, :], masks, deterministic=True)
print(f"Op used for value (sample): {result2.sampled_op}")
print(f"Op stored as action (argmax): {result2.actions['op']}")
# These will almost always differ!
```

---

## Recommended Fix

### Option A: Reuse `sampled_op` from `forward()` (Preferred)

Modify `get_action()` to use the op from `forward()` for both value and action:

```python
# In get_action() around line 618:
# REMOVE:
# _sample_head("op")

# ADD:
if deterministic:
    # For deterministic mode, use argmax
    op_dist = MaskedCategorical(logits=head_logits["op"], mask=masks["op"])
    actions["op"] = op_dist.masked_logits.argmax(dim=-1)
    log_probs["op"] = op_dist.log_prob(actions["op"])

    # CRITICAL: Recompute value with argmax op for consistency
    # Need to expose lstm_out from forward() for this
    value = self._compute_value(
        lstm_out=output["lstm_out"][:, 0, :],
        op=actions["op"]
    )
else:
    # For stochastic mode, reuse sampled_op from forward()
    actions["op"] = sampled_op
    op_dist = MaskedCategorical(logits=head_logits["op"], mask=masks["op"])
    log_probs["op"] = op_dist.log_prob(actions["op"])
    # Value already computed with this op in forward()
    value = output["value"][:, 0]
```

**Required changes**:
1. Expose `lstm_out` in `forward()` return dict
2. Modify `get_action()` to recompute value when `deterministic=True`
3. Remove `_sample_head("op")` call

### Option B: Pass `deterministic` flag to `forward()`

Make `forward()` respect the deterministic flag:

```python
def forward(
    self,
    state: torch.Tensor,
    blueprint_indices: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    slot_mask: torch.Tensor | None = None,
    # ... other masks ...
    deterministic: bool = False,  # NEW
) -> _ForwardOutput:
    # ... existing code ...

    # Sample or argmax based on deterministic flag
    op_dist = MaskedCategorical(logits=op_logits_flat, mask=op_mask_flat)
    if deterministic:
        sampled_op_flat = op_dist.masked_logits.argmax(dim=-1)
    else:
        sampled_op_flat = op_dist.sample()

    sampled_op = sampled_op_flat.reshape(batch_size, seq_len)
    value = self._compute_value(lstm_out, sampled_op)
    # ... rest of code ...
```

Then in `get_action()`, simply reuse `output["sampled_op"]`:

```python
# In get_action():
output = self._network.forward(
    state_3d,
    blueprint_indices_3d,
    hidden,
    # ... masks ...
    deterministic=deterministic,  # Pass through
)

# Reuse sampled_op for action (no independent sampling)
actions["op"] = sampled_op
op_dist = MaskedCategorical(logits=head_logits["op"], mask=masks["op"])
log_probs["op"] = op_dist.log_prob(actions["op"])
```

**Pros**: Simpler, single source of op
**Cons**: Breaks encapsulation (forward() becomes aware of sampling strategy)

---

## Testing Requirements

After implementing the fix:

1. **Unit test**: Verify `get_action()` uses same op for value and action
   ```python
   def test_op_value_action_consistency():
       net = FactoredRecurrentActorCritic(state_dim=126, num_slots=3)
       state = torch.randn(1, 126)
       bp_idx = torch.randint(0, 13, (1, 3))
       masks = {k: None for k in HEAD_NAMES}

       # Deterministic mode
       result = net.get_action(state, bp_idx, masks, deterministic=True)
       assert torch.equal(result.sampled_op, result.actions["op"])

       # Stochastic mode (with seed for reproducibility)
       torch.manual_seed(42)
       result = net.get_action(state, bp_idx, masks, deterministic=False)
       assert torch.equal(result.sampled_op, result.actions["op"])
   ```

2. **Integration test**: Verify bootstrap values use correct op
   ```python
   def test_bootstrap_value_consistency():
       # Bootstrap should use argmax for both value and action
       # Value = Q(s', argmax_op), not Q(s', random_sample)
       pass  # Implement after fix
   ```

3. **Regression test**: Train for 10 episodes, verify advantages are reasonable
   - Check that value estimates align with rewards
   - Verify no systematic bias in advantages

---

## References

- Original design: `docs/plans/tamiyo_next/tamiyo_next.md` (Phase 4)
- Implementation: `src/esper/tamiyo/networks/factored_lstm.py`
- Usage sites:
  - `src/esper/simic/training/vectorized.py:2495` (rollout)
  - `src/esper/simic/training/vectorized.py:3231` (bootstrap)

---

## Priority

**P0 CRITICAL** - Must fix before Phase 7 validation.

This bug fundamentally breaks the Q(s, op) design and causes biased advantage estimates in all training runs since Phase 4 implementation.
