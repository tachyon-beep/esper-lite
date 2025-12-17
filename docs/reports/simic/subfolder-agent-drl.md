# Deep RL Review: simic/agent/ Subfolder

**Reviewer**: DRL Specialist Agent
**Date**: 2025-12-17
**Scope**: `/home/john/esper-lite/src/esper/simic/agent/`
**Files Reviewed**: ppo.py (808 lines), rollout_buffer.py (426 lines), network.py (363 lines), advantages.py (73 lines), types.py (86 lines), __init__.py (51 lines)

---

## Executive Summary

The simic/agent/ subfolder implements a well-engineered PPO agent with a **factored recurrent architecture** designed for multi-slot seed lifecycle control. The implementation demonstrates strong understanding of DRL best practices including:

- Correct PPO clipping implementation with separate value clip
- Proper GAE computation with per-environment isolation
- Appropriate recurrent architecture with LSTM and LayerNorm
- Sophisticated factored action space with causal masking

**Overall Assessment**: This is a **high-quality implementation** suitable for production use. The issues identified are primarily medium-priority refinements rather than critical bugs.

### Issue Summary

| Priority | Count | Description |
|----------|-------|-------------|
| Critical | 0 | No algorithm-breaking bugs found |
| High | 3 | Training stability concerns |
| Medium | 6 | Best practice improvements |
| Low | 5 | Minor refinements and suggestions |

---

## Critical Issues

**None found.** The core PPO algorithm, GAE computation, and action distribution handling are all implemented correctly.

---

## High-Priority Issues (Training Stability)

### H1. Recurrent PPO with Multiple Epochs Risk

**File**: `ppo.py` (lines 241-242)
**Issue**: While the code defaults `recurrent_n_epochs=1` for LSTM safety, the docstring mentions users can increase this. Multiple epochs with recurrent networks cause hidden state staleness since the stored hidden states were computed with a different policy.

```python
# Recurrent PPO with multiple epochs can cause hidden state staleness (policy drift)
# Default to 1 epoch for LSTM safety; increase with caution
self.recurrent_n_epochs = recurrent_n_epochs if recurrent_n_epochs is not None else 1
```

**Risk**: If users set `recurrent_n_epochs > 1`, the policy used to generate the rollout diverges from the current policy, but hidden states remain from the old policy. This can cause value estimation errors and policy gradient bias.

**Recommendation**: Either:
1. Hard-cap `recurrent_n_epochs` to 1 for LSTM architectures
2. Add a warning when `recurrent_n_epochs > 1` is used
3. Implement hidden state replay (re-forward through trajectory each epoch)

---

### H2. Entropy Coefficient Per-Head May Cause Gradient Imbalance

**File**: `ppo.py` (lines 252-258, 600-604)
**Issue**: Per-head entropy coefficients are applied as simple multipliers without normalization across heads. If one head has significantly lower entropy (e.g., blueprint head only active during GERMINATE), its gradient contribution may be drowned out by other heads.

```python
for key, ent in entropy.items():
    head_coef = self.entropy_coef_per_head.get(key, 1.0)
    entropy_loss = entropy_loss - head_coef * ent.mean()
```

**Risk**: Gradient starvation for infrequently-used heads (blueprint, blend).

**Recommendation**: Consider normalizing entropy contributions by action frequency or using separate optimizers/learning rates per head.

---

### H3. KL Divergence Computation for Factored Actions

**File**: `ppo.py` (lines 535-544)
**Issue**: The joint KL is computed as `sum` of per-head KLs. While mathematically valid for independent factorizations, the heads are NOT truly independent (blueprint/blend only matter when op=GERMINATE). This inflates KL estimates when masked heads have zero advantage but non-zero KL.

```python
for key in HEAD_NAMES:
    mask = head_masks[key]
    log_ratio = log_probs[key] - old_log_probs[key]
    kl_per_step = (torch.exp(log_ratio) - 1) - log_ratio
    n_valid = mask.sum().clamp(min=1)
    head_kl = (kl_per_step * mask.float()).sum() / n_valid
    head_kls.append(head_kl)
approx_kl = torch.stack(head_kls).sum().item()
```

**Risk**: Premature early stopping due to inflated KL from heads that didn't causally affect the reward.

**Recommendation**: Weight KL contribution by causal relevance (similar to how advantages are masked) or only include KL from causally-active heads per timestep.

---

## Medium-Priority Issues (Best Practices)

### M1. Value Loss Scale Mismatch with Policy Loss

**File**: `ppo.py` (lines 581-593)
**Issue**: Value loss uses MSE which has squared scale, while policy loss is linear. With `value_coef=0.5`, the value gradients may dominate early training when value estimates are far from returns.

```python
if self.clip_value:
    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
else:
    value_loss = F.mse_loss(values, valid_returns)
```

**Recommendation**: Consider Huber loss for value function to bound gradient magnitude, or reduce `value_coef` to 0.25-0.3.

---

### M2. Advantage Normalization Timing

**File**: `rollout_buffer.py` (lines 315-338)
**Issue**: Advantages are normalized globally after GAE computation but before per-head masking. This means causally-irrelevant zeros (from masked heads) don't affect the statistics, which is correct. However, the normalization uses a hardcoded epsilon of `1e-8`.

```python
self.advantages[env_id, :num_steps] = (
    self.advantages[env_id, :num_steps] - mean
) / (std + 1e-8)
```

**Risk**: If `std` is very small (near-constant rewards), division by small number can create unstable advantages.

**Recommendation**: Use a minimum std threshold (e.g., `max(std, 1e-3)`) rather than just adding epsilon.

---

### M3. Missing Gradient Norm Logging After Clipping

**File**: `ppo.py` (lines 612-632)
**Issue**: Per-head gradient norms are collected BEFORE clipping (which is useful), but the post-clipping total gradient norm is not logged. This makes it harder to diagnose if the clip threshold is too aggressive.

```python
# Collect per-head gradient norms BEFORE clipping (P4-6)
# ... (collection code)

nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
self.optimizer.step()
```

**Recommendation**: Capture the return value of `clip_grad_norm_` (which returns the total norm before clipping) and log it:
```python
total_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
metrics["total_grad_norm"].append(total_norm.item())
```

---

### M4. Rollout Buffer LSTM Hidden State Memory

**File**: `rollout_buffer.py` (lines 176-178)
**Issue**: Hidden states are stored at every timestep: `[num_envs, max_steps, lstm_layers, hidden_dim]`. For 4 envs x 25 steps x 1 layer x 128 hidden = 12,800 floats per state (h and c) = ~100KB. This is acceptable but inefficient since only the initial hidden state per episode is needed for replay.

```python
# LSTM hidden states: [num_envs, max_steps, lstm_layers, hidden_dim]
self.hidden_h = torch.zeros(n, m, self.lstm_layers, self.lstm_hidden_dim, device=device)
self.hidden_c = torch.zeros(n, m, self.lstm_layers, self.lstm_hidden_dim, device=device)
```

**Recommendation**: For memory efficiency, consider storing only initial episode hidden states plus a flag to re-compute during `evaluate_actions`. The current approach is fine for 4 envs but may not scale.

---

### M5. Action Mask Validation Location

**File**: `action_masks.py` (lines 285-297)
**Issue**: `MaskedCategorical._validate_action_mask` is decorated with `@torch.compiler.disable` but is called every time a distribution is constructed. This is correct for safety but adds latency.

```python
@torch.compiler.disable
def _validate_action_mask(mask: torch.Tensor) -> None:
    valid_count = mask.sum(dim=-1)
    if (valid_count == 0).any():
        raise InvalidStateMachineError(...)
```

**Recommendation**: Consider making validation configurable (disabled in production) or moving to a debug-only code path.

---

### M6. Network Weight Initialization Gain Values

**File**: `network.py` (lines 134-156)
**Issue**: Output layer initialization uses `gain=0.01` for policy heads and `gain=1.0` for value head. While small policy init is standard (prevents early overconfident actions), `gain=0.01` may be too aggressive for LSTM-based policies where the signal needs to propagate through multiple layers.

```python
for head in [self.slot_head, self.blueprint_head, self.blend_head, self.op_head]:
    nn.init.orthogonal_(head[-1].weight, gain=0.01)
nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
```

**Recommendation**: Consider `gain=0.1` for policy heads when using recurrent architectures, or make this configurable.

---

## Low-Priority Suggestions

### L1. Feature Extraction Layer Count

**File**: `network.py` (lines 71-76)
**Observation**: The feature extraction network is a single linear layer with LayerNorm and ReLU. For the 80-dim observation space (50 base + 30 telemetry), this may be insufficient capacity.

```python
self.feature_net = nn.Sequential(
    nn.Linear(state_dim, feature_dim),
    nn.LayerNorm(feature_dim),
    nn.ReLU(),
)
```

**Suggestion**: Consider a 2-layer MLP for feature extraction: `state_dim -> 256 -> feature_dim` with residual connection.

---

### L2. Checkpoint Version Documentation

**File**: `ppo.py` (lines 51-52)
**Observation**: `CHECKPOINT_VERSION = 1` is defined but there's no documentation of what changed between versions or what version 0 contained.

**Suggestion**: Add a comment block documenting version history:
```python
# CHECKPOINT_VERSION history:
# v0: Legacy (pre-SlotConfig), inferred architecture from weights
# v1: Added slot_config serialization, explicit state_dim
```

---

### L3. Type Annotation Completeness

**File**: `types.py`
**Observation**: `PPOUpdateMetrics` TypedDict includes some fields as `list[float]` but the actual implementation returns single floats after aggregation.

```python
class PPOUpdateMetrics(TypedDict):
    policy_loss: list[float]  # But returned as float after aggregation
```

**Suggestion**: Create separate TypedDict for intermediate metrics vs. final aggregated metrics.

---

### L4. Head Hidden Dimension Calculation

**File**: `network.py` (line 103)
**Observation**: Head hidden dimension is hardcoded as `lstm_hidden_dim // 2`. This may be suboptimal for different `lstm_hidden_dim` values.

```python
head_hidden = lstm_hidden_dim // 2
```

**Suggestion**: Make this configurable or use a formula that scales better (e.g., `max(32, lstm_hidden_dim // 2)`).

---

### L5. Entropy Normalization Edge Case

**File**: `action_masks.py` (lines 376-384)
**Observation**: When `num_valid == 1`, entropy is exactly 0 (correct), but the log operation for `max_entropy` happens even though result is not used.

```python
max_entropy = torch.log(num_valid.float())
normalized = raw_entropy / max_entropy.clamp(min=1e-8)
return torch.where(num_valid == 1, torch.zeros_like(normalized), normalized)
```

**Suggestion**: Minor optimization to avoid log computation:
```python
return torch.where(
    num_valid == 1,
    torch.zeros_like(raw_entropy),
    raw_entropy / torch.log(num_valid.float().clamp(min=2))
)
```

---

## Cross-File Architectural Observations

### A1. Consistent HEAD_NAMES Usage (+)

The `HEAD_NAMES` tuple from `leyline/__init__.py` is consistently used across all files for iterating over action heads. This ensures consistent ordering: `("slot", "blueprint", "blend", "op")`.

**Files**: ppo.py (lines 39, 459, 493, 525), network.py (lines 27, 289, 341), advantages.py (implicit in return dict)

---

### A2. Factored Action Space Causal Structure (+)

The causal structure documented in `advantages.py` (lines 1-24) is correctly implemented:
- `op` head always gets full advantages
- `slot` head masked on WAIT (slot irrelevant)
- `blueprint`/`blend` heads only active on GERMINATE

This is correctly mirrored in:
- `ppo.py` (lines 507-514): head_masks construction
- `ppo.py` (lines 566-578): masked mean for policy loss

---

### A3. LSTM Hidden State Contract (+)

The hidden state shape contract is consistent:
- **Network**: expects `[lstm_layers, batch, hidden_dim]` (standard PyTorch LSTM format)
- **Buffer**: stores as `[num_envs, max_steps, lstm_layers, hidden_dim]` (indexable by env)
- **Conversion**: `get_batched_sequences` correctly permutes: `[:, 0, :, :].permute(1, 0, 2)` (line 385)

---

### A4. SlotConfig Integration (+)

`SlotConfig` is consistently threaded through:
- `PPOAgent.__init__`: accepts and stores `slot_config`
- `TamiyoRolloutBuffer.__post_init__`: uses `slot_config.num_slots` for mask tensor shapes
- `FactoredRecurrentActorCritic.__init__`: uses `num_slots` for slot head output dim

**Note**: There's a subtle coupling where network's `num_slots` must match buffer's `num_slots`, validated at lines 288-291 of ppo.py.

---

### A5. Compilation Safety (+)

The codebase correctly handles `torch.compile`:
- `_validate_action_mask` decorated with `@torch.compiler.disable` (action_masks.py:285)
- `compute_advantages_and_returns` decorated with `@torch.compiler.disable` (rollout_buffer.py:261)
- `_base_network` property handles compiled vs. uncompiled access (ppo.py:341-353)

---

### A6. Potential Inconsistency: Entropy Normalization

**File**: network.py vs action_masks.py

`FactoredRecurrentActorCritic` computes `max_entropies` for per-head normalization (lines 91-100), but `MaskedCategorical.entropy()` already returns normalized entropy (lines 369-384 in action_masks.py).

```python
# network.py - computes max entropy for normalization
self.max_entropies = {
    "slot": max(math.log(num_slots), 1.0),
    ...
}

# action_masks.py - MaskedCategorical already normalizes
max_entropy = torch.log(num_valid.float())
normalized = raw_entropy / max_entropy.clamp(min=1e-8)
```

**Observation**: The `max_entropies` dict in network.py appears unused. The entropy returned from `evaluate_actions` comes from `MaskedCategorical.entropy()` which is already normalized.

**Impact**: None (unused code), but should be removed to avoid confusion.

---

## GAE Implementation Analysis

**File**: `rollout_buffer.py` (lines 261-313)

### Correctness Verification

The GAE implementation is **correct**. Key aspects verified:

1. **Per-environment isolation** (lines 272-313): GAE computed separately for each `env_id`, preventing cross-contamination between parallel environments.

2. **Truncation handling** (lines 288-302): Correctly distinguishes between:
   - True terminal (`done=True, truncated=False`): next_value=0, reset GAE
   - Truncation (`done=True, truncated=True`): use bootstrap_value, DON'T reset GAE

   ```python
   if truncated[t]:
       next_value = bootstrap_values[t]
       next_non_terminal = 1.0  # Truncation is NOT a true terminal
   else:
       next_value = 0.0
       next_non_terminal = 1.0 - float(dones[t])
   ```

3. **TD error computation** (line 308): Standard form `delta = r + gamma * V(s') * (1-done) - V(s)`

4. **GAE recursion** (line 309): Correct `lambda-return` formula: `A_t = delta_t + gamma * lambda * (1-done) * A_{t+1}`

### Minor Note

The GAE loop iterates in reverse order which is the textbook approach. For vectorized efficiency, this could use `torch.flip` but the current scalar approach is clearer and the 25-step episode length makes vectorization negligible.

---

## PPO Update Correctness Analysis

**File**: `ppo.py` (lines 417-681)

### Correctness Verification

1. **Clipped surrogate objective** (lines 571-574): Correctly implements `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`

2. **Value clipping** (lines 586-591): Implements pessimistic value clipping with separate `value_clip` parameter (not reusing policy `clip_ratio`)

3. **KL early stopping** (lines 554-561): Triggers at `1.5 * target_kl` which is the standard multiplier

4. **Entropy bonus** (lines 600-608): Correctly subtracted from loss (negative entropy = encourages exploration)

5. **Per-head gradient attribution**: The masked mean (lines 575-578) correctly handles causally-irrelevant positions without biasing the loss.

---

## Recommendations Summary

### Immediate Actions (within next sprint)

1. **H3**: Fix KL computation to exclude/downweight causally-irrelevant heads
2. **M3**: Add post-clipping gradient norm logging
3. **A6**: Remove unused `max_entropies` dict from network.py

### Near-term Improvements (next 2-4 weeks)

1. **H1**: Add runtime warning for `recurrent_n_epochs > 1`
2. **M1**: Evaluate Huber loss for value function
3. **M2**: Use `max(std, 1e-3)` for advantage normalization

### Future Considerations (backlog)

1. **H2**: Investigate per-head learning rates for gradient balancing
2. **M4**: Optimize hidden state storage for larger env counts
3. **L1**: Evaluate 2-layer feature extraction

---

## Conclusion

The simic/agent/ implementation represents solid DRL engineering with appropriate attention to:
- Numerical stability (LayerNorm, gradient clipping, masked operations)
- Algorithmic correctness (GAE, PPO clipping, entropy regularization)
- Software quality (type annotations, docstrings, consistent naming)

The factored action space with causal masking is a sophisticated design that correctly handles the hierarchical nature of seed lifecycle decisions. The integration with leyline contracts ensures type safety and consistent behavior across the codebase.

**Verdict**: Ready for production use with the recommended immediate actions applied.
