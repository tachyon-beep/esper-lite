# Deep DRL Review: vectorized.py

**File**: `/home/john/esper-lite/src/esper/simic/training/vectorized.py`
**Lines**: 2458
**Reviewer**: DRL Specialist Agent
**Date**: 2025-12-17

## Executive Summary

The `vectorized.py` file implements a sophisticated vectorized PPO training loop for the Esper project's morphogenetic neural network system. The implementation demonstrates strong engineering practices and careful attention to RL algorithm correctness. Overall, this is a well-designed PPO implementation with several notable strengths:

**Strengths:**
- Correct GAE computation with proper episode boundary handling
- Proper truncation bootstrapping for time-limited episodes
- Well-designed observation and reward normalization
- Good separation of concerns between training loop and algorithm components
- Comprehensive telemetry and anomaly detection
- Proper CUDA stream management for multi-GPU training

**Areas Requiring Attention:**
- A few algorithmic edge cases that could affect training stability
- Some minor best-practice improvements for robustness

The implementation correctly handles the most common PPO failure modes (GAE interleaving, truncation bias, reward scaling), which are the issues that typically cause training to fail silently.

---

## Critical Issues (Algorithm Correctness)

### CRITICAL-1: Observation Normalizer Update Timing is Correct but Documentation Could Mislead

**Location**: Lines 1573-1582

```python
# Accumulate raw states for deferred normalizer update
raw_states_for_normalizer_update.append(states_batch.detach())

# Normalize using FROZEN statistics during rollout collection.
# IMPORTANT: We do NOT update obs_normalizer here - statistics are updated
# AFTER the PPO update to ensure all states in a rollout batch use identical
# normalization parameters.
states_batch_normalized = obs_normalizer.normalize(states_batch)
```

**Assessment**: The implementation is **CORRECT**. The normalizer statistics are frozen during rollout collection and only updated after PPO updates (lines 290-292 in `_run_ppo_updates`). This prevents the "normalizer drift" bug where PPO ratio calculations become biased.

**However**, there is a subtle issue: the normalizer update happens in `_run_ppo_updates()` which is called once per batch of episodes. The raw states accumulated span multiple epochs within a batch. This is generally fine but worth noting that the normalizer sees batched data rather than streaming updates.

**Severity**: No action required - implementation is correct.

---

### CRITICAL-2: Truncation Bootstrapping Implementation is Correct

**Location**: Lines 1905-1975 (bootstrap value computation) and rollout_buffer.py lines 287-313

The implementation correctly handles time-limited episode truncation:

```python
# Bootstrap value for truncation: use V(s_{t+1}), not V(s_t)
# For truncated episodes (time limit), we need the value of the POST-action
# state to correctly estimate returns.
if truncated:
    # ... builds post-action state
    with torch.inference_mode():
        _, _, bootstrap_tensor, _ = agent.network.get_action(
            post_action_normalized,
            hidden=env_state.lstm_hidden,
            # ... masks
            deterministic=True,
        )
        bootstrap_value = bootstrap_tensor[0].item()
```

And in `compute_advantages_and_returns()`:

```python
if truncated[t]:
    next_value = bootstrap_values[t]
    # Truncation is NOT a true terminal - the episode was cut off
    # by time limit. We MUST use next_non_terminal=1.0 so the
    # bootstrap value contributes to delta and GAE propagates.
    next_non_terminal = 1.0
```

**Assessment**: **CORRECT**. This is the proper implementation per (Pardo et al., 2018) "Time Limits in Reinforcement Learning". Using `next_non_terminal=1.0` for truncation ensures GAE propagates through the bootstrap value.

**Severity**: No action required.

---

## High-Priority Issues (Training Stability)

### HIGH-1: Reward Normalizer Does Not Use Batch Statistics

**Location**: Lines 1898-1899

```python
raw_reward = reward
normalized_reward = reward_normalizer.update_and_normalize(reward)
```

The `RewardNormalizer` (normalization.py lines 150-211) processes rewards one-at-a-time with Welford's algorithm. In vectorized training with multiple environments, this means:

1. Rewards from env 0 are normalized with stats from all previous timesteps
2. Rewards from env 1 in the same epoch are normalized with updated stats including env 0's reward

This creates a subtle inconsistency where rewards within the same logical timestep (same epoch, different envs) use slightly different normalization.

**Impact**: Minor - the effect diminishes as count increases. With 4 envs and 25 epochs per episode, after a few batches the normalizer has seen hundreds of samples and the per-step drift is negligible.

**Recommendation**: Consider batch-updating the normalizer after all env rewards are computed for the epoch, or document that this is intentional.

**Severity**: Low - the current implementation is functionally correct.

---

### HIGH-2: KL Early Stopping Threshold Uses Fixed Multiplier

**Location**: ppo.py lines 554-561

```python
# Early stopping: if KL exceeds threshold, skip this update entirely
# 1.5x multiplier is standard (OpenAI baselines, Stable-Baselines3)
if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
    early_stopped = True
    metrics["early_stop_epoch"] = [epoch_i]
    # ...
    break  # Skip loss computation, backward, and optimizer step
```

**Assessment**: The 1.5x multiplier is standard practice. The implementation correctly computes KL **before** the optimizer step (fixing BUG-003 per the comment), which is the right design for single-epoch recurrent PPO.

**Note**: With `recurrent_n_epochs=1` (the default per line 242 in ppo.py), early stopping can only skip the entire update, not intermediate epochs. This is the intended behavior for LSTM policies where hidden state staleness is a concern.

**Severity**: No action required - implementation matches best practices.

---

### HIGH-3: Per-Head Advantage Masking is Sound

**Location**: advantages.py lines 33-70 and ppo.py lines 499-514

```python
# Compute per-head advantages with causal masking
valid_op_actions = data["op_actions"][valid_mask]
per_head_advantages = compute_per_head_advantages(
    valid_advantages, valid_op_actions
)

# Compute causal masks for masked mean computation
is_wait = valid_op_actions == LifecycleOp.WAIT
is_germinate = valid_op_actions == LifecycleOp.GERMINATE
head_masks = {
    "op": torch.ones_like(is_wait),  # op always relevant
    "slot": ~is_wait,  # slot relevant except WAIT
    "blueprint": is_germinate,  # only for GERMINATE
    "blend": is_germinate,  # only for GERMINATE
}
```

**Assessment**: **CORRECT**. The causal masking correctly zeroes out advantages for heads that had no causal effect:
- `op` head: always causally relevant (decides the action type)
- `slot` head: irrelevant for WAIT (no target slot needed)
- `blueprint`/`blend` heads: only relevant for GERMINATE (architecture selection)

The masked mean computation (lines 576-578) prevents zeros from biasing the loss:

```python
n_valid = mask.sum().clamp(min=1)  # Avoid div-by-zero
head_loss = -(clipped_surr * mask.float()).sum() / n_valid
```

**Severity**: No action required.

---

### HIGH-4: Value Clipping Uses Separate Range (Correct Design)

**Location**: ppo.py lines 583-593

```python
if self.clip_value:
    # Use separate value_clip (not policy clip_ratio) since value scale differs
    # Value predictions can range from -10 to +50, so clip_ratio=0.2 is too tight
    values_clipped = valid_old_values + torch.clamp(
        values - valid_old_values, -self.value_clip, self.value_clip
    )
    value_loss_unclipped = (values - valid_returns) ** 2
    value_loss_clipped = (values_clipped - valid_returns) ** 2
    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

**Assessment**: **CORRECT**. Using a separate `value_clip` (default 10.0 per leyline) instead of `clip_ratio` (0.2) is the right design. Policy clip ratio is designed for log-probability space; applying it to value predictions (which can span [-10, +50]) would be far too restrictive.

**Note**: Some research (Engstrom et al., 2020, "Implementation Matters in Deep RL") suggests value clipping often hurts performance. The code documents this with the comment at line 208-209 in ppo.py. Consider making `clip_value=False` the default or running ablations.

**Severity**: Low - current implementation is reasonable, but ablation recommended.

---

## Medium-Priority Issues (Best Practices)

### MEDIUM-1: LSTM Hidden State Batching Could Be Optimized

**Location**: Lines 1594-1612

```python
if env_states[0].lstm_hidden is not None:
    # Concatenate per-env hidden states along batch dimension
    h_list = [env_state.lstm_hidden[0] for env_state in env_states]
    c_list = [env_state.lstm_hidden[1] for env_state in env_states]
    # Clone pre-step hidden states for buffer storage
    pre_step_hiddens = [(h.clone(), c.clone()) for h, c in zip(h_list, c_list)]
    batched_h = torch.cat(h_list, dim=1)  # [layers, batch, hidden]
    batched_c = torch.cat(c_list, dim=1)
    batched_hidden = (batched_h, batched_c)
```

**Issue**: The cloning and concatenation happen every epoch for every env. With 4 envs and 25 epochs, this is 100 clone operations and 100 concatenations per batch.

**Recommendation**: Pre-allocate a batched hidden state tensor at the start of the episode batch and use in-place updates or views. This would reduce allocations and potentially improve throughput.

**Severity**: Medium - affects performance, not correctness.

---

### MEDIUM-2: Governor Rollback Clears Per-Env Transitions (Good Design)

**Location**: Lines 2050-2089

```python
# If any Governor rollback occurred, clear only the affected env transitions.
# This is more sample-efficient than discarding the entire batch.
rollback_env_indices = [i for i, occurred in enumerate(env_rollback_occurred) if occurred]

if rollback_env_indices:
    # ... tracking code ...
    for env_idx in rollback_env_indices:
        agent.buffer.clear_env(env_idx)
```

**Assessment**: **EXCELLENT DESIGN**. Per-environment rollback clearing is more sample-efficient than batch-level clearing. The rollout buffer's `clear_env()` method correctly zeros LSTM hidden states to prevent stale state leakage.

**Severity**: No action required - this is a strength.

---

### MEDIUM-3: Entropy Coefficient Per-Head Weighting

**Location**: ppo.py lines 596-604

```python
# Entropy loss with per-head weighting.
# NOTE: Entropy floors were removed because torch.clamp to a constant
# provides zero gradient...
entropy_loss = 0.0
for key, ent in entropy.items():
    head_coef = self.entropy_coef_per_head.get(key, 1.0)
    entropy_loss = entropy_loss - head_coef * ent.mean()
```

**Issue**: The default per-head coefficients are all 1.0 (line 252-257 in ppo.py):

```python
self.entropy_coef_per_head = entropy_coef_per_head or {
    "slot": 1.0,
    "blueprint": 1.0,
    "blend": 1.0,
    "op": 1.0,
}
```

**Concern**: The `blueprint` and `blend` heads are only active during GERMINATE actions, which may be infrequent (especially if the policy learns to WAIT often). This means these heads receive fewer gradient updates from the entropy term.

**Recommendation**: Consider increasing `blueprint` and `blend` coefficients (e.g., 1.5-2.0) to compensate for lower update frequency, or track per-head entropy separately to verify sufficient exploration.

**Severity**: Medium - may affect exploration of germination options.

---

### MEDIUM-4: Gradient EMA Tracker Drift Detection

**Location**: Lines 2134-2147

```python
# Gradient drift detection (P4-9) - catches slow degradation
if grad_ema_tracker is not None and ppo_grad_norm is not None:
    # Simple gradient health: 1.0 if norm in [0.01, 100], scales down outside
    grad_health = 1.0 if 0.01 <= ppo_grad_norm <= 100.0 else max(0.0, 1.0 - abs(ppo_grad_norm - 50) / 100)
    has_drift, drift_metrics = grad_ema_tracker.check_drift(ppo_grad_norm, grad_health)
```

**Issue**: The `grad_health` formula has a discontinuity at norm=50 and the scaling is asymmetric. For norm < 0.01 or norm > 100, the formula `1.0 - abs(ppo_grad_norm - 50) / 100` produces:
- norm=0.01: health = 1.0 - 49.99/100 = 0.5
- norm=100: health = 1.0 - 50/100 = 0.5
- norm=0: health = 1.0 - 50/100 = 0.5
- norm=200: health = 1.0 - 150/100 = -0.5, clamped to 0.0

**Assessment**: The formula is reasonable but the transition is abrupt. A smoother sigmoid-based health score would be more stable.

**Severity**: Low - only affects anomaly detection telemetry.

---

## Low-Priority Suggestions

### LOW-1: Consider torch.inference_mode() for Bootstrap Value

**Location**: Lines 1963-1973

The bootstrap value computation correctly uses `torch.inference_mode()`:

```python
with torch.inference_mode():
    _, _, bootstrap_tensor, _ = agent.network.get_action(
        post_action_normalized,
        hidden=env_state.lstm_hidden,
        # ...
        deterministic=True,
    )
```

**Assessment**: Correct - no gradients needed for bootstrap value estimation.

---

### LOW-2: Magic Numbers in Telemetry Thresholds

**Location**: Lines 857-858 (Governor defaults), Line 2137 (gradient health)

Several threshold values are inline magic numbers:
- `sensitivity=6.0` for Governor
- `[0.01, 100]` for gradient health range
- `50` for gradient health center

**Recommendation**: Move these to leyline constants for consistency with the project's design philosophy.

**Severity**: Low - cosmetic/maintainability issue.

---

### LOW-3: SharedBatchIterator Synchronization is Correct

**Location**: Lines 1185-1189

```python
# CRITICAL: SharedBatchIterator does non_blocking transfers on the default stream.
# We must sync env_state.stream with default stream before using the data,
# otherwise we may access partially-transferred data (race condition).
if env_state.stream:
    env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))
```

**Assessment**: **CORRECT**. This is the proper pattern for multi-stream CUDA programming. The `wait_stream` call creates a synchronization barrier between the data transfer (default stream) and the compute (env stream).

---

### LOW-4: Counterfactual Computation in Same Loop as Validation (Efficient)

**Location**: Lines 1313-1325

```python
# COUNTERFACTUAL (alpha=0) - SAME BATCH, no DataLoader reload!
# Data is already on GPU from the main validation pass.
if i in slots_needing_counterfactual:
    for slot_id in slots_needing_counterfactual[i]:
        with stream_ctx:
            with env_state.model.seed_slots[slot_id].force_alpha(0.0):
                _, cf_correct_tensor, cf_total = process_val_batch(...)
            env_state.cf_correct_accums[slot_id].add_(cf_correct_tensor)
```

**Assessment**: **EXCELLENT DESIGN**. Running counterfactual validation in the same loop as main validation:
1. Avoids a second DataLoader iteration (O(N) saved)
2. Keeps data warm in GPU cache
3. Maintains stream isolation per environment

This is a sophisticated optimization that many implementations miss.

---

## Summary of Findings

| Priority | Count | Action Required |
|----------|-------|-----------------|
| Critical | 0 | None - implementation is correct |
| High | 0 | None - all assessed as correct |
| Medium | 4 | Consider optimizations |
| Low | 4 | Cosmetic/optional |

## Recommendations

1. **No Critical Changes Required**: The PPO implementation is algorithmically correct.

2. **Consider Performance Optimization**: The LSTM hidden state batching could be optimized with pre-allocated tensors.

3. **Consider Ablation Study**: Test `clip_value=False` to see if value clipping helps or hurts on this task.

4. **Consider Per-Head Entropy Tuning**: Blueprint/blend heads may benefit from higher entropy coefficients due to lower update frequency.

5. **Document Design Decisions**: The codebase has excellent inline comments explaining rationale. Consider a dedicated doc for the reward normalization strategy.

## Conclusion

This is a high-quality PPO implementation that correctly handles the most common failure modes:
- GAE is computed per-environment with proper episode boundaries
- Truncation bootstrapping uses V(s_{t+1}) not V(s_t)
- Observation normalization is frozen during rollout
- Reward normalization stabilizes critic learning
- Action masking correctly handles factored action spaces
- CUDA stream management prevents race conditions

The code reflects deep understanding of both PPO theory and PyTorch engineering best practices. The main opportunities are in performance optimization, not algorithm correctness.
