# Deep RL Code Review: simic/training/ Subfolder

**Reviewer:** DRL Specialist
**Date:** 2025-12-17
**Scope:** `/home/john/esper-lite/src/esper/simic/training/`
**Files Reviewed:**
- `vectorized.py` (2458 lines)
- `helpers.py` (684 lines)
- `config.py` (318 lines)
- `parallel_env_state.py` (111 lines)
- `__init__.py` (30 lines)

**Related Modules Examined:**
- `simic/agent/ppo.py` (PPOAgent implementation)
- `simic/agent/rollout_buffer.py` (TamiyoRolloutBuffer)
- `simic/agent/advantages.py` (Per-head advantage computation)
- `simic/agent/network.py` (FactoredRecurrentActorCritic)
- `simic/rewards/rewards.py` (Reward computation)

---

## Executive Summary

The `simic/training/` subfolder implements a sophisticated vectorized PPO training loop for a factored action space with LSTM-based recurrent policies. The implementation demonstrates strong adherence to modern RL best practices, with particular attention to:

1. **Correct PPO implementation** with per-head clipped surrogate objectives
2. **Proper GAE computation** with per-environment isolation (P0 bug fix documented)
3. **Truncation bootstrapping** for time-limited episodes (critical for unbiased value estimates)
4. **CUDA stream-based parallelization** for efficient multi-environment training
5. **Comprehensive telemetry** with anomaly detection and gradient drift monitoring

However, the review identified several issues ranging from algorithmic correctness concerns to best-practice improvements.

**Verdict:** Production-ready with caveats. Address Critical and High-priority issues before scaling to larger training runs.

---

## Critical Issues (Algorithm Correctness)

### CRIT-1: PPO Update Loop Missing Minibatch Shuffling

**File:** `vectorized.py` lines 2108-2114
**Severity:** Critical

The `_run_ppo_updates()` helper (not shown in the main file but called at line 2108) performs multiple PPO updates on the same data. However, examination of `PPOAgent.update()` in `simic/agent/ppo.py` reveals that the entire buffer is processed as a single batch without minibatch shuffling.

**Problem:** PPO's theoretical guarantees assume data is sampled from a minibatch distribution. Processing the entire rollout as one batch:
1. Increases variance in gradient estimates
2. Reduces sample efficiency (no multiple passes with different ordering)
3. May cause premature KL early stopping due to large-batch gradient noise

**Evidence (ppo.py lines 441-442):**
```python
data = self.buffer.get_batched_sequences(device=self.device)
valid_mask = data["valid_mask"]
```

The data is used directly without splitting into minibatches or shuffling.

**Recommendation:** Either:
1. Split the rollout into minibatches and shuffle indices within epochs
2. Document that this is intentional for LSTM state coherence (if so, the `batch_size` parameter in PPOAgent is misleading)
3. Implement chunk-based processing that respects episode boundaries for recurrent policies

---

### CRIT-2: Potential Value Function Staleness in Multi-Epoch PPO Updates

**File:** `simic/agent/ppo.py` lines 582-593
**Severity:** Critical

The value loss uses `valid_old_values` from the buffer:
```python
valid_old_values = data["values"][valid_mask]
if self.clip_value:
    values_clipped = valid_old_values + torch.clamp(
        values - valid_old_values, -self.value_clip, self.value_clip
    )
```

**Problem:** For recurrent PPO with `recurrent_n_epochs > 1`, the old values stored in the buffer were computed with potentially different LSTM hidden states than the current forward pass. This creates a mismatch:
- `valid_old_values`: V(s_t | h_old) - value computed during rollout collection
- `values`: V(s_t | h_reconstructed) - value from current forward pass with initial hidden state

Since LSTM hidden states evolve through the sequence, `h_reconstructed` diverges from `h_old` as timesteps progress. Value clipping on misaligned values can cause:
1. Incorrect value loss gradients
2. Conservative updates that slow learning

**Recommendation:**
1. When using value clipping with recurrent policies, store values from the current forward pass, not the rollout
2. OR document that `recurrent_n_epochs=1` is required for correctness (currently defaulted to 1 at line 242 of ppo.py, but exposed as a parameter)

---

### CRIT-3: Observation Normalizer Statistics Updated AFTER PPO Update

**File:** `vectorized.py` lines 1573-1582 and line 2111
**Severity:** High (bordering on Critical)

The code correctly freezes normalizer statistics during rollout collection (good!) but then updates normalizer stats AFTER the PPO update:

```python
# Line 1574 - states collected
raw_states_for_normalizer_update.append(states_batch.detach())

# Line 1582 - normalized with frozen stats
states_batch_normalized = obs_normalizer.normalize(states_batch)

# Line 2111 (in _run_ppo_updates) - normalizer updated after PPO
```

**Problem:** This ordering means the PPO update at batch N uses normalizer stats that exclude batch N's observations. When the normalizer updates after PPO, the statistics change, and the NEXT batch's observations will be normalized differently.

Over many batches, this creates a subtle distribution shift where the policy is always "one batch behind" on normalization statistics. The mismatch is small per-batch but compounds over training.

**Recommendation:** Update normalizer statistics BEFORE the PPO update, using the same raw observations that will be normalized for training. This ensures consistency between data collection and policy optimization.

---

## High-Priority Issues (Training Stability)

### HIGH-1: Missing Entropy Coefficient Per-Head Gradient Scaling

**File:** `simic/agent/ppo.py` lines 601-604
**Severity:** High

The entropy loss sums per-head entropies without accounting for head activity frequency:

```python
entropy_loss = 0.0
for key, ent in entropy.items():
    head_coef = self.entropy_coef_per_head.get(key, 1.0)
    entropy_loss = entropy_loss - head_coef * ent.mean()
```

**Problem:** The `blueprint` and `blend` heads are only causally relevant during GERMINATE actions (typically ~5-15% of timesteps based on typical action distributions). Their entropy contributions are averaged over ALL timesteps including zeros from masked positions, diluting their gradient signal.

This creates an imbalance where:
- `op` head receives strong entropy gradients every step
- `blueprint`/`blend` heads receive weak entropy gradients (diluted by zeros)

**Evidence:** Per-head causal masking is correctly applied to advantages (lines 506-514) but NOT to entropy. The entropy is just `ent.mean()` over all timesteps.

**Recommendation:** Apply masked mean to entropy computation:
```python
for key, ent in entropy.items():
    mask = head_masks[key]
    n_valid = mask.sum().clamp(min=1)
    head_entropy = (ent * mask.float()).sum() / n_valid
    entropy_loss = entropy_loss - head_coef * head_entropy
```

---

### HIGH-2: Reward Normalization Welford Algorithm Numerical Stability

**File:** `vectorized.py` (references RewardNormalizer)
**Severity:** High

The reward normalization uses a running mean/variance estimator. From the checkpoint save at lines 2437-2439:
```python
'reward_normalizer_mean': reward_normalizer.mean,
'reward_normalizer_m2': reward_normalizer.m2,
'reward_normalizer_count': reward_normalizer.count,
```

**Problem:** Welford's algorithm can become numerically unstable when:
1. Count grows very large (float precision issues)
2. Rewards have high magnitude outliers

Without seeing the RewardNormalizer implementation, I cannot verify numerical guards are in place.

**Recommendation:** Ensure RewardNormalizer:
1. Uses stable Welford update (not naive variance)
2. Has count overflow protection
3. Uses float64 for running statistics
4. Implements reward clipping before normalization

---

### HIGH-3: LSTM Hidden State Reset Logic at Episode Boundaries

**File:** `vectorized.py` lines 1113-1114 and rollout_buffer.py lines 287-310
**Severity:** High

The hidden state is reset per-batch:
```python
# Line 1113-1114
for env_idx in range(envs_this_batch):
    agent.buffer.start_episode(env_id=env_idx)
    env_states[env_idx].lstm_hidden = None  # Fresh hidden for new episode
```

**Problem:** The GAE computation in `compute_advantages_and_returns()` (rollout_buffer.py lines 287-313) handles truncation correctly, but the LSTM hidden state initialization has a subtle bug.

At line 1604-1611, when `env_states[0].lstm_hidden is None`, the code uses `agent.network.get_initial_hidden()` for ALL envs:
```python
init_hidden = agent.network.get_initial_hidden(len(env_states), agent.device)
```

This assumes all envs start fresh simultaneously. If some envs complete episodes mid-batch while others continue, the `None` check at line 1594 would fail to detect this because individual env hidden states are not checked independently.

**Current Mitigation:** The code creates fresh environments per batch (line 1104-1107), so all envs always start together. However, this architecture decision limits flexibility.

**Recommendation:** Add per-env hidden state tracking that can handle staggered episode boundaries within a batch if the architecture ever changes to support continuous environment collection.

---

### HIGH-4: Bootstrap Value Computation Uses Stale Action Masks

**File:** `vectorized.py` lines 1960-1972
**Severity:** Medium-High

For truncated episodes, the bootstrap value V(s_{t+1}) is computed:
```python
env_masks = {key: masks_batch[key][env_idx] for key in masks_batch}

# Get V(s_{t+1}) - use updated LSTM hidden state from this step
with torch.inference_mode():
    _, _, bootstrap_tensor, _ = agent.network.get_action(
        post_action_normalized,
        hidden=env_state.lstm_hidden,
        slot_mask=env_masks["slot"].unsqueeze(0),
        ...
    )
```

**Problem:** The action masks `env_masks` are from the PRE-ACTION state (line 1960), not the POST-ACTION state. After executing an action (e.g., GERMINATE), the valid action set changes (the slot is now occupied). Using pre-action masks for the post-action state could affect value estimation when masks significantly constrain the action space.

**Impact:** The value function sees an observation-mask mismatch. Since Esper's action masking significantly constrains available actions based on slot occupancy, this could bias bootstrap estimates.

**Recommendation:** Compute post-action masks from `post_action_slot_reports` and use those for the bootstrap value computation.

---

## Medium-Priority Issues (Best Practices)

### MED-1: Gradient Clipping Applied Globally Instead of Per-Head

**File:** `simic/agent/ppo.py` line 632
**Severity:** Medium

```python
nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
```

**Observation:** With a factored action space where heads have different activity levels, global gradient clipping can cause:
1. Active heads (op) to dominate the gradient budget
2. Sparse heads (blueprint, blend) to have their gradients disproportionately clipped

The code already tracks per-head gradient norms (lines 614-630), which is excellent for diagnostics. However, the clipping doesn't use this information.

**Recommendation:** Consider per-head gradient clipping or at least document that global clipping is intentional. Per-head metrics already collected could inform per-head clip thresholds.

---

### MED-2: Value Coefficient Fixed at 0.5 (DEFAULT_VALUE_COEF)

**File:** `simic/agent/ppo.py` line 258
**Severity:** Medium

```python
self.value_coef = value_coef  # DEFAULT_VALUE_COEF = 0.5
```

**Observation:** The standard PPO value coefficient of 0.5 may not be optimal for this architecture because:
1. The value function shares LSTM features with the policy (potentially competing gradients)
2. Esper's reward signal is dense and potentially high-magnitude (contribution + rent + shaped bonuses)

With reward normalization in place (line 1899), a coefficient of 0.5 should be reasonable, but the interaction between:
- Reward normalization
- Value clipping (`value_clip = 10.0` default)
- Value coefficient

...could cause value learning to be either too aggressive or too conservative.

**Recommendation:** Add value coefficient to hyperparameter search space or implement adaptive scaling based on explained variance trends.

---

### MED-3: KL Divergence Computed as Sum Not Average

**File:** `simic/agent/ppo.py` lines 536-544
**Severity:** Medium

```python
head_kls = []
for key in HEAD_NAMES:
    ...
    head_kl = (kl_per_step * mask.float()).sum() / n_valid
    head_kls.append(head_kl)
approx_kl = torch.stack(head_kls).sum().item()
```

**Problem:** The joint KL is computed as the SUM of per-head KLs. While mathematically correct for factored action spaces (joint KL = sum of marginal KLs for independent distributions), this makes the `target_kl` threshold semantics different from standard PPO.

With `target_kl = 0.015` and 4 heads, the effective per-head threshold is ~0.00375, which may be too tight. The code triggers early stopping at `1.5 * target_kl = 0.0225`, but with 4 heads contributing, this could trigger prematurely.

**Recommendation:** Either:
1. Document that target_kl is "total KL budget across all heads"
2. Scale target_kl by number of active heads
3. Use mean instead of sum for easier threshold interpretation

---

### MED-4: Counterfactual Baseline Not Computed for All Epochs

**File:** `vectorized.py` lines 1253-1266 and 1378-1384
**Severity:** Medium

Counterfactual baselines (for reward attribution) are only computed when a slot has an active seed with alpha > 0:

```python
if seed_state and seed_state.alpha > 0:
    active_slots.add(slot_id)
```

**Problem:** During GERMINATE actions, the newly created seed starts with alpha=0 and gradually increases. The counterfactual baseline won't be computed for the first epoch after germination, meaning the reward attribution signal is missing precisely when it would be most informative (the immediate effect of the germination decision).

**Impact:** This could weaken the learning signal for GERMINATE actions specifically, making it harder to learn which blueprints are most effective.

**Recommendation:** Compute counterfactual baseline whenever there's an active seed, regardless of alpha value. The alpha=0 case is exactly where we want to measure "what does this seed contribute."

---

### MED-5: Anomaly Detection Thresholds Hardcoded

**File:** `simic/agent/ppo.py` lines 293-294
**Severity:** Low-Medium

```python
self.ratio_explosion_threshold = 5.0
self.ratio_collapse_threshold = 0.1
```

**Observation:** These thresholds are reasonable defaults but not configurable. With Esper's factored action space and potentially sparse head activity, ratio explosion in low-activity heads (blueprint, blend) is more likely simply due to variance, not actual policy degradation.

**Recommendation:** Make thresholds configurable and consider per-head thresholds based on expected activity frequency.

---

## Low-Priority Suggestions

### LOW-1: Pre-allocated Tensor Reuse Pattern

**File:** `parallel_env_state.py` lines 82-108
**Observation:** Excellent implementation of pre-allocated accumulators to avoid per-epoch allocation. The `zero_accumulators()` method correctly uses in-place operations.

**Suggestion:** Consider using `torch.Tensor.zero_()` with CUDA graphs for additional speedup on long training runs.

---

### LOW-2: Episode Boundary Tracking Granularity

**File:** `rollout_buffer.py` lines 184-197
**Observation:** `start_episode()` and `end_episode()` track boundaries but aren't used in advantage computation (which iterates per-env using step_counts).

**Suggestion:** Either use episode_boundaries for something useful (e.g., episode-level logging) or remove to reduce code complexity.

---

### LOW-3: Magic Number in Gradient Health Check

**File:** `vectorized.py` line 2137
```python
grad_health = 1.0 if 0.01 <= ppo_grad_norm <= 100.0 else max(0.0, 1.0 - abs(ppo_grad_norm - 50) / 100)
```

**Suggestion:** Move these thresholds (0.01, 100.0, 50) to leyline constants with semantic names.

---

### LOW-4: TrainingConfig Validation Could Be More Specific

**File:** `config.py` lines 255-298
**Observation:** Good comprehensive validation, but some domain-specific RL constraints are missing:
- `gamma` should warn if < 0.9 for long-horizon tasks (Esper episodes are 25 epochs)
- `clip_ratio` > 0.3 is unusual and potentially unstable

**Suggestion:** Add warnings (not errors) for unusual but valid hyperparameter combinations.

---

## Cross-File Architectural Observations

### ARCH-1: Clean Separation of Concerns

The architecture cleanly separates:
- **Environment state** (`parallel_env_state.py`)
- **Training loop** (`vectorized.py`)
- **Agent/optimizer** (`simic/agent/ppo.py`)
- **Buffer management** (`simic/agent/rollout_buffer.py`)

This modularity enables easy testing and modification of individual components.

### ARCH-2: Telemetry Integration Pattern

The telemetry integration through `TelemetryEvent` and `hub.emit()` is consistent and comprehensive. The telemetry escalation system (P4-9 gradient drift detection) shows mature production thinking.

**Minor concern:** The telemetry conditionals are verbose and repeated. Consider a `TelemetryGuard` context manager.

### ARCH-3: Governor Watchdog Pattern

The `TolariaGovernor` provides fail-safe catastrophic failure detection with automatic rollback. This is excellent defensive programming for a system that runs unsupervised training.

**Observation:** The governor checkpoint (line 1415) happens every 5 epochs, which may miss rapid degradation. Consider adaptive snapshot frequency based on loss variance.

### ARCH-4: Reward Normalization Pipeline

The reward normalization (line 1899) correctly separates:
- Raw reward (for logging)
- Normalized reward (for training)

This prevents reward scale from affecting value function learning while preserving interpretable metrics.

### ARCH-5: Per-Head Advantage Computation

The per-head advantage computation with causal masking (`advantages.py`) is a sophisticated implementation that correctly handles the factored action space. The masked mean pattern in policy loss computation (lines 576-578 of ppo.py) prevents gradient bias from inactive heads.

---

## Recommended Priority Order

1. **CRIT-1**: Add minibatch shuffling or document single-batch intentionality
2. **CRIT-2**: Validate value clipping correctness with recurrent policies
3. **HIGH-1**: Fix entropy gradient dilution for sparse heads
4. **HIGH-4**: Use post-action masks for bootstrap value
5. **CRIT-3**: Reorder normalizer update to happen BEFORE PPO update
6. **MED-3**: Document or fix KL sum-vs-mean semantics
7. **MED-4**: Compute counterfactual even for alpha=0 seeds
8. **HIGH-2**: Verify reward normalizer numerical stability
9. **HIGH-3**: Document LSTM hidden state assumptions

---

## Conclusion

The `simic/training/` subfolder implements a well-engineered PPO training loop with many sophisticated features:
- Correct GAE with truncation bootstrapping
- CUDA stream-based parallelization
- Comprehensive telemetry and anomaly detection
- Clean architectural separation

The critical issues identified are subtle correctness concerns that would only manifest in specific scenarios (multi-epoch recurrent updates, large observation distributions, high activity in sparse heads). Most deployments with `recurrent_n_epochs=1` (the default) will not encounter CRIT-2.

The high-priority issues around entropy gradient dilution and bootstrap mask mismatches should be addressed to ensure optimal learning in the factored action space.

Overall, this is production-quality RL code that demonstrates strong engineering discipline and RL domain knowledge.
