# Batch 8 Deep Dive: Simic Training - Main Training Loop

**Reviewer Specialization:** Deep Reinforcement Learning
**Date:** 2025-12-27
**Files Reviewed:**
1. `/home/john/esper-lite/src/esper/simic/training/config.py`
2. `/home/john/esper-lite/src/esper/simic/training/dual_ab.py`
3. `/home/john/esper-lite/src/esper/simic/training/helpers.py`
4. `/home/john/esper-lite/src/esper/simic/training/__init__.py`
5. `/home/john/esper-lite/src/esper/simic/training/parallel_env_state.py`
6. `/home/john/esper-lite/src/esper/simic/training/policy_group.py`
7. `/home/john/esper-lite/src/esper/simic/training/vectorized.py`

---

## Executive Summary

This batch contains the heart of the Esper RL training system - the vectorized PPO training loop that orchestrates multiple parallel environments, action masking, reward computation, and policy updates. The code is well-structured with clear separation of concerns, proper handling of CUDA streams for async execution, and careful attention to RL-specific requirements like bootstrap value computation and LSTM hidden state management.

Key strengths:
- Inverted control flow (batch-first iteration) for GPU utilization
- Proper bootstrap value computation for truncated episodes (GAE-correct)
- CUDA stream isolation with per-environment scalers for AMP safety
- Comprehensive action masking with physical constraint enforcement
- Reward normalization with running statistics

Key concerns identified:
- Hidden state management during episode boundaries may cause credit leakage
- Observation normalizer update timing creates one-batch lag during rollout
- Potential for stale optimizer state after dynamic seed lifecycle changes
- Some telemetry-driven code paths have performance implications in hot loops

---

## File-by-File Analysis

### 1. `config.py` - TrainingConfig

**Purpose:** Strict, JSON-loadable hyperparameter configuration for PPO training. Single source of truth for all training parameters.

**DRL Analysis:**

The configuration properly centralizes critical PPO hyperparameters with sensible defaults from leyline. The validation is thorough.

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-01 | P3 | `chunk_length` is forced to equal `max_epochs` (line 377-379), but this conflates episode length with BPTT window size. For very long episodes, you may want shorter chunks for gradient stability. The current design prevents this flexibility. |
| C8-02 | P4 | The `to_ppo_kwargs()` method computes `entropy_steps` via division (line 283-284), but this is also computed in `train_ppo_vectorized`. Consider consolidating to avoid drift. |
| C8-03 | P3 | `reward_mode_per_env` validation (line 433-445) correctly checks length and type, but doesn't validate that all modes in the tuple are valid for the chosen `reward_family`. If `reward_family == LOSS` but `reward_mode_per_env` contains SPARSE, the validation will pass but runtime will fail. |

**Positive Notes:**
- Good use of `slots=True` for dataclass memory efficiency
- Proper enum coercion in `from_dict()` for JSON deserialization
- Task-specific presets (`for_cifar10_stable()`) encode domain knowledge appropriately

---

### 2. `dual_ab.py` - Dual-Policy A/B Testing

**Purpose:** Train separate policies on separate GPUs for true A/B comparison of reward modes.

**DRL Analysis:**

The design is sound for policy isolation - each group gets its own agent, seed, and device. However, the sequential training (noted in limitations comment) introduces confounds.

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-04 | P2 | Sequential training means later groups benefit from CUDA warmup and torch.compile caching. For fair comparison, groups should train in parallel or alternating lockstep. The comment acknowledges this but the impact on results is not trivial - compilation can take minutes and affects early-episode performance. |
| C8-05 | P3 | The deterministic seed offset (line 191-192) uses MD5 hash of group_id, but this creates reproducibility fragility - changing group_id from "A" to "alpha" would change all seeds. Consider using a fixed offset table. |
| C8-06 | P4 | `_print_dual_ab_comparison()` (line 232-294) uses simple "final accuracy" as the winner metric, but doesn't report statistical significance (e.g., Mann-Whitney U test on episode rewards). For proper A/B testing, you need confidence intervals. |

---

### 3. `helpers.py` - Training Loop Helpers

**Purpose:** Compiled training step, one-epoch training, heuristic policy training.

**DRL Analysis:**

Contains the core forward/backward step logic and a complete heuristic training path for baseline comparison.

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-07 | P2 | `_train_one_epoch()` (line 213-304) doesn't handle the case where `seed_optimizer` exists but the seed has been pruned mid-epoch. If a seed is pruned by another subsystem during an epoch, the seed_optimizer will step with stale/empty param groups. However, reviewing the callsites, this function is only used by heuristic training which doesn't have concurrent pruning. |
| C8-08 | P3 | `_convert_flat_to_factored()` (line 311-381) hardcodes `slot_idx=0` for all actions. This breaks multi-slot support in the heuristic path - heuristic policy decisions will always target slot 0 regardless of which slot the seed is actually in. |
| C8-09 | P4 | `run_heuristic_episode()` (line 384-721) has extensive inline code that duplicates much of `train_ppo_vectorized()`. Consider extracting shared logic to reduce maintenance burden. |
| C8-10 | P3 | In `run_heuristic_episode()`, line 592-594, `available_slots` counts slots where `state is None`. However, the subsequent `signals.update()` uses this count, but the heuristic policy may not use it consistently. This could lead to the heuristic making decisions based on stale slot availability. |

**Positive Notes:**
- `compute_rent_and_shock_inputs()` correctly handles the BLEND_OUT freeze (requires_grad toggle) by using cached param counts
- Good use of `@functools.cache` for lazy compilation initialization in `_get_compiled_train_step()`

---

### 4. `__init__.py` - Module Exports

**Purpose:** Clean re-exports of public API.

**Analysis:** No issues. Proper `__all__` declaration.

---

### 5. `parallel_env_state.py` - ParallelEnvState

**Purpose:** Dataclass holding all state for a single parallel training environment.

**DRL Analysis:**

This is a critical class that encapsulates per-environment state including optimizers, LSTM hidden states, and pre-allocated accumulators.

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-11 | P1 | `lstm_hidden` (line 79) is stored per-environment but `reset_episode_state()` (line 167-197) does NOT reset it to None. This means LSTM hidden state from the previous episode can leak into the next episode, causing temporal credit assignment corruption. The hidden state should be reset at episode boundaries for episodic tasks. |
| C8-12 | P3 | `scaffold_boost_ledger` uses `defaultdict(list)` (line 94-96), which auto-creates keys on access. This can cause subtle bugs if you check `slot_id in scaffold_boost_ledger` vs checking if the list is non-empty. |
| C8-13 | P4 | `__post_init__()` initializes counters with LifecycleOp names (line 104-109), but doesn't include all potential action names. If the action enum changes, this will silently fail. Consider using a shared constant from leyline. |
| C8-14 | P2 | `counterfactual_helper._last_matrix = None` (line 192) in `reset_episode_state()` reaches into private state of CounterfactualHelper. This should be a public method on the helper. |

**Positive Notes:**
- Pre-allocated accumulators (`train_loss_accum`, etc.) avoid per-epoch allocation churn
- `autocast_enabled` pre-computation saves repeated device type checks in hot path
- Good use of `slots=True` for memory efficiency

---

### 6. `policy_group.py` - PolicyGroup

**Purpose:** Abstraction for A/B testing with independent policy per GPU.

**DRL Analysis:**

Currently a thin wrapper that's partially implemented (envs field marked as "for future parallel implementation").

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-15 | P4 | The `envs` field is documented as "for future parallel implementation" but the default factory creates an empty list. This could cause confusion - the field exists but is never populated. Consider marking it more explicitly or using a sentinel. |

---

### 7. `vectorized.py` - Main Training Loop

**Purpose:** High-performance vectorized PPO training with CUDA streams, inverted control flow, and comprehensive telemetry.

**This is the heart of the system.** My analysis will be detailed.

#### Episode and Rollout Handling

**Bootstrap Value Computation (Lines 2969-3000, 3101-3130):**

The code correctly:
1. Collects post-action state AFTER mechanical lifecycle advance (line 2969 comment: "Fix BUG-022")
2. Computes bootstrap values in a single batched forward pass (line 3103-3130)
3. Uses `truncated` flag to determine when bootstrap is needed (line 3136-3139)

This is GAE-correct implementation. When episodes are truncated (not terminated), the bootstrap value provides an estimate of remaining return.

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-16 | P1 | **Hidden State Management During Episode Reset (Lines 2939-2954):** When `done=True`, the code calls `agent.buffer.end_episode()` and resets the LSTM hidden state for that environment. However, this happens AFTER storing the transition with the OLD hidden state. This is correct for training (we want the hidden state that was used for action selection). BUT the hidden state reset creates new tensors via `.clone()` (lines 2950-2954) which may cause issues if the batch isn't the last one. More critically, the newly initialized hidden state is used for the NEXT episode, but if there are remaining epochs in the current batch after `done`, those will incorrectly use the fresh hidden state. Reviewing the loop structure: this only happens when `epoch == max_epochs`, which IS the last epoch, so this is actually fine. However, the condition `if done:` (line 2939) is redundant with the `epoch == max_epochs` check (line 3003). |
| C8-17 | P2 | **Observation Normalizer Timing (Lines 2436-2440, 3304-3314):** The observation normalizer is updated in `_run_ppo_updates()` with raw states collected during the batch. However, during rollout collection, `obs_normalizer.normalize()` (line 2440) uses FROZEN statistics. This means the very first batch has non-normalized (or poorly-normalized) observations. The comment in `_run_ppo_updates()` (lines 3305-3310) acknowledges this "one-batch lag" and argues it's intentional. From a DRL perspective, this is acceptable but suboptimal - the policy may learn slightly different representations early in training. |
| C8-18 | P2 | **Reward Normalizer Update Position (Line 2656):** `reward_normalizer.update_and_normalize(reward)` updates statistics AND normalizes in one call. This is correct, but the reward components (`bounded_attribution`, `compute_rent`, etc.) at lines 2662-2668 are accumulated BEFORE normalization. This means the `reward_summary_accum` contains raw rewards while `normalized_reward` (what goes to the buffer) is normalized. When these are later reported in telemetry, the ratio between raw and normalized may be confusing. |
| C8-19 | P1 | **Seed Optimizer Lifecycle (Lines 1411-1422, 2693, 2776, etc.):** When a seed is germinated, pruned, or fossilized, the code calls `env_state.seed_optimizers.pop(slot_id, None)`. However, if an action FAILS (e.g., `action_success = False` at line 2696), the optimizer is NOT popped but the seed state may have changed. More importantly, the optimizer is only created lazily in `process_train_batch()` when the slot has an active seed. If a seed is germinated in epoch N but pruned in epoch N+1 before any training batch, the optimizer will never have been created, so the pop is a no-op. This is fine, but the flow is fragile and depends on training batches happening after action execution. |
| C8-20 | P3 | **Hindsight Credit Cap (Lines 2734, line 2736):** `total_credit = min(total_credit, MAX_HINDSIGHT_CREDIT)` caps the credit, but this happens BEFORE adding to `pending_hindsight_credit`. If hindsight credit accumulates across multiple fossilizations in the same episode, the total could exceed the cap. Consider capping at the point of application (line 2649). |

#### Action Execution and Masking

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-21 | P3 | **Action Validity vs Action Success Mismatch (Lines 2524-2817):** `_parse_sampled_action()` returns `action_valid_for_reward` and `action_for_reward`. The code then checks physical preconditions AGAIN during action execution (e.g., line 2676: `model.seed_slots[target_slot].state is None`). If the model state changed between action selection and execution (unlikely but possible with concurrent modifications), the validity check and execution check could disagree. This is defensive but creates code duplication. |
| C8-22 | P4 | **OP_WAIT Always Succeeds (Line 2808-2809):** `elif op_idx == OP_WAIT: action_success = True` - WAIT always succeeds, but the code still adds it to `successful_action_counts`. This inflates the success rate metric if the policy learns to spam WAIT. |
| C8-23 | P3 | **PRUNE Age Check Duplication (Lines 2756-2757):** The MIN_PRUNE_AGE check happens BOTH in `_parse_sampled_action()` (line 1302-1304) AND in the execution block (lines 2756-2757). This is redundant and creates a maintenance burden if the threshold changes. |

#### CUDA Stream and AMP Handling

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-24 | P2 | **GradScaler Per-Environment (Lines 1132-1136, 1451-1517):** The comment at lines 1495-1508 explains the GradScaler stream safety. However, if multiple environments share the same physical GPU (e.g., `env_device_map = ["cuda:0", "cuda:0", ...]`), they share CUDA memory but have separate GradScalers with independent `_scale` values. Over time, these can diverge, leading to different effective learning rates for seeds on the same GPU. This is a subtle issue but could cause A/B test confounds if not controlled. |
| C8-25 | P4 | **record_stream() on Cloned Tensors (Lines 1392-1394, 1556-1560):** The comments note that iterators now return clones, but `record_stream()` is still called "to prevent premature deallocation". Since the clone owns its own storage, `record_stream()` is technically unnecessary (but harmless). Consider removing to reduce CUDA API overhead. |

#### Validation and Counterfactual Phase

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-26 | P2 | **Fused Forward Alpha Override Semantics (Lines 2009-2069):** The comment at lines 2011-2015 explains that passing alpha_override forces the blending path even for TRAINING seeds. The code guards against this by only including slots that have explicit overrides in configs. However, if a solo ablation config sets `slot_id: 0.0` and that slot is in TRAINING stage with GATE alpha_algorithm, the code (lines 2036-2050) creates an alpha_schedule if one doesn't exist. This is a side effect of the fused validation pass that persists beyond the validation phase. |
| C8-27 | P3 | **Contribution Velocity EMA (Lines 2142-2150):** The contribution velocity uses a fixed EMA decay of 0.7. For the default 25-epoch episodes, this gives recent changes high weight. However, this value should probably be configurable or computed based on episode length for different task configurations. |
| C8-28 | P4 | **Shapley Results Approximation (Line 2166):** `shapley_results[i][shapley_tuple] = (0.0, acc)` - The loss component is hardcoded to 0.0 because "we only track acc here". This means Shapley-based attribution only considers accuracy, not loss reduction. For LM tasks, loss is more informative than token accuracy. |

#### PPO Update Phase

**Concerns:**

| ID | Severity | Finding |
|----|----------|---------|
| C8-29 | P3 | **Rollback Buffer Clearing (Lines 3183-3188):** When governor panic occurs, the entire environment's buffer is cleared (`agent.buffer.clear_env(env_idx)`). However, this happens AFTER the transitions were already added (line 3141-3175). This means the GAE computation for that environment's transitions will have been based on values that are now stale (the model rolled back). The correct approach would be to NOT add transitions from rolled-back environments, or to recompute advantages after rollback. |
| C8-30 | P4 | **Metrics Aggregation for Dict Values (Line 2288-2289):** `_aggregate_ppo_metrics()` takes the FIRST dict for keys like `head_entropies` and `ratio_diagnostic`. If values differ across PPO epochs, only the first is kept. This could hide important information about entropy collapse across update epochs. |

---

## Cross-Cutting Integration Risks

### 1. Hidden State Leakage Across Episodes (HIGH RISK)

**Location:** `parallel_env_state.py:reset_episode_state()`, `vectorized.py:2939-2954`

The LSTM hidden state is NOT reset when `reset_episode_state()` is called. The reset only happens on `done=True` inside the epoch loop. If an episode is terminated and a new one starts in the same batch, the hidden state reset happens correctly. But if episodes span batches, the hidden state persists via `batched_lstm_hidden` which is re-created fresh each batch anyway (line 1711).

**Assessment:** After careful analysis, this is ACTUALLY CORRECT. Each batch creates fresh environments with fresh models. The `batched_lstm_hidden` is initialized to None at line 1711 and created fresh on first use (line 2452-2455). The concern in C8-11 is mitigated by the batch-level environment recreation.

### 2. Observation Normalization Distribution Shift

**Location:** `vectorized.py:2436-2440`, `_run_ppo_updates:3304-3314`

The one-batch lag in normalizer updates means early training has non-representative statistics. This is a known PPO issue (Schulman et al., 2017 didn't use obs normalization; it was added by OpenAI baselines). The current implementation is industry-standard.

### 3. Reward Normalization Interaction with PBRS

**Location:** `vectorized.py:2656`

The reward normalizer operates on the TOTAL reward (including PBRS shaping components). This is correct - PBRS guarantees policy invariance so normalizing the shaped reward is equivalent to normalizing the sparse reward. However, if the PBRS potential function has high variance (e.g., large terminal bonuses), the normalizer will need many samples to stabilize.

### 4. Action Masking Contract

**Location:** `vectorized.py:2402-2414`, `tamiyo/policy/action_masks.py`

Action masks are computed based on current model state BEFORE action selection. Between mask computation and action execution, the model state can change (via `step_epoch()` or other subsystems). The code handles this defensively by re-checking preconditions at execution time (e.g., line 2676), but this creates a contract where masking is "advisory" rather than "guaranteed".

---

## Severity-Tagged Findings List

### P0 - Critical (None)

No critical bugs found. The core RL loop is sound.

### P1 - Correctness Bugs

| ID | Location | Finding |
|----|----------|---------|
| C8-11 | `parallel_env_state.py:167-197` | `lstm_hidden` not reset in `reset_episode_state()`. **Mitigated by batch-level recreation.** |
| C8-16 | `vectorized.py:2939-2954` | Hidden state reset logic is correct but the `if done:` branch is redundant with epoch check. Minor cleanup. |
| C8-19 | `vectorized.py:1411-1422` | Seed optimizer lifecycle is fragile - relies on training batch execution order. Should be robust to any ordering. |

### P2 - Performance/Resource Issues

| ID | Location | Finding |
|----|----------|---------|
| C8-04 | `dual_ab.py:178-216` | Sequential training confounds A/B comparison due to CUDA warmup. |
| C8-07 | `helpers.py:213-304` | `_train_one_epoch()` doesn't handle mid-epoch pruning (but callsites don't have this issue). |
| C8-14 | `parallel_env_state.py:192` | Reaching into private `_last_matrix` of CounterfactualHelper. |
| C8-17 | `vectorized.py:2436-2440` | Observation normalizer one-batch lag (standard practice but noted). |
| C8-18 | `vectorized.py:2656` | Reward summary accumulates raw while buffer gets normalized - telemetry confusion. |
| C8-24 | `vectorized.py:1132-1136` | Multiple envs on same GPU have independent GradScalers that can diverge. |
| C8-26 | `vectorized.py:2036-2050` | Fused validation creates persistent alpha_schedule as side effect. |
| C8-29 | `vectorized.py:3183-3188` | Rollback clears buffer after transitions added - stale GAE values. |

### P3 - Code Quality/Maintainability

| ID | Location | Finding |
|----|----------|---------|
| C8-01 | `config.py:377-379` | `chunk_length == max_epochs` constraint prevents BPTT flexibility. |
| C8-03 | `config.py:433-445` | `reward_mode_per_env` not validated against `reward_family`. |
| C8-05 | `dual_ab.py:191-192` | MD5-based seed offset is fragile to group_id changes. |
| C8-08 | `helpers.py:311-381` | `_convert_flat_to_factored()` hardcodes `slot_idx=0`, breaks multi-slot heuristic. |
| C8-10 | `helpers.py:592-594` | `available_slots` calculation may not match heuristic policy expectations. |
| C8-12 | `parallel_env_state.py:94-96` | `defaultdict` can cause subtle bugs with `in` checks. |
| C8-20 | `vectorized.py:2734-2736` | Hindsight credit cap applied before accumulation, can exceed cap. |
| C8-21 | `vectorized.py:2524-2817` | Action validity checked twice (in parse and execution). |
| C8-23 | `vectorized.py:2756-2757` | MIN_PRUNE_AGE check duplicated in parse and execution. |
| C8-27 | `vectorized.py:2142-2150` | Contribution velocity EMA decay is hardcoded. |

### P4 - Style/Minor

| ID | Location | Finding |
|----|----------|---------|
| C8-02 | `config.py:283-284` | `entropy_steps` computed in multiple places. |
| C8-06 | `dual_ab.py:232-294` | Winner determination lacks statistical significance testing. |
| C8-09 | `helpers.py:384-721` | `run_heuristic_episode()` duplicates much of vectorized logic. |
| C8-13 | `parallel_env_state.py:104-109` | Action counter initialization could use leyline constant. |
| C8-15 | `policy_group.py:72` | `envs` field is placeholder for future work. |
| C8-22 | `vectorized.py:2808-2809` | WAIT always succeeds, inflates success metrics. |
| C8-25 | `vectorized.py:1392-1394` | `record_stream()` on clones is unnecessary (but harmless). |
| C8-28 | `vectorized.py:2166` | Shapley loss component hardcoded to 0.0. |
| C8-30 | `vectorized.py:2288-2289` | Dict metrics aggregation only keeps first value. |

---

## Recommendations

### High Priority

1. **Fix Rollback Buffer Handling (C8-29):** When governor rollback occurs, either (a) don't add transitions from that environment to the buffer, or (b) mark them for exclusion in GAE computation.

2. **Validate reward_mode_per_env vs reward_family (C8-03):** Add cross-validation to prevent incompatible mode/family combinations.

3. **Consider Parallel Dual-AB Training (C8-04):** For proper A/B testing, use multiprocessing to train groups truly in parallel, or alternate episodes in lockstep.

### Medium Priority

4. **Consolidate Action Validity Checks (C8-21, C8-23):** Either check once in parse (and trust the mask), or check once at execution (and remove from parse).

5. **Make Hindsight Credit Cap Per-Application (C8-20):** Cap `pending_hindsight_credit` when it's applied to reward, not when accumulated.

6. **Add CounterfactualHelper.reset() Method (C8-14):** Instead of reaching into `_last_matrix`, add a public method.

### Low Priority

7. **Make Contribution Velocity EMA Configurable (C8-27):** Add to TrainingConfig or compute from `max_epochs`.

8. **Add Statistical Significance to A/B Results (C8-06):** Report confidence intervals or p-values in the comparison output.

---

## Test Coverage Assessment

The existing tests in `test_vectorized.py` cover:
- Telemetry emission functions
- Seed advancement logic
- PPO update helpers (including target_kl early stopping)
- Anomaly handling
- Threshold detection

**Missing Test Coverage:**
- Bootstrap value computation for truncated episodes
- LSTM hidden state management across episode boundaries
- Rollback buffer handling
- Fused validation alpha override semantics
- Multi-slot action execution ordering

---

## Conclusion

The vectorized training loop is a sophisticated piece of RL engineering that correctly implements PPO with GAE, LSTM policies, action masking, and parallel environments. The core RL mechanics are sound. The identified issues are primarily around edge cases (rollback handling), code duplication (validity checks), and configuration flexibility (chunk_length constraints).

The most concerning issue from a correctness standpoint is C8-29 (stale GAE after rollback), which could cause learning instability during governor interventions. However, governor rollback is a rare safety mechanism, so this may not affect typical training runs.

Overall assessment: **Production-ready with minor improvements recommended.**
