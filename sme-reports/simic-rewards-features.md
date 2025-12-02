# Simic Rewards and Features Analysis

**Reviewer:** Deep RL + PyTorch SME
**Date:** 2025-12-02
**Files Analyzed:**
- `/home/john/esper-lite/src/esper/simic/rewards.py` (733 lines)
- `/home/john/esper-lite/src/esper/simic/features.py` (166 lines)
- `/home/john/esper-lite/src/esper/simic/normalization.py` (88 lines)

---

## 1. rewards.py

### Purpose

Implements reward shaping for a seed lifecycle controller in a hierarchical RL system. The module provides both accuracy-primary (`compute_shaped_reward`) and loss-primary (`compute_loss_reward`) reward functions for online PPO training, offline data generation, and offline RL (IQL).

### Key Classes/Functions

| Name | Type | Description |
|------|------|-------------|
| `RewardConfig` | dataclass | 25+ hyperparameters for accuracy-primary reward shaping |
| `LossRewardConfig` | dataclass | Configuration for loss-primary rewards with task-specific presets |
| `SeedInfo` | NamedTuple | Lightweight seed state for hot-path reward computation |
| `compute_shaped_reward()` | function | Main reward function combining base, PBRS, and action-specific shaping |
| `compute_loss_reward()` | function | Phase-2 loss-primary reward for cross-task comparability |
| `compute_seed_potential()` | function | Potential function for PBRS over seed lifecycle stages |
| `compute_pbrs_bonus()` | function | Standard PBRS: gamma * phi(s') - phi(s) |
| `_germinate_shaping()` | function | Action-specific shaping for GERMINATE |
| `_advance_shaping()` | function | Action-specific shaping for FOSSILIZE |
| `_cull_shaping()` | function | Action-specific shaping for CULL with age protection |
| `_wait_shaping()` | function | Action-specific shaping for WAIT |

### DRL Assessment

**Reward Shaping Correctness:**

1. **PBRS Implementation (Correct):** The `compute_pbrs_bonus()` function correctly implements F(s,s') = gamma * phi(s') - phi(s) per Ng et al. (1999). This guarantees policy invariance when added to any base reward.

2. **Potential Function Design (Well-designed):** `compute_seed_potential()` uses monotonically increasing stage potentials (2.0 -> 7.5) with a capped progress bonus. The flattening from original (5->35) to (2->7.5) addresses "fossilization farming" - a reward hacking vector where agents would rush to FOSSILIZED stage for the PBRS bonus rather than for actual accuracy improvement.

3. **Stage Transition Tracking (Correct):** Lines 309-328 properly reconstruct previous state for PBRS calculation using `epochs_in_stage` and `previous_stage`. When `epochs_in_stage == 0`, it correctly uses `previous_stage`; otherwise, it uses current stage with decremented epoch count.

4. **Terminal PBRS Correction in CULL (Correct):** Lines 452-460 properly account for potential loss when culling destroys a seed. This is critical for PBRS validity - terminal states must have their potential zeroed.

5. **Action-Specific Shaping (Well-balanced):**
   - Germinate bonuses set to 0.0 to prevent churn exploitation (line 64-65)
   - Age penalty in `_cull_shaping()` prevents "germinate then immediately cull" anti-pattern
   - FOSSILIZE only rewarded from PROBATIONARY state (respects Leyline state machine)
   - WAIT correctly treated as neutral/positive for mechanical stages

**PBRS Validity Concerns:**

| Issue | Severity | Analysis |
|-------|----------|----------|
| Action-specific shaping is NOT PBRS | MEDIUM | Functions `_germinate_shaping`, `_advance_shaping`, etc. add rewards based on action AND state, not just state transitions. This is classic reward shaping, not PBRS, meaning it CAN change optimal policy. However, the design appears intentional to guide exploration. |
| Dual potential systems | LOW | `compute_potential()` (line 521) and `compute_seed_potential()` are separate potential functions. Only `compute_seed_potential()` is used in the main reward. `compute_potential()` appears unused. |

### PyTorch Assessment

**Hot Path Efficiency:**

1. **No PyTorch Operations:** This module is pure Python - no tensor operations. Designed intentionally for the hot path to avoid GPU synchronization.

2. **NamedTuple for SeedInfo (Excellent):** Using `NamedTuple` (line 172) instead of a full dataclass avoids allocation overhead in the hot path.

3. **Default Config Singleton (Good):** `_DEFAULT_CONFIG` (line 513) prevents repeated `RewardConfig()` allocations.

4. **Dictionary-Based Lookups:** `stage_potentials` dict (line 589) and `INTERVENTION_COSTS_BY_NAME` (line 693) use dict lookups (O(1)) rather than conditionals.

### Issues

| Severity | Line(s) | Issue |
|----------|---------|-------|
| **MEDIUM** | 331-341 | Action-specific shaping via `action.name` string comparison is fragile. If action names change, rewards break silently. Consider using action indices or an explicit mapping. |
| **MEDIUM** | 521-539 | `compute_potential()` function appears unused. If it's legacy code, delete it per project policy. If intended for future use, mark with TODO. |
| **LOW** | 241 | `action_enum` parameter documented as "NEW" but only used for legacy compat comment. Unclear if actually needed. |
| **LOW** | 589-596 | `stage_potentials` dict is recreated on every `compute_seed_potential()` call. Should be module-level constant like `_STAGE_POTENTIALS` (line 609). |
| **LOW** | 276-280 | Compute rent uses `growth_ratio ** compute_rent_exponent` which is expensive for non-integer exponents. Consider lookup table for common cases. |

### Recommendations

1. **Consolidate Potential Dictionaries:** Move `stage_potentials` (line 589) to module level to match `_STAGE_POTENTIALS` pattern. Consider merging them if semantically equivalent.

2. **Remove or Document `compute_potential()`:** The function appears unused. Per project policy (No Legacy Code), delete it or add clear documentation for intended future use.

3. **Add Reward Component Logging:** For debugging reward hacking, consider adding optional logging of individual reward components (base, PBRS, action shaping, rent).

4. **Clarify PBRS vs Non-PBRS:** Add docstring clarification that action-specific shaping is intentionally NOT PBRS and may influence optimal policy.

---

## 2. features.py

### Purpose

Extracts features from observation dictionaries for RL training. Explicitly designed for the hot path with minimal imports (only `leyline`). Provides both raw feature extraction (`obs_to_base_features`) and normalized observation transformation.

### Key Classes/Functions

| Name | Type | Description |
|------|------|-------------|
| `safe()` | function | Converts values to safe floats, handling None/inf/nan |
| `obs_to_base_features()` | function | Extracts 27-dimensional feature vector from observation dict |
| `TaskConfig` | dataclass | Task-specific normalization parameters |
| `normalize_observation()` | function | Normalizes observations for stable PPO training |

### DRL Assessment

**Feature Engineering:**

1. **Feature Dimensions (27):** Well-designed feature set covering:
   - Temporal: epoch, global_step (2)
   - Performance: losses (3), accuracies (3)
   - Trends: plateau, best values (3)
   - History: 5-step windows for loss and accuracy (10)
   - Seed state: 5 features
   - Capacity: 1

2. **Normalization Strategy (Phase 2):** `normalize_observation()` applies task-specific normalization:
   - Time features: [0, 1] range
   - Loss features: relative to achievable range
   - Loss delta: z-score normalization using task-specific std
   - Seed stage: [0, 1] range (divided by 7)

3. **Missing Features for Offline RL:** No features for behavior policy likelihood or dataset statistics, which are important for offline RL methods (CQL, IQL).

### PyTorch Assessment

**Hot Path Compliance:**

1. **Import Discipline (Excellent):** Only imports from `leyline`. Type hints guarded by `TYPE_CHECKING`. Comment explicitly warns about hot path constraints.

2. **Pure Python (Intentional):** No tensor operations - designed to run before tensorization.

3. **List Comprehensions:** Feature extraction uses Python list operations. For large batch sizes, this could become a bottleneck.

### Issues

| Severity | Line(s) | Issue |
|----------|---------|-------|
| **HIGH** | 92-93 | List unpacking with `*[safe(v, 10.0) for v in obs['loss_history_5']]` and `*obs['accuracy_history_5']` creates intermediate lists. For vectorized training, this is called per observation. |
| **MEDIUM** | 152-165 | `normalize_observation()` returns a new dict with only 9 keys, but `obs_to_base_features()` expects 27 features. These functions appear incompatible. |
| **MEDIUM** | 18 | Imports `TensorSchema, TENSOR_SCHEMA_SIZE` from leyline but never uses them. |
| **LOW** | 37-52 | `safe()` function does two `isinstance` checks and bounds checking per value. For history arrays (10 values), this adds overhead. |

### Recommendations

1. **Vectorize Feature Extraction:** For batch processing, provide a `obs_batch_to_features()` that operates on stacked numpy arrays or tensors directly, avoiding per-observation Python overhead.

2. **Reconcile Normalization with Feature Extraction:** Either:
   - Expand `normalize_observation()` to produce all 27 normalized features, OR
   - Document that these serve different purposes (e.g., logging vs training)

3. **Remove Unused Imports:** `TensorSchema` and `TENSOR_SCHEMA_SIZE` are imported but not used.

4. **Pre-allocate Feature Buffer:** For known observation structure, pre-allocate output list to avoid dynamic resizing.

---

## 3. normalization.py

### Purpose

Provides GPU-native running mean/std normalization using Welford's numerically stable online algorithm. Used by vectorized PPO for observation preprocessing while keeping all operations on-device.

### Key Classes/Functions

| Name | Type | Description |
|------|------|-------------|
| `RunningMeanStd` | class | Online mean/variance tracking with Welford's algorithm |
| `update()` | method | Updates statistics from new observation batch |
| `normalize()` | method | Applies normalization with clipping |
| `to()` | method | Moves statistics to specified device |

### DRL Assessment

**Observation Normalization:**

1. **Welford's Algorithm (Correct):** Lines 47-61 implement Welford's online algorithm for numerically stable variance computation. This is the correct approach for streaming data.

2. **Epsilon Initialization:** `count` initialized to `epsilon` (line 26) prevents division by zero and provides sensible initial variance estimate.

3. **Clipping (Standard):** Default clip value of 10.0 is standard for PPO observation normalization.

4. **Missing Decay/Windowing:** No mechanism for forgetting old statistics. For non-stationary environments or curriculum learning, a windowed or exponential moving average might be needed.

### PyTorch Assessment

**GPU-Native Design (Excellent):**

1. **Device-Aware Tensors:** All statistics stored as tensors, not Python floats.

2. **Tensor Count (Smart):** Line 26 uses `torch.tensor(epsilon, device=device)` for count instead of a Python int. This keeps all arithmetic on-device.

3. **Auto-Migration:** `update()` method (lines 37-39) automatically migrates stats to input device on first call, with one-time migration cost.

4. **@torch.no_grad():** Update method correctly wrapped to prevent gradient tracking overhead.

### Issues

| Severity | Line(s) | Issue |
|----------|---------|-------|
| **HIGH** | 42-43 | `x.mean(dim=0)` and `x.var(dim=0)` are reduction operations that trigger GPU synchronization when batch dimension is small. For single observations, this is expensive. |
| **MEDIUM** | 38-39 | Device comparison `self.mean.device != x.device` may trigger synchronization. Consider caching device as string and comparing strings. |
| **MEDIUM** | 63-71 | `normalize()` lacks `@torch.no_grad()` decorator. While typically called in inference context, explicit decoration prevents accidental gradient computation. |
| **LOW** | 56 | `delta ** 2` followed by multiplication - could be `delta.square()` for clarity, though performance is identical. |
| **LOW** | 73-79 | `to()` method should return `self` for chaining (it does), but type hint shows `"RunningMeanStd"` as string - should use `Self` from typing or forward ref properly. |

### Recommendations

1. **Add `@torch.no_grad()` to `normalize()`:** Explicit is better than implicit for preventing gradient tracking.

2. **Consider Batch Size Threshold:** For small batches (n < 32), reduction operations may be slower than CPU. Consider a fast path for single-observation updates.

3. **Add State Dict Methods:** For checkpointing, add `state_dict()` and `load_state_dict()` methods to save/restore normalization statistics.

4. **Document Non-Stationarity:** Add note about when to reset or use windowed statistics for curriculum learning or domain shift.

---

## Summary of Critical Issues

| File | Severity | Issue | Impact |
|------|----------|-------|--------|
| rewards.py | MEDIUM | Unused `compute_potential()` function | Code bloat, confusion |
| rewards.py | MEDIUM | Action name string matching fragile | Silent breakage on refactor |
| features.py | HIGH | List unpacking in hot path | O(n) allocations per observation |
| features.py | MEDIUM | `normalize_observation()` incompatible with `obs_to_base_features()` | API confusion |
| normalization.py | HIGH | Reduction sync on small batches | GPU underutilization |
| normalization.py | MEDIUM | Missing `@torch.no_grad()` on normalize | Potential gradient leaks |

## Overall Assessment

**rewards.py:** Well-designed reward shaping with proper PBRS implementation. The separation of PBRS-compatible stage bonuses from action-specific shaping is correct. Minor cleanup needed for unused code.

**features.py:** Functional but not optimized for vectorized training. The hot-path discipline is good (import restrictions), but Python-level feature extraction will become a bottleneck at scale.

**normalization.py:** Clean GPU-native implementation of Welford's algorithm. The auto-migration pattern is elegant. Missing checkpoint support and windowed variants for non-stationary settings.

**Recommendation Priority:**
1. Add `@torch.no_grad()` to `normalize()` (quick fix, prevents bugs)
2. Delete unused `compute_potential()` (per No Legacy Code policy)
3. Vectorize feature extraction for batch processing (performance)
4. Add state dict methods to RunningMeanStd (checkpointing support)
