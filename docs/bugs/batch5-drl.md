# Batch 5 Code Review: Simic Attribution + Control

**Reviewer Specialization:** Deep Reinforcement Learning
**Date:** 2025-12-27
**Branch:** ux-overwatch-refactor

## Executive Summary

This batch covers counterfactual attribution (Shapley values for credit assignment) and observation/reward normalization. These are critical for RL:

- **Attribution**: Determines which seeds get credit/blame, affecting lifecycle decisions
- **Normalization**: Directly impacts policy gradient stability and value function learning

Overall quality is solid with well-documented code and good numerical stability practices. I found 1 P1 issue (reproducibility from unseeded random), 2 P2 issues (variance bias, EMA documentation gap), and several P3/P4 items.

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py`

**Purpose:** Core counterfactual engine computing Shapley values for seed contribution attribution. Implements factorial matrix, ablation, and permutation sampling approaches.

**Strengths:**
- Excellent docstring explaining the confounding issue (removal cost vs causal contribution)
- Multiple strategies (full_factorial, shapley, ablation_only) with auto-selection
- Antithetic variance reduction for Shapley sampling
- Proper timeout handling between configs (CUDA-safe pattern)
- Telemetry integration via callback injection

**Concerns:**

| Severity | Line | Issue |
|----------|------|-------|
| **P1** | 353, 402 | **Unseeded random.shuffle** - Shapley value computation uses `random.shuffle` without setting a seed. This breaks reproducibility for debugging and makes experiment comparison unreliable. Two runs with identical configs will produce different Shapley estimates. |
| **P2** | 424-427 | **Population vs sample variance** - Uses `sum((v - mean) ** 2) / len(values)` which is population variance (N denominator) but the `ShapleyEstimate.std` is semantically a sample standard deviation. For 20 samples, this underestimates true variance by ~5%. Should use `/ (len(values) - 1)` for sample variance. |
| **P3** | 155-159 | **is_significant hardcoded z-score** - Only supports 95% and 99% confidence. The hardcoded logic is fine but the signature suggests arbitrary confidence is supported. |
| **P3** | 174-180 | **Magic threshold 0.5 in regime classification** - The synergy/interference thresholds (0.5, -0.5) are hardcoded with no reference to expected accuracy scales. These should be config parameters or documented. |
| **P3** | 396 | **Capping n_perms at 100** - The Shapley sampling caps at 100 permutations regardless of shapley_samples config. This silent cap could surprise users expecting more samples. |
| **P4** | 352 | **Integer division** - `n_samples // 2` is correct but inconsistent with `n_samples / 2` on line 351. Both work but one is cleaner. |

**RL-Specific Observations:**

The counterfactual methodology correctly measures *removal cost* (how much accuracy drops when disabling a seed) rather than *causal contribution* (how much the seed added). The docstring explicitly acknowledges this distinction - excellent. However:

1. **Credit assignment timing**: Shapley values are computed at episode end. During early training when the host is adapting, these values are noisy. Consider temporal smoothing (EMA of Shapley values across episodes).

2. **Interaction terms capped at n<=3**: The `compute_interaction_terms` function returns empty for n>3 seeds. This is fine for complexity reasons but means synergy/interference detection fails silently when you have 4+ active seeds.

---

### 2. `/home/john/esper-lite/src/esper/simic/attribution/counterfactual_helper.py`

**Purpose:** Training-loop-friendly wrapper around CounterfactualEngine. Handles telemetry wiring and result processing.

**Strengths:**
- Clean separation between engine (computation) and helper (integration)
- `get_required_configs` enables fusion with validation pass
- `compute_simple_ablation` provides lightweight fallback

**Concerns:**

| Severity | Line | Issue |
|----------|------|-------|
| **P3** | 143 | **Heuristic for Shapley computation** - `len(matrix.configs) > len(slot_ids) + 1` decides whether to compute Shapley. This works but the reasoning isn't documented. It assumes ablation_only produces exactly n+2 configs. |
| **P4** | 108-109 | **Unused epoch parameter** - `epoch` is accepted but never used in `compute_contributions`. The helper ignores it but the engine uses `matrix.epoch` in telemetry. |
| **P4** | 164-168 | **get_interaction_terms only works after computation** - Returns empty dict before any computation. Could document this or raise an error. |

**RL-Specific Observations:**

The `ContributionResult` dataclass correctly separates marginal contribution from Shapley values. This is good practice since:
- Marginal contribution is faster (O(2^n) or O(n) depending on configs available)
- Shapley values require more computation but are theoretically "fair"
- Having both allows the reward system to make informed tradeoffs

---

### 3. `/home/john/esper-lite/src/esper/simic/attribution/__init__.py`

**Purpose:** Clean public API re-export.

**Assessment:** No issues. Standard module pattern with `__all__` for explicit exports.

---

### 4. `/home/john/esper-lite/src/esper/simic/control/__init__.py`

**Purpose:** Control interface exports.

**Assessment:** No issues. Clean re-export.

---

### 5. `/home/john/esper-lite/src/esper/simic/control/normalization.py`

**Purpose:** Running mean/std for observation normalization and reward scaling. Critical for PPO stability.

**Strengths:**
- Uses Welford's online algorithm - excellent for numerical stability
- GPU-native with auto-migration (avoids CPU sync on hot path)
- Momentum option for EMA (prevents distribution shift in long runs)
- RewardNormalizer correctly divides by std only (no mean subtraction)
- Extensive thread safety documentation (H13)
- State dict methods for checkpointing

**Concerns:**

| Severity | Line | Issue |
|----------|------|-------|
| **P2** | 93-105 | **EMA variance formula undocumented complexity** - The law of total variance cross-term is correctly implemented but the comment explaining it could be clearer. The formula `m*(1-m)*(old_mean - batch_mean)^2` is the cross-term, not just `delta^2`. More importantly, this EMA variance is a *biased* estimator - it weights recent observations more but doesn't account for the effective sample size. Over very long runs, this can underestimate true variance. |
| **P3** | 56 | **count initialized to epsilon** - `self.count = torch.tensor(epsilon, device=device)` starts count at 1e-4 not 0. This is intentional (prevents div-by-zero in first update) but differs from RewardNormalizer which starts at 0. Document the asymmetry. |
| **P3** | 217 | **std computed as sample std** - RewardNormalizer uses `(self.m2 / (self.count - 1)) ** 0.5` (sample std) but RunningMeanStd stores population variance and takes sqrt directly. They're slightly inconsistent estimators. |
| **P4** | 139-145 | **to() modifies in place but also returns self** - Standard PyTorch pattern but could trip up users expecting a pure transform. |

**RL-Specific Observations:**

1. **Observation Normalization Impact on PPO:**
   - The `normalize()` method with clipping to [-10, 10] is standard practice
   - EMA momentum=0.99 (used in vectorized.py) means ~100-step half-life for statistics updates
   - This is conservative - prevents sudden distribution shifts that break the PPO ratio calculation
   - However, if the observation distribution changes fundamentally (e.g., after fossilization), the EMA may take many batches to adapt

2. **Reward Normalization Design:**
   - Dividing by std only (no mean subtraction) is the correct choice for critic stability
   - The docstring explains why: mean subtraction creates non-stationary targets
   - However, the `clip=10.0` means rewards outside [-10*std, 10*std] get compressed
   - With sparse rewards (24 zeros then spike), the initial std estimate is unstable

3. **Non-Stationarity from Normalization During Training:**
   - Normalizing observations during rollout collection vs PPO updates uses the same statistics
   - This is correct - you freeze statistics during rollout, update after
   - The code in `vectorized.py` confirms this pattern (update happens in `_do_ppo_updates`)

4. **Numerical Stability with Sparse Data:**
   - RewardNormalizer handles the sparse reward case correctly (returns clipped raw for <2 samples)
   - The Welford algorithm is stable even with extreme values
   - Tests in `test_sparse_training.py` verify this works

---

## Cross-Cutting Integration Risks

### 1. Attribution-Reward Coupling

The counterfactual attribution feeds into reward computation (via `ContributionRewardConfig`). If Shapley values are noisy due to:
- Few permutation samples (default 20)
- Unseeded randomness (P1 issue above)
- High variance in underlying evaluation function

Then the reward signal to the policy is noisy, potentially causing training instability.

**Recommendation:** Consider clamping or smoothing Shapley-based rewards.

### 2. Normalization State Across Checkpoints

From `vectorized.py` lines 816-835, checkpoint restoration manually reconstructs normalizer state from metadata. This is fragile:
- `momentum` is optional in the checkpoint
- Shape mismatches aren't validated
- Count could be negative if corrupted

The `state_dict`/`load_state_dict` methods exist but aren't used for checkpointing. This is inconsistent.

### 3. Device Migration Latency

`RunningMeanStd.to()` does a synchronous copy when stats migrate between devices. The warning comment is good, but in multi-GPU setups this could cause unexpected stalls if an observation tensor arrives on a different device than expected.

### 4. RewardNormalizer vs RunningMeanStd Discrepancy

Two different normalization classes with subtly different semantics:
- RunningMeanStd: count starts at epsilon, uses population variance
- RewardNormalizer: count starts at 0, uses sample variance

This could confuse maintainers. Consider unifying or documenting the differences more explicitly.

---

## Severity-Tagged Findings Summary

### P1 - Critical/Correctness

| ID | File | Line | Finding |
|----|------|------|---------|
| 5.1 | counterfactual.py | 353, 402 | Unseeded `random.shuffle` breaks reproducibility of Shapley estimates |

### P2 - Performance/Subtle Bugs

| ID | File | Line | Finding |
|----|------|------|---------|
| 5.2 | counterfactual.py | 424-427 | Population variance used where sample variance expected (underestimates uncertainty) |
| 5.3 | normalization.py | 93-105 | EMA variance is biased estimator; may underestimate true variance in long runs |

### P3 - Code Quality/Maintainability

| ID | File | Line | Finding |
|----|------|------|---------|
| 5.4 | counterfactual.py | 155-159 | Hardcoded z-scores in is_significant() |
| 5.5 | counterfactual.py | 174-180 | Magic threshold 0.5 for synergy/interference |
| 5.6 | counterfactual.py | 396 | Silent cap at 100 permutations |
| 5.7 | counterfactual_helper.py | 143 | Undocumented heuristic for Shapley computation |
| 5.8 | normalization.py | 56 | RunningMeanStd count starts at epsilon vs RewardNormalizer at 0 |
| 5.9 | normalization.py | 217 | Inconsistent variance estimators between classes |

### P4 - Style/Minor

| ID | File | Line | Finding |
|----|------|------|---------|
| 5.10 | counterfactual.py | 352 | Inconsistent `/ 2` vs `// 2` for integer division |
| 5.11 | counterfactual_helper.py | 108-109 | Unused epoch parameter |
| 5.12 | counterfactual_helper.py | 164-168 | get_interaction_terms silently returns empty before computation |
| 5.13 | normalization.py | 139-145 | to() modifies in place AND returns self |

---

## Test Coverage Assessment

| Component | Coverage | Notes |
|-----------|----------|-------|
| RunningMeanStd | Good | Unit tests + property tests (convergence, bounds, edge cases) |
| RewardNormalizer | Good | Sparse reward scenarios tested |
| CounterfactualEngine | Moderate | Telemetry emission tested, but no tests for Shapley accuracy |
| CounterfactualHelper | Moderate | Integration tests exist but no unit tests |
| Interaction terms | Weak | No tests for compute_interaction_terms |

**Missing Tests:**
1. Shapley value correctness (known-answer test with simple game)
2. Counterfactual with failing evaluation function (exception path)
3. Interaction term calculation verification
4. EMA normalization vs Welford normalization comparison

---

## Recommendations

1. **Fix P1 immediately**: Add `random.seed()` call or accept a `seed` parameter in CounterfactualConfig for reproducibility.

2. **Address variance bias (P2)**: Use `n-1` denominator in Shapley std calculation.

3. **Add known-answer Shapley test**: Create a simple 2-player game where Shapley values are analytically known, verify the implementation matches.

4. **Unify normalization classes**: Either merge RunningMeanStd and RewardNormalizer base logic, or add explicit documentation about why they differ.

5. **Use state_dict for checkpointing**: The normalizers have proper state_dict methods; use them in checkpoint save/restore instead of manual metadata extraction.
