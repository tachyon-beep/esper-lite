# DRL Code Review: `simic/rewards/` Subfolder

**Reviewer:** Deep Reinforcement Learning Specialist
**Date:** 2025-12-17
**Files Reviewed:**
- `src/esper/simic/rewards/rewards.py` (1377 lines)
- `src/esper/simic/rewards/reward_telemetry.py` (104 lines)
- `src/esper/simic/rewards/__init__.py` (88 lines)

---

## Executive Summary

The rewards module implements a sophisticated multi-component reward function for seed lifecycle control in a morphogenetic neural network system. The design demonstrates **strong theoretical grounding** in potential-based reward shaping (PBRS, Ng et al., 1999) and incorporates **multiple anti-gaming mechanisms** to prevent reward hacking. The code is well-documented with extensive inline rationale and property-based tests.

**Overall Assessment:** GOOD with some areas for improvement.

**Strengths:**
- Rigorous PBRS implementation with telescoping property preservation
- Comprehensive anti-ransomware mechanisms (attribution discount, ratio penalty)
- Multi-mode reward support (SHAPED, SPARSE, MINIMAL) for credit assignment research
- Excellent telemetry infrastructure for reward debugging
- Extensive property-based test coverage

**Areas for Concern:**
- PBRS gamma consistency relies on manual synchronization
- Some reward component scales may need empirical validation
- Holding indecision penalty may be too aggressive in edge cases
- Terminal bonus counting logic has potential double-counting risk

---

## Critical Issues

### C1: PBRS Gamma Synchronization is Fragile (rewards.py:119-122)

**Severity:** CRITICAL
**Lines:** 119-122, 203-204

The PBRS implementation correctly notes that `gamma_pbrs MUST equal gamma_ppo` for policy invariance (Ng et al., 1999), but the synchronization relies on importing `DEFAULT_GAMMA` from leyline rather than receiving it as a runtime parameter.

```python
# Line 119-122
# DEFAULT_GAMMA imported from leyline - single source of truth for PPO/PBRS gamma.
# PBRS theory requires gamma_pbrs == gamma_ppo for policy invariance (Ng et al., 1999).
```

**Problem:** If any caller constructs a `ContributionRewardConfig` with a non-default gamma that differs from the PPO trainer's gamma, PBRS policy invariance breaks. The comment warns about this, but there's no runtime enforcement.

**Impact:** Misconfigured gamma can change the optimal policy, invalidating the theoretical guarantees of PBRS. This would manifest as unexplained policy divergence that's extremely hard to debug.

**Recommendation:** Add runtime assertion in the reward computation path that validates gamma consistency with the PPO trainer, or better, pass gamma explicitly from the PPO trainer to the reward function.

---

### C2: Ratio Penalty Division by Zero Guard Missing (rewards.py:474-481)

**Severity:** CRITICAL
**Lines:** 474-481

The ratio penalty calculation divides by `total_imp` without guarding against zero:

```python
# Line 474-481
if total_imp > config.improvement_safe_threshold:
    ratio = seed_contribution / total_imp  # Potential division if threshold is 0.0
    if ratio > config.hacking_ratio_threshold:
        ratio_penalty = -min(...)
```

**Problem:** While `improvement_safe_threshold` is set to 0.1 by default, this is a configurable parameter. If set to 0.0, division by zero could occur when `total_imp` is exactly 0.0.

**Impact:** NaN propagation into rewards, causing training collapse.

**Recommendation:** Add explicit guard: `if total_imp > max(config.improvement_safe_threshold, 1e-8):`

---

## High-Priority Issues

### H1: Attribution Discount Sigmoid Steepness is Task-Dependent (rewards.py:466)

**Severity:** HIGH
**Lines:** 464-466

The attribution discount uses a fixed sigmoid steepness of 10.0:

```python
# Line 466
attribution_discount = 1.0 / (1.0 + math.exp(-config.attribution_sigmoid_steepness * total_imp))
```

**Analysis:** The sigmoid with steepness 10.0 means:
- At total_imp = -0.2: discount = 0.12 (88% penalty)
- At total_imp = -0.5: discount = 0.007 (99.3% penalty)

This is appropriate for CIFAR-10 where typical improvements are 1-5%, but may be too aggressive for tasks with naturally higher variance (e.g., TinyStories with loss-based metrics).

**Impact:** Task-specific tuning may be required. The current parameterization is exposed via config but the default may cause premature attribution zeroing on high-variance tasks.

**Recommendation:** Document the expected improvement ranges for different tasks. Consider adding task-specific preset configurations like `LossRewardConfig.for_cifar10()` already does.

---

### H2: Holding Indecision Penalty Exponential Growth (rewards.py:594-599)

**Severity:** HIGH
**Lines:** 594-599

The holding indecision penalty grows exponentially:

```python
# Line 596-599
epochs_waiting = seed_info.epochs_in_stage - 1
holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
holding_warning = max(holding_warning, -10.0)  # Capped at -10
```

**Schedule:**
- Epoch 2: -1.0
- Epoch 3: -3.0
- Epoch 4: -9.0
- Epoch 5+: -10.0 (capped)

**Analysis:** This aggressive escalation is designed to overcome the +7.5 maximum attribution reward (per DRL Expert comment). However, the exponential growth assumes that WAIT-farming at epoch 4+ is always pathological.

**Concern:** If the environment has high observation noise, the agent may legitimately need more information-gathering time. The penalty could force premature FOSSILIZE/CULL decisions on seeds where the optimal action is genuinely unclear.

**Impact:** Suboptimal policy convergence on noisy environments.

**Recommendation:** Consider softening to linear growth or adding a configurable penalty function. The current implementation is well-suited for the controlled CIFAR-10 environment but may need adjustment for noisier domains.

---

### H3: Sparse Reward Clamping Reduces Gradient Signal (rewards.py:720-724)

**Severity:** HIGH
**Lines:** 720-724

The sparse reward function clamps output to [-1, 1]:

```python
# Line 724
return max(-1.0, min(1.0, reward))
```

With `sparse_reward_scale=2.5`, the raw reward `2.5 * (0.78) = 1.95` is clamped to 1.0.

**Problem:** The clamping negates the benefit of the scale factor. The DRL Expert comment at line 214 recommends trying 2.0-3.0 scale if learning fails, but the current clamping makes this ineffective.

**Impact:** Reduced gradient signal variance, potentially slower learning in SPARSE mode.

**Recommendation:** Either remove clamping (allow rewards in [-scale, +scale]) or document that clamping defeats the scale parameter's purpose. The bounded reward may be intentional for training stability, but the interaction with scale should be clarified.

---

### H4: PBRS Telescoping Warning for Zero Previous Epochs (rewards.py:962-967)

**Severity:** HIGH
**Lines:** 962-967

The code logs a warning when `previous_epochs_in_stage=0`:

```python
# Line 962-967
if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
    _logger.warning(
        "PBRS telescoping risk: transition from stage %d with previous_epochs_in_stage=0. "
        "phi_prev will be underestimated...",
        seed_info.previous_stage,
    )
```

**Problem:** This warning suggests the upstream `SeedInfo` construction may not always populate `previous_epochs_in_stage` correctly. Underestimated `phi_prev` inflates the PBRS bonus, potentially encouraging unnecessary stage transitions.

**Impact:** PBRS telescoping breaks down, causing reward scale drift over training. This is acknowledged in the property tests with relaxed tolerances.

**Recommendation:** Fix upstream `SeedInfo` construction in `kasmina` to always populate correct `previous_epochs_in_stage`, or formalize this as accepted implementation limitation with explicit documentation of the impact magnitude.

---

## Medium-Priority Issues

### M1: Fossilize Terminal Bonus Double-Counting Risk (rewards.py:667)

**Severity:** MEDIUM
**Lines:** 659-668

The terminal bonus calculation:

```python
# Line 667
fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
```

**Concern:** The `num_contributing_fossilized` is passed in from the caller. If the same seed is counted multiple times across environments or the count is stale, the terminal bonus could be inflated.

**Impact:** Overestimated episode returns, biased value function.

**Recommendation:** Add assertions or documentation about the expected count semantics. Consider computing the count internally from a list of fossilized seeds rather than accepting it as a parameter.

---

### M2: Proxy Signal Weight Ratio Not Enforced (rewards.py:155-157)

**Severity:** MEDIUM
**Lines:** 150-157

The comment documents a 3:1 ratio:

```python
# Line 155-157
# Proportionally reduced from 1.0 to 0.3 (maintains 3:1 ratio with contribution_weight)
proxy_contribution_weight: float = 0.3
```

**Problem:** This ratio constraint is documented but not enforced. If `contribution_weight` is changed without updating `proxy_contribution_weight`, the intentional scaling relationship breaks.

**Impact:** Inconsistent reward signals between pre-blending and post-blending stages.

**Recommendation:** Either compute `proxy_contribution_weight` as a derived property (`contribution_weight / 3.0`) or add a `__post_init__` validation.

---

### M3: Cull Shaping Has Asymmetric Scaling (rewards.py:1069)

**Severity:** MEDIUM
**Lines:** 1063-1069

The cull shaping for good seeds has unbounded negative scaling:

```python
# Line 1069
return config.cull_good_seed_penalty - 0.05 * seed_contribution
```

For a seed with 40% contribution, penalty = -0.3 - 2.0 = -2.3. This is much larger than `cull_hurting_bonus` (0.3) for culling bad seeds.

**Analysis:** The asymmetry is intentional (penalizing bad decisions more than rewarding good ones), but the unbounded negative scaling could produce extreme penalties for high-contribution seeds.

**Impact:** Strong negative gradients for culling good seeds, which is behaviorally correct but may cause training instability.

**Recommendation:** Consider capping the penalty: `return max(config.cull_good_seed_penalty - 0.05 * seed_contribution, -2.0)`.

---

### M4: Loss-Primary Reward Config Task Presets are Limited (rewards.py:266-281)

**Severity:** MEDIUM
**Lines:** 266-281

Only CIFAR-10 and TinyStories presets exist for `LossRewardConfig`:

```python
@staticmethod
def for_cifar10() -> "LossRewardConfig":
    ...

@staticmethod
def for_tinystories() -> "LossRewardConfig":
    ...
```

**Problem:** The ROADMAP indicates plans for additional domains. New tasks will need their own presets with correct `baseline_loss`, `target_loss`, and `typical_loss_delta_std`.

**Impact:** Loss-primary reward mode will underperform on new tasks until proper presets are created.

**Recommendation:** Document the process for deriving these parameters from a new task (e.g., measure random init loss, achievable loss from baseline runs, etc.).

---

### M5: Intervention Costs Dictionary Duplicates Config (rewards.py:1327-1332)

**Severity:** MEDIUM
**Lines:** 1327-1332

```python
INTERVENTION_COSTS: dict[LifecycleOp, float] = {
    LifecycleOp.WAIT: 0.0,
    LifecycleOp.GERMINATE: -0.02,
    LifecycleOp.FOSSILIZE: -0.01,
    LifecycleOp.CULL: -0.005,
}
```

These values duplicate those in `ContributionRewardConfig`:
- `germinate_cost: float = -0.02`
- `fossilize_cost: float = -0.01`
- `cull_cost: float = -0.005`

**Impact:** Configuration drift if one is updated without the other.

**Recommendation:** Remove the dictionary or derive it from a default config instance.

---

## Low-Priority Suggestions

### L1: Unwired Telemetry Functions (rewards.py:1076-1077)

**Lines:** 1076-1077

```python
# TODO: [UNWIRED TELEMETRY] - Call _check_reward_hacking() and _check_ransomware_signature()
# from compute_contribution_reward() when attribution is computed.
```

The reward hacking detection functions `_check_reward_hacking()` and `_check_ransomware_signature()` are implemented but not called. They require a telemetry hub parameter not available in the current call path.

**Recommendation:** Wire these up or remove them if telemetry is handled elsewhere.

---

### L2: RewardComponentsTelemetry Has Unused Field (reward_telemetry.py:21)

**Lines:** 21

```python
base_acc_delta: float = 0.0
```

This field is never set in `compute_contribution_reward()`. It appears to be a legacy field from an earlier reward design.

**Recommendation:** Remove unused field or document its intended use.

---

### L3: Stage Value 5 Gap Documentation (rewards.py:114)

**Lines:** 108-117

```python
STAGE_POTENTIALS = {
    ...
    4: 3.5,   # BLENDING
    # Value 5 intentionally skipped (was SHADOWING, removed)
    6: 5.5,   # HOLDING
    ...
}
```

The gap at value 5 is documented in multiple places but could cause confusion when iterating over stages numerically.

**Recommendation:** Add a comment in leyline/stages.py about this historical artifact and why it's preserved (enum value stability).

---

### L4: compute_seed_potential Uses Hardcoded Progress Bonus (rewards.py:1242-1243)

**Lines:** 1242-1243

```python
# epoch_progress_bonus=0.3, max_progress_bonus=2.0
progress_bonus = min(epochs_in_stage * 0.3, 2.0)
```

These match `ContributionRewardConfig` defaults but are hardcoded here rather than parameterized.

**Recommendation:** Either accept a config parameter or document that these must stay synchronized.

---

### L5: Negative Terminal Bonus Possible (rewards.py:662)

**Lines:** 660-668

```python
if epoch == max_epochs:
    terminal_bonus = val_acc * config.terminal_acc_weight  # Could be 0 if val_acc is 0
    fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
    terminal_bonus += fossilize_terminal_bonus
```

If `val_acc = 0` and `num_contributing_fossilized = 0`, terminal bonus is 0.0. This is fine, but there's no scenario where terminal bonus is negative. The comment at line 138 of test_reward_semantics.py notes "At terminal, bonus CAN be zero".

**Non-issue:** This is correct behavior. Documenting for completeness.

---

## Reward Design Principles Assessment

### PBRS Correctness

The implementation correctly follows Ng et al. (1999):
- F(s, s') = gamma * phi(s') - phi(s) (line 1206)
- Single `STAGE_POTENTIALS` dictionary used throughout (line 1239)
- Telescoping property verified via property tests

**Grade: EXCELLENT**

### Multi-Component Reward Balancing

Components are well-organized with clear purposes:
1. **Attribution (primary):** Counterfactual seed contribution
2. **PBRS bonus:** Stage progression incentive
3. **Compute rent:** Parameter bloat penalty
4. **Action shaping:** State machine compliance
5. **Terminal bonus:** Episode completion incentive
6. **Warning penalties:** Anti-gaming mechanisms

The component sum equals total reward (verified by property tests).

**Grade: GOOD**

### Reward Scale Appropriateness

Default weights produce rewards roughly in [-10, +10] range:
- Attribution: ~[-3, +3] (contribution_weight=1.0)
- PBRS: ~[-2, +2] (pbrs_weight=0.3)
- Rent: ~[0, -8] (rent_weight=0.5, max_rent=8.0)
- Terminal: ~[0, +23] (5% of 100 + 6 seeds * 3.0)

The terminal bonus can dominate, which is intentional for episode completion incentive.

**Grade: GOOD**

### Sparse vs Dense Reward Handling

Three modes implemented correctly:
- **SHAPED:** Dense with all components
- **SPARSE:** Terminal-only (tests credit assignment)
- **MINIMAL:** Sparse + early-cull penalty

The sparse reward mode correctly returns 0.0 for non-terminal timesteps.

**Grade: EXCELLENT**

### Reward Hacking Prevention

Multiple anti-gaming mechanisms:
1. **Attribution discount:** Sigmoid penalty for negative total_improvement
2. **Ratio penalty:** Catches high contribution with low improvement
3. **Ransomware detection:** High contribution + negative improvement = critical alert
4. **Legitimacy discount:** Rapid fossilization penalized
5. **CULL inversion:** Culling good seeds gives negative attribution

These mechanisms are well-designed and address real failure modes discovered in production (per code comments).

**Grade: EXCELLENT**

### Credit Assignment Implications

- 25-epoch episodes with gamma=0.995: gamma^25 = 0.88 (good credit assignment)
- PBRS provides dense signal while preserving optimal policy
- Terminal bonus scaled to compete with PBRS accumulation

**Grade: GOOD**

---

## Telemetry Usefulness

The `RewardComponentsTelemetry` dataclass provides:
- All reward components individually
- Context fields (action, epoch, stage)
- DRL Expert diagnostic fields (growth_ratio, progress_since_germination)
- `shaped_reward_ratio` property for detecting reward hacking

**Grade: EXCELLENT**

The `to_dict()` method uses explicit dict construction (not `asdict()`) for performance in the hot path.

---

## Test Coverage Assessment

Based on reviewed test files:

- **Property tests:** Mathematical invariants, semantic constraints, anti-gaming properties
- **Unit tests:** All major code paths covered
- **PBRS telescoping:** Verified via hypothesis-based property tests

**Grade: EXCELLENT**

---

## Recommendations Summary

### Must Fix
1. Add runtime gamma consistency check (C1)
2. Add division-by-zero guard for ratio penalty (C2)

### Should Fix
1. Document task-specific sigmoid steepness (H1)
2. Consider softening holding penalty for noisy environments (H2)
3. Fix or document sparse reward clamping behavior (H3)
4. Fix upstream SeedInfo construction for previous_epochs_in_stage (H4)

### Consider
1. Derive proxy_contribution_weight from contribution_weight (M2)
2. Cap cull shaping penalty (M3)
3. Remove duplicate intervention costs dictionary (M5)
4. Wire up reward hacking telemetry functions (L1)

---

## Conclusion

The `simic/rewards/` module demonstrates **sophisticated reward engineering** with strong theoretical grounding in PBRS and comprehensive anti-gaming mechanisms. The code quality is high, with excellent documentation of design rationale and extensive property-based testing.

The critical issues identified (gamma synchronization, division guard) are easily fixable. The high-priority issues reflect design decisions that may need empirical validation on diverse tasks.

**Approval Status:** APPROVED with recommended fixes for C1 and C2 before production use.
