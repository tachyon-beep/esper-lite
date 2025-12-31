# Deep Reinforcement Learning Code Review: rewards.py

**File**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`
**Lines**: 1377
**Reviewer**: DRL Specialist
**Date**: 2025-12-17

---

## Executive Summary

The `rewards.py` module implements a sophisticated reward engineering system for Esper's morphogenetic neural network growth. The reward design demonstrates significant DRL expertise, particularly in potential-based reward shaping (PBRS), anti-gaming mechanisms, and multi-objective balancing.

**Overall Assessment**: The reward function is well-designed for the domain but has several areas requiring attention for long-term training stability and credit assignment efficiency.

**Strengths**:
- Rigorous PBRS implementation with correct telescoping property (Ng et al., 1999)
- Comprehensive anti-reward-hacking mechanisms ("ransomware" detection)
- Proper separation of dense (SHAPED), sparse, and minimal reward modes for experimentation
- Strong telemetry support for reward component analysis

**Areas of Concern**:
- Reward magnitude imbalance risks value function estimation issues
- Sparse reward mode may suffer from credit assignment challenges over 25-step episodes
- Some reward components exhibit potential for unintended interactions

---

## Critical Issues

### C1: Unbounded Reward Accumulation in Terminal Bonus

**Location**: Lines 658-674 (`compute_contribution_reward`)

**Problem**: The terminal bonus scales linearly with `num_contributing_fossilized` without an upper bound:

```python
# Lines 666-668
fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
terminal_bonus += fossilize_terminal_bonus
reward += terminal_bonus
```

With `fossilize_terminal_scale = 3.0` (default), 10 fossilized seeds yields +30.0 terminal bonus. Combined with `val_acc * 0.05` (e.g., +4.0 for 80% accuracy), terminal rewards can reach +34.0 or higher.

**Impact**:
- Value function targets will have high variance between episodes with many vs few fossilized seeds
- PPO advantage estimates become noisy when value targets span 0-40+
- May cause critic underfitting and policy oscillation

**Recommendation**: Cap the terminal fossilization bonus or use diminishing returns:
```python
# Option 1: Hard cap
fossilize_terminal_bonus = min(num_contributing_fossilized * config.fossilize_terminal_scale, 15.0)

# Option 2: Diminishing returns (sqrt scaling)
fossilize_terminal_bonus = math.sqrt(num_contributing_fossilized) * config.fossilize_terminal_scale
```

**Severity**: Critical - Can destabilize value function learning in multi-slot scenarios.

---

### C2: Reward Magnitude Imbalance Between Components

**Location**: Lines 447-602 (attribution, warnings, PBRS sections)

**Problem**: Per-step reward components have vastly different magnitudes:

| Component | Typical Magnitude | Notes |
|-----------|------------------|-------|
| `bounded_attribution` | 0.0 to 7.5+ | Can reach high values with good seeds |
| `pbrs_bonus` | -0.5 to +1.5 | Per STAGE_POTENTIALS deltas |
| `compute_rent` | -0.05 to -0.5 | Logarithmic, usually small |
| `holding_warning` | -10.0 to 0.0 | Exponential escalation |
| `blending_warning` | -0.4 to 0.0 | Capped escalation |
| `terminal_bonus` | 0.0 to 34.0+ | Only at max_epochs |

The 100x difference between `holding_warning` (-10.0) and `compute_rent` (-0.05) means the policy will heavily optimize to avoid holding penalties while essentially ignoring compute efficiency.

**Impact**:
- Multi-objective reward balancing is implicitly determined by magnitude, not deliberate design
- Rent signal effectively becomes noise compared to larger penalties/bonuses
- Policy may ignore parameter efficiency entirely

**Recommendation**: Normalize reward components to similar scales or use explicit multi-objective weighting:
```python
# Component-wise scaling for balanced multi-objective optimization
reward = (
    attribution_scale * bounded_attribution +  # e.g., 1.0
    pbrs_scale * pbrs_bonus +                   # e.g., 0.5
    rent_scale * compute_rent +                 # e.g., 2.0 to amplify
    warning_scale * (blending_warning + holding_warning) +  # e.g., 0.3
    terminal_scale * terminal_bonus             # e.g., 0.5
)
```

**Severity**: Critical - Reward imbalance prevents meaningful multi-objective optimization.

---

### C3: Sparse Reward Credit Assignment Over 25 Steps

**Location**: Lines 682-724 (`compute_sparse_reward`)

**Problem**: The sparse reward mode returns 0.0 for all non-terminal timesteps:

```python
# Lines 713-714
if epoch != max_epochs:
    return 0.0
```

With 25-epoch episodes and gamma=0.995, credit must propagate 25 steps. While the LSTM should theoretically handle this, sparse rewards are notoriously difficult for policy gradient methods.

**Analysis**:
- GAE with lambda=0.97 helps, but advantage estimates for early actions rely entirely on value function accuracy
- Value function must learn to predict terminal outcome from early states - a challenging regression task
- Initial policy updates will have extremely high variance

**Observed in Tests**: The test suite (lines 712-724) clamps sparse rewards to [-1, 1], but this aggressive clamping may discard critical signal magnitude information.

**Recommendation**: Consider semi-sparse alternatives if pure sparse fails:
1. Add a small dense signal (e.g., 0.1 * acc_delta) to reduce variance
2. Increase `sparse_reward_scale` beyond 2.0-3.0 if learning fails
3. Use longer LSTM windows or transformer memory for better credit assignment
4. Document expected training time increase for sparse mode (10-100x more samples typical)

**Severity**: Critical for SPARSE mode - May be unlearnable without modifications.

---

## High-Priority Issues

### H1: Holding Warning Exponential Escalation

**Location**: Lines 594-600

**Problem**: The holding penalty escalates exponentially:

```python
# Lines 596-599
epochs_waiting = seed_info.epochs_in_stage - 1
holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
# Cap at -10.0 (clip boundary) to avoid extreme penalties
holding_warning = max(holding_warning, -10.0)
```

Schedule: epoch 2 -> -1.0, epoch 3 -> -3.0, epoch 4 -> -9.0, epoch 5+ -> -10.0

**Impact**:
- Single timestep penalties can dominate entire episode returns
- Creates sharp cliffs in the reward landscape
- Policy may learn to avoid HOLDING entirely rather than making informed decisions

**The guard at line 588 (`if bounded_attribution > 0`) partially mitigates this**, but legitimate seeds being "farmed" still face severe penalties.

**Recommendation**: Consider softer escalation or restructure as opportunity cost:
```python
# Linear escalation with max
holding_warning = -min(epochs_waiting * 1.5, 8.0)

# Or: Use discounted foregone returns instead of penalties
# This measures "what you gave up by not deciding" more directly
```

**Severity**: High - Sharp penalty cliffs create difficult optimization landscapes.

---

### H2: Attribution Discount Sigmoid May Be Too Aggressive

**Location**: Lines 465-466

**Problem**: The sigmoid discount for negative `total_improvement` uses a steep coefficient:

```python
# Lines 465-466
if total_imp < 0:
    attribution_discount = 1.0 / (1.0 + math.exp(-config.attribution_sigmoid_steepness * total_imp))
```

With `attribution_sigmoid_steepness = 10.0` (default), the discount drops rapidly:
- At total_imp = -0.2: discount = 0.12
- At total_imp = -0.5: discount = 0.007
- At total_imp = -1.0: discount = 0.00005

**Impact**:
- Seeds with minor negative improvement (-0.1 to -0.3) are heavily penalized despite potentially being recoverable
- Early-stage fluctuations (normal during blending) trigger near-zero attribution
- Policy receives almost no gradient signal for "borderline" seeds

**Recommendation**: Consider a two-phase approach:
1. Gentle discount for minor regression (-0.5 to 0): linear or mild sigmoid
2. Steep discount for severe regression (< -0.5): current aggressive sigmoid

```python
if total_imp < -0.5:
    attribution_discount = 0.1  # Severe regression
elif total_imp < 0:
    attribution_discount = 1.0 + total_imp  # Linear: -0.3 -> 0.7
else:
    attribution_discount = 1.0
```

**Severity**: High - May prevent learning recovery strategies for struggling seeds.

---

### H3: Geometric Mean in Attribution May Obscure Signal

**Location**: Lines 501-507

**Problem**: The "high causal" case uses geometric mean:

```python
# Lines 501-507
if seed_contribution >= progress:
    # High causal, low progress: timing mismatch, seed is valuable
    # Geometric mean recovers signal: sqrt(5% * 47%) = 15.3% vs min = 5%
    attributed = math.sqrt(progress * seed_contribution)
else:
    # Low causal, high progress: host did the work
    # Cap at actual contribution to prevent free-riding
    attributed = seed_contribution
```

**Analysis**:
The geometric mean is mathematically elegant but creates a discontinuity at the boundary:
- If contribution = progress = 5.0: geometric mean = 5.0
- If contribution = 5.01, progress = 5.0: geometric mean = 5.005
- If contribution = 4.99, progress = 5.0: attributed = 4.99

This discontinuity is minor but may cause gradient noise near the boundary.

**More significantly**: The rationale (timing mismatch) may not always hold. A seed with 47% contribution and 5% progress could also indicate reward hacking (the ransomware pattern) rather than timing mismatch.

**Recommendation**: Add an additional check for suspiciously high contribution/progress ratios:
```python
if seed_contribution >= progress:
    if seed_contribution > 10 * progress and progress < 1.0:
        # Suspiciously high ratio with minimal progress - potential hacking
        attributed = progress  # Conservative: only credit actual progress
    else:
        attributed = math.sqrt(progress * seed_contribution)
```

**Severity**: High - May inadvertently reward certain hacking patterns.

---

### H4: CULL Attribution Inversion Logic

**Location**: Lines 543-544

**Problem**: The CULL action inverts attribution:

```python
# Lines 543-544
if action == LifecycleOp.CULL:
    bounded_attribution = -bounded_attribution
```

**Analysis**:
This is correct in principle (culling a good seed = bad, culling a bad seed = good). However, the inversion happens after all the anti-hacking checks, which were designed for positive attribution scenarios.

**Edge case**: If a ransomware seed has high positive `seed_contribution` but negative `total_improvement`, the attribution discount reduces `bounded_attribution` to near-zero. After inversion, it remains near-zero rather than becoming a reward for culling a bad seed.

**Impact**: The policy receives weak signal for correctly culling ransomware seeds because the ransomware defenses (attribution_discount) fire before the CULL inversion.

**Recommendation**: Consider computing CULL rewards from a different perspective:
```python
if action == LifecycleOp.CULL:
    if seed_contribution is not None:
        if seed_contribution < 0:
            # Good: Pruned harmful seed
            bounded_attribution = abs(seed_contribution) * config.contribution_weight
        elif seed_info.total_improvement < -0.2:
            # Good: Pruned ransomware seed (high contribution but hurt performance)
            bounded_attribution = 0.5  # Fixed bonus for correct detection
        else:
            # Bad: Pruned beneficial seed
            bounded_attribution = -seed_contribution * config.contribution_weight
```

**Severity**: High - Weakens learning signal for ransomware removal.

---

## Medium-Priority Issues

### M1: PBRS Epoch Progress Bonus Creates Non-Telescoping Reward

**Location**: Lines 954-981 (`_contribution_pbrs_bonus`)

**Problem**: The epoch progress bonus within stages breaks strict PBRS telescoping:

```python
# Lines 954-957
phi_current = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
phi_current += min(
    seed_info.epochs_in_stage * config.epoch_progress_bonus,
    config.max_progress_bonus,
)
```

While stage transitions telescope correctly, the within-stage epoch increments create additional reward that doesn't perfectly cancel.

**Analysis**: The tests acknowledge this (line 280 in test_pbrs_properties.py):
```python
# We use a relaxed tolerance to catch major breaks while allowing known issues
tolerance = 2.0 + 1.0 * T  # Relaxed tolerance for known implementation limitations
```

**Impact**: This is a documented deviation from strict PBRS. The practical impact is limited since:
1. Within-stage bonuses are small (0.3 per epoch, capped at 2.0)
2. The main PBRS benefit (stage transition incentives) is preserved

**Recommendation**: Document this deviation prominently in the STAGE_POTENTIALS docstring and consider whether within-stage bonuses are necessary given the epoch_progress_bonus component.

**Severity**: Medium - Violates theoretical PBRS guarantee but impact is bounded.

---

### M2: Loss-Primary Reward Function Less Tested

**Location**: Lines 1272-1320 (`compute_loss_reward`)

**Problem**: The loss-primary reward function appears less battle-tested than the contribution-primary version:

```python
# Lines 1290-1294
normalized_delta = loss_delta / config.typical_loss_delta_std
clipped = max(-config.max_loss_delta, min(normalized_delta, config.max_loss_delta))
if clipped > 0:
    clipped *= config.regression_penalty_scale
reward += (-clipped) * config.loss_delta_weight
```

**Observations**:
1. No anti-hacking mechanisms equivalent to ransomware detection
2. Asymmetric clipping (regression_penalty_scale = 0.5) may be insufficient for loss-based exploitation
3. Task-specific constants (baseline_loss, target_loss) require manual tuning per domain

**Impact**: Loss-primary mode may be more susceptible to reward hacking if/when used.

**Recommendation**: If loss-primary mode is experimental, mark it clearly. If intended for production, port the anti-hacking mechanisms from contribution-primary.

**Severity**: Medium - Experimental feature with potential vulnerabilities.

---

### M3: RewardNormalizer Not Applied to All Components

**Location**: Vectorized training (vectorized.py:555)

**Problem**: The `RewardNormalizer` is created but its application to rewards may not capture the full reward distribution shift over training.

**Analysis**: Reward normalization divides by running std (correctly not subtracting mean for critic stability), but the reward distribution changes significantly as:
1. Seeds progress through lifecycle stages (different attribution magnitudes)
2. Terminal bonuses appear (large positive spikes)
3. Holding warnings escalate (large negative spikes)

**Impact**: Early training sees small rewards; late training may see 10x larger rewards. Normalization statistics may lag behind, causing value function instability during phase transitions.

**Recommendation**: Consider:
1. Warm-starting normalizer statistics from expected reward ranges
2. Using separate normalization for per-step vs terminal rewards
3. Monitoring `reward_normalizer.m2` growth over training for distribution shift detection

**Severity**: Medium - May cause training instability during regime changes.

---

### M4: Intervention Costs Are Too Small

**Location**: Lines 1327-1341

**Problem**: Intervention costs are negligible compared to other reward components:

```python
INTERVENTION_COSTS: dict[LifecycleOp, float] = {
    LifecycleOp.WAIT: 0.0,
    LifecycleOp.GERMINATE: -0.02,
    LifecycleOp.FOSSILIZE: -0.01,
    LifecycleOp.CULL: -0.005,
}
```

With attribution bonuses potentially +7.5 and terminal bonuses +30, a -0.02 germinate cost is negligible (0.27%).

**Impact**: The costs provide essentially no friction against unnecessary interventions. They exist but have no practical effect.

**Recommendation**: Either:
1. Increase costs to meaningful levels (e.g., -0.5 to -1.0) if friction is desired
2. Remove them entirely to reduce reward function complexity
3. Document that they exist for future tuning and are currently inactive

**Severity**: Medium - Dead code adding complexity without benefit.

---

### M5: Unwired Telemetry Functions

**Location**: Lines 1076-1077 (TODO comment)

**Problem**: Two telemetry functions are defined but never called:

```python
# TODO: [UNWIRED TELEMETRY] - Call _check_reward_hacking() and _check_ransomware_signature()
# from compute_contribution_reward() when attribution is computed. See telemetry-phase3.md Task 5.
```

**Impact**: Valuable diagnostic telemetry for reward hacking detection is implemented but not integrated into the main reward computation path.

**Recommendation**: Either wire these into `compute_contribution_reward` or archive them if superseded by the inline detection logic.

**Severity**: Medium - Technical debt reducing observability.

---

## Low-Priority Suggestions

### L1: Consider Per-Episode Reward Normalization

**Location**: Not currently implemented

**Suggestion**: The current approach normalizes rewards globally. Consider episode-level normalization or return normalization (normalizing the computed returns rather than raw rewards) for better training stability.

This is particularly relevant for morphogenetic training where early episodes (pre-fossilization) have fundamentally different reward distributions than late episodes (post-fossilization).

---

### L2: Document Expected Reward Ranges

**Location**: Module docstring or config comments

**Suggestion**: Add documentation of expected reward ranges per component and per training phase. This would help future maintainers tune weights and diagnose anomalies.

```python
# Expected per-step reward ranges:
#   Early training (no fossilized seeds): [-2.0, +5.0]
#   Mid training (active seeds): [-10.0, +10.0]
#   Terminal (fossilized seeds): [+3.0, +40.0]
```

---

### L3: Type Annotations for Reward Config Fields

**Location**: Lines 139-228 (`ContributionRewardConfig`)

**Suggestion**: Consider adding explicit value constraints via `__post_init__` validation:

```python
def __post_init__(self):
    assert 0.0 < self.gamma <= 1.0, "gamma must be in (0, 1]"
    assert self.contribution_weight >= 0.0, "contribution_weight must be non-negative"
    # etc.
```

This would catch configuration errors early rather than during training.

---

### L4: Consolidate Default Config Singleton

**Location**: Lines 352-353

**Issue**: Both `_DEFAULT_CONTRIBUTION_CONFIG` (module-level) and inline `ContributionRewardConfig()` construction are used:

```python
# Line 353
_DEFAULT_CONTRIBUTION_CONFIG = ContributionRewardConfig()

# Line 821 (compute_reward)
if config is None:
    config = ContributionRewardConfig()  # Creates new instance
```

**Suggestion**: Consistently use the singleton to avoid unnecessary allocations in hot paths.

---

### L5: Progress Constant Could Use Leyline

**Location**: Line 1243

**Issue**: Magic numbers for progress bonus:

```python
progress_bonus = min(epochs_in_stage * 0.3, 2.0)
```

These match `ContributionRewardConfig` defaults but are hardcoded.

**Suggestion**: Import from config or leyline for single source of truth.

---

## Summary of Recommendations by Priority

| ID | Issue | Priority | Estimated Impact |
|----|-------|----------|------------------|
| C1 | Unbounded terminal bonus | Critical | Value function instability |
| C2 | Reward magnitude imbalance | Critical | Multi-objective failure |
| C3 | Sparse mode credit assignment | Critical | Mode may be unlearnable |
| H1 | Exponential holding penalty | High | Harsh optimization landscape |
| H2 | Aggressive attribution discount | High | Lost learning signal |
| H3 | Geometric mean edge cases | High | Potential hacking vector |
| H4 | CULL attribution logic | High | Weak ransomware removal signal |
| M1 | Non-telescoping epoch bonus | Medium | PBRS guarantee violation |
| M2 | Loss-primary less tested | Medium | Potential vulnerabilities |
| M3 | Reward normalization lag | Medium | Training instability |
| M4 | Negligible intervention costs | Medium | Dead complexity |
| M5 | Unwired telemetry | Medium | Reduced observability |

---

## Conclusion

The reward engineering in `rewards.py` demonstrates sophisticated understanding of RL reward design challenges, particularly around reward hacking and multi-stage credit assignment. The PBRS implementation is theoretically grounded and well-tested.

The main concerns center on reward magnitude balancing and the practical challenges of the sparse reward mode. For production use, I recommend:

1. **Immediate**: Cap terminal bonuses (C1)
2. **Short-term**: Normalize reward component magnitudes (C2)
3. **Medium-term**: Reconsider sigmoid steepness and holding escalation (H1, H2)
4. **Long-term**: Comprehensive hyperparameter sensitivity analysis across reward weights

The codebase shows evidence of iterative refinement (DRL Expert review comments, multiple configuration options), suggesting an empirically-driven development process. Continue this approach with ablation studies on the identified issues.

---

*Report generated by DRL Specialist review on 2025-12-17*
