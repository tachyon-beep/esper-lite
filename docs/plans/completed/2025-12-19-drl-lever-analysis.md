# Deep RL Analysis: Action Space Design for Morphogenetic Control

**Date:** 2025-12-19
**Status:** Analysis Complete
**Author:** Claude (DRL Specialist)

## Executive Summary

From a deep RL perspective, Esper presents a fascinating but challenging control problem: **morphogenetic control is a delayed-effect, high-dimensional POMDP where the agent must learn temporal correlations across 10-25 epoch horizons while the underlying environment (the host model) is non-stationary.**

This report analyzes what makes levers learnable, proposes design principles for expanding Tamiyo's action space, and provides specific guidance on the proposed "blend tempo" lever and other candidates.

---

## 1. What Makes a Lever Learnable?

### 1.1 The Learnability Criteria

A lever is learnable by PPO when:

1. **The effect is observable** - The agent must see signals that correlate with the lever's effect
2. **The credit assignment horizon is tractable** - PPO with GAE can handle delays of 5-15 timesteps reliably; beyond 25 timesteps requires architectural help (LSTM) or additional shaping
3. **The effect size exceeds noise** - The lever's impact must be distinguishable from environment stochasticity (training variance, gradient noise)
4. **The exploration cost is acceptable** - Trying different lever values shouldn't catastrophically hurt performance

### 1.2 Esper's Current Temporal Structure

Looking at the current system:

```
Episode Length: 25 epochs (timesteps)
LSTM Hidden: 128 dims
Observation: 74 dims base + telemetry (up to ~104)
Action Space: 4-head factored (slot, blueprint, blend_algo, lifecycle_op)
```

The LSTM provides crucial temporal context - without it, Tamiyo would struggle to connect early-episode decisions to late-episode outcomes. The forget gate bias of 1.0 is specifically tuned for this 25-epoch horizon.

### 1.3 Credit Assignment Analysis for Current Actions

| Action | Effect Delay | Observability | Learnability |
|--------|--------------|---------------|--------------|
| GERMINATE | ~0-1 epochs | Immediate (seed_telemetry appears) | High |
| WAIT | 0 epochs | Immediate (no state change) | High |
| CULL | 0-1 epochs | Immediate (seed removed) | High |
| FOSSILIZE | 0 epochs | Immediate (terminal state) | High |
| Blueprint choice | 3-15 epochs | Delayed (gradient health, contribution) | Medium |
| Blend algorithm | 5-10 epochs | Delayed (blending effectiveness) | Low-Medium |

**Key insight:** Current actions have relatively short feedback loops. Blueprint choice is the hardest to learn because its effect only manifests after TRAINING + BLENDING phases.

---

## 2. Analysis: The Proposed "Blend Tempo" Lever

### 2.1 Current Blending Mechanics

Blending is currently fixed at 5 epochs:
- Alpha ramps from 0.0 to 1.0 over `blending_steps` (fixed at 5)
- During blending, seed contribution becomes measurable via counterfactual
- Quality gates (G3, G5) check for stability before advancing

### 2.2 Proposed Lever: Blend Tempo

The proposal is to let Tamiyo choose blend speed:
- **Fast**: 3 epochs (alpha ramp 0 -> 1.0 quickly)
- **Normal**: 5 epochs (current default)
- **Slow**: 8-10 epochs (gradual integration)

### 2.3 DRL Analysis: Learnability of Blend Tempo

**Effect Delay**: 5-10 epochs

When Tamiyo chooses "slow blend" at TRAINING->BLENDING transition (epoch ~5-8), the consequences only become apparent:
- By epoch 13-18: Whether seed stabilized during slow ramp
- At terminal (epoch 25): Final accuracy, fossilization success

This is **borderline tractable for 25-epoch episodes**. The LSTM can theoretically learn this, but:

1. **Signal-to-noise ratio is poor**: The difference between fast/slow blending outcomes may be masked by:
   - Host training variance (loss fluctuation)
   - Seed quality variance (some seeds just fail regardless of tempo)
   - Blueprint-tempo interactions (some blueprints need slow blending, others don't)

2. **Sparse feedback**: You only get ~1 blend per episode (sometimes 0), so learning blend tempo requires many episodes

3. **Confounding factors**: Did the seed fail because of tempo, or because of blueprint choice, or because of cull timing?

### 2.4 Recommendation: Blend Tempo is Learnable with Telemetry Support

**Verdict: YES, add blend tempo, but with supporting telemetry**

Required telemetry additions:
1. **Alpha stability gradient** - Does loss variance increase/decrease as alpha ramps?
2. **Blend phase duration** - How many epochs in BLENDING before PROBATIONARY
3. **Alpha-contribution correlation** - Running correlation between alpha value and seed_contribution

Without these signals, the agent would need ~10x more episodes to learn tempo effects purely from terminal reward.

---

## 3. Action Space Design Principles

### 3.1 Discrete vs Continuous

**Recommendation: Always start discrete**

PPO handles discrete action spaces more robustly than continuous for several reasons:

1. **Exploration is easier** - Epsilon/entropy naturally explores all options
2. **Gradient estimation is simpler** - No reparameterization needed
3. **Masking is straightforward** - Invalid actions can be masked cleanly
4. **Interpretability** - "Chose FAST" is more debuggable than "chose 0.23"

For blend tempo, use discrete options (FAST/NORMAL/SLOW) rather than a continuous duration parameter.

### 3.2 Factored vs Combinatorial

The current factored design is excellent:

```
slot_idx x blueprint x blend_algo x lifecycle_op
```

This gives `3 x 13 x 3 x 4 = 468` combinations, but the agent only needs to learn ~23 separate "skills" (the sum of dimensions).

**Principle: Factor aggressively**

When adding blend tempo:
- **Option A (Recommended)**: Add as a 5th head - `tempo: FAST|NORMAL|SLOW`
- **Option B**: Merge with blend_algo - `blend: LINEAR_FAST|LINEAR_NORMAL|...`

Option A is better because:
- Tempo and blend_algo are conceptually independent
- Reduces interaction complexity for learning
- Allows tempo to be relevant even with LINEAR blending

### 3.3 Hierarchical Actions

Some proposed levers (quality gate thresholds, learning rate modulation) would create **meta-decisions** - decisions about how to make decisions.

**Warning: Hierarchical actions are hard for vanilla PPO**

The Options Framework (Sutton et al., 1999) would be ideal but requires:
- Temporally extended actions (options)
- Termination conditions
- Intra-option learning

For Esper, I recommend staying flat until the base system is working well. Hierarchical control can be Phase 4+ (Narset integration).

---

## 4. Telemetry Requirements for Learnability

### 4.1 The First Commandment in Practice

ROADMAP.md states: "Never add a capability without a corresponding sensor."

For any new lever, the telemetry checklist is:

| Lever | Required Sensor | Rationale |
|-------|-----------------|-----------|
| Blend tempo | `alpha_stability_gradient`, `blend_duration`, `alpha_contribution_corr` | Connect tempo choice to blending outcome |
| Gradient isolation strength | `isolation_leakage`, `host_grad_pollution` | Measure if isolation is actually working |
| Alpha ramp shape | `ramp_phase_loss_variance`, `ramp_derivative` | See how loss responds to alpha changes |
| Stage dwell times | `stage_efficiency` (improvement/epochs_in_stage) | Connect dwell choice to efficiency |
| LR modulation | `seed_lr_efficacy`, `lr_gradient_correlation` | Connect LR choice to seed learning rate |

### 4.2 The Temporal Credit Assignment Problem

For slow-effect levers, the agent faces a classic temporal credit assignment challenge:

```
"I chose slow blend at epoch 8.
At epoch 25, I got +3.5 reward.
Was that because of the slow blend, or
because I chose depthwise convolution, or
because I culled at epoch 15?"
```

**Solutions:**

1. **PBRS for Intermediate Progress** (already implemented)
   - Stage potentials provide incremental signal
   - Ng et al. (1999) guarantees this preserves optimal policy

2. **Counterfactual Attribution** (already implemented)
   - `seed_contribution` isolates seed impact from host drift
   - This is crucial for credit assignment

3. **Per-Lever Outcome Tracking** (recommended addition)
   - Track success rate per (lever_value, context) pair
   - Emit telemetry: "slow_blend with depthwise: 80% fossilize rate"
   - The agent doesn't see this directly, but humans can tune reward weights

4. **LSTM Memory** (already implemented)
   - The LSTM hidden state carries temporal context
   - Forget gate bias of 1.0 preserves information across the episode

---

## 5. Potential Levers: Detailed Analysis

### 5.1 Blend Tempo (Proposed) - **RECOMMEND**

**Discrete options**: FAST (3 epochs), NORMAL (5 epochs), SLOW (8 epochs)

**Learnability**: Medium-High
- Effect delay: 5-10 epochs (within LSTM horizon)
- Observable via: `seed_contribution` during BLENDING, `fossilize_rate`
- Exploration cost: Low (worst case: slightly slower/faster blending)

**Implementation notes**:
- Add 5th action head with 3 options
- Only valid during TRAINING stage (when choosing to advance to BLENDING)
- Mask FAST if seed is unstable (high loss variance in TRAINING)

### 5.2 Gradient Isolation Strength - **CONDITIONAL RECOMMEND**

**Discrete options**: FULL (current), PARTIAL (50% isolation), NONE

**Learnability**: Low-Medium
- Effect delay: Entire seed lifecycle (10-20 epochs)
- Observable via: New sensors needed (`isolation_leakage`, `host_stability`)
- Exploration cost: Medium-High (NONE can destabilize host)

**Recommendation**: Only add if there's evidence current full isolation is suboptimal. The "Governor" safety system would need to handle destabilization risks.

### 5.3 Alpha Ramp Shape - **DEFER**

**Discrete options**: LINEAR (current), SIGMOID, EXPONENTIAL

**Learnability**: Low
- Effect delay: 5-10 epochs during blending
- Observable via: Very subtle differences in loss curves
- Exploration cost: Low, but signal-to-noise ratio is poor

**Problem**: The difference between LINEAR and SIGMOID ramps is likely smaller than training variance. The agent would need 1000+ episodes to reliably distinguish outcomes.

**Recommendation**: Defer until we have evidence ramp shape matters. Could be a good candidate for Bayesian optimization rather than RL.

### 5.4 Stage Dwell Times - **RECOMMEND (Simplified)**

**Discrete options**: SHORT (min_epochs), NORMAL (1.5x min), LONG (2x min)

**Learnability**: Medium
- Effect delay: Direct (affects stage duration)
- Observable via: `epochs_in_stage`, `improvement_since_stage_start`
- Exploration cost: Low (LONG just takes more time)

**Implementation notes**:
- This is orthogonal to "advance" action - "advance" moves to next stage, dwell time sets the *minimum* before advance is available
- Alternative: Make this a continuous knob via heuristic (don't give to RL) based on seed quality metrics

### 5.5 Quality Gate Thresholds - **NOT RECOMMENDED**

**Problem**: This is a meta-decision that affects the meaning of other actions

If Tamiyo can loosen G5 threshold (fossilization gate), then "FOSSILIZE" becomes a different action. This violates the principle that action semantics should be stable.

**Alternative**: Let the Governor dynamically adjust thresholds based on episode history (outside RL loop).

### 5.6 Learning Rate Modulation for Seeds - **CONDITIONAL RECOMMEND**

**Discrete options**: LOW (0.5x), NORMAL (1x), HIGH (2x)

**Learnability**: Medium
- Effect delay: 3-8 epochs (fast feedback during TRAINING)
- Observable via: `seed_grad_norm`, `seed_loss_decrease_rate`
- Exploration cost: Medium (HIGH can cause divergence)

**Implementation notes**:
- Only valid during GERMINATE action (sets seed LR for its lifetime)
- Requires new telemetry: `seed_lr`, `seed_convergence_rate`
- Governor should detect and penalize divergence

---

## 6. What NOT to Give as Levers

### 6.1 Levers That Add Pure Noise

1. **Per-layer injection point selection** - The host architecture should define valid injection points. Letting RL choose *where* to inject adds combinatorial explosion without clear benefit.

2. **Optimizer hyperparameters (momentum, weight decay)** - These have very subtle effects that require 100+ epochs to distinguish. Leave to hyperparameter search.

3. **Batch size selection** - Affects training speed but has minimal interaction with morphogenetic decisions.

### 6.2 Levers That Violate Policy Invariance

1. **Reward weights** - If Tamiyo can adjust its own reward function, you get reward hacking.

2. **Gate pass/fail thresholds** - Changing these changes the meaning of actions.

3. **Episode length** - This would change the MDP structure mid-training.

### 6.3 Levers with Catastrophic Failure Modes

1. **Full host learning rate control** - Tamiyo could tank the host model accidentally.

2. **Seed deletion during FOSSILIZED** - Once fossilized, the seed is permanent. Allowing reversal creates instability.

3. **Multi-seed coupling** - Allowing Tamiyo to create dependencies between seeds would make credit assignment intractable.

### 6.4 The "More Control Hurts Learning" Pattern

**Symptom**: Adding a lever causes policy entropy to remain high forever, or policy collapses to ignoring the lever entirely.

**Diagnosis**: The lever's effect is not observable in the current telemetry, so exploration never converges.

**Rule of thumb**: If you add a lever and PPO's per-head entropy for that lever stays at max(entropy) after 100+ episodes, the lever is noise.

---

## 7. What Tamiyo Can Reasonably Learn

### 7.1 Given Current Sample Complexity

With PPO on LSTM and 25-epoch episodes:

**Reliably learnable (100-500 episodes)**:
- Blueprint selection for known injection points
- Cull timing (when to give up on a seed)
- Fossilize timing (when seed is ready)
- Germinate timing (when host is ready for augmentation)

**Learnable with effort (500-2000 episodes)**:
- Blueprint-slot interactions (which blueprints work at which positions)
- Blend algorithm selection (which blending strategy for which context)
- **Blend tempo** (with proper telemetry)
- Stage dwell preferences (with simplified discrete options)

**Hard to learn (2000+ episodes)**:
- Alpha ramp shape preferences
- Gradient isolation strength modulation
- Complex conditional policies (if X then Y else Z)

### 7.2 The Validation Evidence

The key validation mentioned in the context:
> "The agent independently discovered depthwise separable convolutions are more efficient than heavy conv blocks under time constraints"

This is exactly the kind of insight that emerges when:
1. The lever (blueprint choice) has clear effect (param count, gradient flow)
2. The telemetry supports credit assignment (seed_contribution, param_ratio)
3. The exploration space is tractable (13 blueprints, not 1000)

---

## 8. Design Principles Summary

### 8.1 The Three-Filter Test for New Levers

Before adding any lever, ask:

1. **Effect Size Filter**: Is the effect larger than training variance?
   - Measure: Run 20 episodes with lever fixed at each value, compare outcome distributions
   - Pass if: Distributions are statistically distinguishable (p < 0.05)

2. **Credit Assignment Filter**: Can the effect be attributed within the episode?
   - Measure: Is effect visible within 15 timesteps of the decision?
   - Pass if: Effect correlates with existing telemetry within horizon

3. **Exploration Cost Filter**: Is trying all lever values safe?
   - Measure: Does any lever value risk Governor intervention or catastrophic failure?
   - Pass if: All lever values are recoverable

### 8.2 Implementation Checklist

When adding a lever:

1. [ ] Add corresponding telemetry sensors
2. [ ] Add action masking for invalid lever values
3. [ ] Add to factored action space as new head (not merged)
4. [ ] Add property tests for PBRS telescoping with new lever
5. [ ] Run ablation: train with lever vs without lever
6. [ ] Monitor per-head entropy - should decrease over training
7. [ ] Add Governor guardrails for dangerous lever values

### 8.3 The Morphogenetic Control Manifold

The profound insight from the user:
> "We can't actually understand why she makes decisions, because she sees the telemetry in a lot more dimensionality than we do."

This is the fundamental tension: Tamiyo learns correlations in a 50-80 dimensional observation space, projecting to 468-dimensional action effects, across 25 timesteps. Humans can only visualize 2-3 dimensions at once.

**Implications**:
1. Don't add levers based on human intuition alone - validate with ablations
2. Trust emergent behavior (like discovering depthwise convs) over prescribed heuristics
3. The telemetry is for credit assignment, not for humans to "understand" decisions
4. Policy interpretability is a separate research problem - focus on outcomes

---

## 9. Concrete Recommendations

### 9.1 Immediate (Phase 2.5)

1. **Add blend tempo** as 5th action head with 3 options
2. Add supporting telemetry:
   - `alpha_stability_gradient`
   - `blend_duration_actual`
   - `blend_success_rate_by_tempo` (aggregate for human analysis)

### 9.2 Near-term (Phase 3)

1. **Add stage dwell preference** as discrete option (SHORT/NORMAL/LONG)
2. Add seed LR modulation (LOW/NORMAL/HIGH) with Governor guardrails

### 9.3 Longer-term (Phase 4+)

1. Gradient isolation strength (after evidence it matters)
2. Hierarchical control via Narset (regional managers)
3. Alpha ramp shape (via Bayesian optimization, not RL)

### 9.4 Never Add

1. Per-layer injection selection (use HostProtocol's injection_specs)
2. Quality gate threshold modification
3. Direct host model control beyond seed lifecycle
4. Reward weight adjustment

---

## Appendix: Credit Assignment Math

For a lever with effect delay `d` timesteps, PPO with GAE-lambda needs approximately:

```
Required episodes ≈ (d / (1 - lambda * gamma))^2 * (1 / effect_size)^2 * base_episodes
```

With `gamma=0.99`, `lambda=0.95`:
- `d=5`: ~2.4x base episodes
- `d=10`: ~5.7x base episodes
- `d=15`: ~12.8x base episodes
- `d=25`: ~51.5x base episodes

This explains why 5-epoch blend tempo is tractable (2.4x), but 25-epoch effects require thousands of episodes without additional shaping.

The LSTM mitigates this somewhat by maintaining hidden state, but the gradient flow still weakens over time steps. The forget gate bias of 1.0 helps but doesn't eliminate the problem.

---

*Report prepared from DRL perspective for Esper morphogenetic control system. Key references: Ng et al. 1999 (PBRS), Schulman et al. 2017 (PPO), Sutton et al. 1999 (Options Framework), Gers et al. 2000 (LSTM forget gate bias).*
