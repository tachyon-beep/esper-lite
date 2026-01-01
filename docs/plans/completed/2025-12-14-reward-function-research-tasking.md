# Research Tasking: Optimal Reward Function Design for Morphogenetic Seed Control

**Created:** 2025-12-14
**Type:** Research Deep Dive
**Status:** REVIEWED - Ready for Execution
**Updated:** 2025-12-14 (Incorporated DRL + PyTorch specialist reviews)
**Review Status:** 75% Ready per DRL specialist; FEASIBLE WITH MODIFICATIONS per PyTorch specialist

---

## 1. Executive Summary

This document provides a comprehensive tasking statement for researching the optimal reward function for the esper-lite morphogenetic neural network training system. The problem involves an RL agent that manages "seeds" (neural network modules) through a lifecycle, making decisions about when to germinate, train, blend, fossilize, or cull seeds to maximize host model performance.

### ⚠️ CRITICAL INSIGHT FROM PRIOR RESEARCH

The companion paper (002-seedinteractions.md) documents a **strongly negative result** about reward engineering:

> **Over-shaped rewards (~974 lines) FAILED.** The system converged to "do nothing" policies and degraded baseline performance. **Minimal rewards (~64 lines) WORKED.**

This fundamentally reframes the research question. We should NOT be asking "what's the best reward function" but rather:

### Research Questions (Revised)

1. **What is the MINIMAL reward function that remains learnable?**
2. **What trajectory-based signals (not snapshot rules) enable regime discrimination?**
3. **What NON-REWARD mechanisms (diagnostics, hygiene, risk management) should be added?**

### Three-Regime Taxonomy (From 002-seedinteractions.md)

The paper introduces a critical taxonomy that changes how we evaluate success:

| Regime | Description | Host-Only Competence | Correct Response |
|--------|-------------|---------------------|------------------|
| **Symbiosis** | Seed adds capability, host preserved | MUST retain | Fossilize good seeds |
| **Parasitism** | Seed displaces host, creates dependency | Degrades with no recovery | Cull early |
| **Metamorphosis** | Seed enables successor topology | MAY degrade (allowed) | Tolerate turbulence, then hygiene |

**Key insight**: "The moral status of displacement depends on the end state and the path to it, not on a snapshot."

### Why This Matters
The current reward function has evolved through iteration but may be **over-engineered**. The paper's historical evidence suggests:
- Dense reward shaping created "unlearnable landscapes"
- Anti-gaming mechanisms solved imagined problems while creating real ones
- The working solution was to **remove complexity**, not add it
- Non-reward mechanisms (hygiene, checkpointing, turbulence budgets) are the right place for safety

---

## 1A. Expert Review Findings (2025-12-14)

This section incorporates critical feedback from DRL and PyTorch specialist reviews.

### DRL Specialist Assessment: **75% Ready**

**Verdict:** Fundamentally sound research plan with gaps in statistical rigor and temporal credit assignment methodology.

#### Critical Issues Identified

| Issue | Severity | Resolution |
|-------|----------|------------|
| **PBRS germination bonus violates Ng et al.** | HIGH | Action-based shaping is NOT PBRS - must reformulate as state potential or remove |
| **"Different γ per component" is mathematically invalid** | HIGH | Violates value function foundations - use return decomposition instead |
| **No power analysis** | MEDIUM | Must calculate required sample sizes before experiments |
| **No multiple comparisons correction** | MEDIUM | 12 hypotheses → use Bonferroni (α=0.004) or FDR control |
| **GAE-lambda comment incorrect** | LOW | High λ = more variance/less bias (comment says opposite) |

#### Theoretical Clarifications

**On H12 ("Safety Outside Reward"):** This claim is **domain-specific**, not a general RL principle. It aligns with Constrained MDP literature for rare failure modes, but counter-arguments exist (Hadfield-Menell et al., 2017) for uncertain constraints. The document should acknowledge this nuance.

**On LSTM PPO:** Must use `n_epochs=1` for policy updates to avoid hidden state staleness. Multiple epochs cause policy_old/policy_new divergence.

### PyTorch Specialist Assessment: **FEASIBLE WITH MODIFICATIONS**

**Verdict:** Core hypotheses are testable with current architecture. Some mechanisms should be deferred.

#### Implementation Feasibility Matrix

| Mechanism | Feasibility | Effort | Risk | Decision |
|-----------|-------------|--------|------|----------|
| Minimal reward | HIGH | 1-2 days | LOW | ✅ Proceed |
| Interaction EMA/derivative | HIGH | 3-4 days | LOW | ✅ Proceed |
| Turbulence budget | HIGH | 2-3 days | LOW | ✅ Proceed |
| Full O(2^n) decomposition | MEDIUM | 1 week | MEDIUM | ⚠️ Conditional |
| Adaptive checkpointing | MEDIUM | 1 week | HIGH | ⏸️ Defer |
| Emrakul hygiene | LOW | 2+ weeks | HIGH | ❌ Separate project |

#### Computational Cost Warning

**Interaction Decomposition:** O(2^n) for n=3 = **4x compute increase** (~59,000 extra forward passes per episode batch). Recommendation: Use sampled decomposition (every 5 epochs) or trajectory features only.

#### torch.compile Compatibility

- **Interaction decomposition:** Requires `@torch.compiler.disable` decorator
- **Adaptive checkpointing:** `load_state_dict()` causes graph breaks - must isolate
- **Emrakul hygiene:** FX tracing incompatible with compiled models

### Consensus Recommendations

Both experts agree:
1. **Phase 0 is critical** - If minimal >> dense, restructure entire plan
2. **Start with trajectory features** (EMA, derivative) before full decomposition
3. **Defer Emrakul** until reward experiments prove it necessary
4. **Add statistical rigor** (power analysis, multiple comparisons correction)

---

## 2. Problem Specification

### 2.1 The Control Problem

The agent controls a **morphogenetic system** where:
- **Host Model**: A neural network (CNN or Transformer) being trained on a task
- **Seeds**: Modular components that can be grafted onto the host at injection points
- **Slots**: Positions where seeds can attach (early/mid/late in the network)
- **Goal**: Improve final host model accuracy through strategic seed management

### 2.2 State Space (Observations)

The agent observes a **50-60 dimensional** feature vector per timestep:

| Feature Group | Dimensions | Semantic Meaning |
|---------------|------------|------------------|
| **Timing** | 2 | epoch, global_step (normalized to training progress) |
| **Loss Metrics** | 3 | train_loss, val_loss, loss_delta |
| **Accuracy Metrics** | 3 | train_accuracy, val_accuracy, accuracy_delta |
| **Trends** | 3 | plateau_epochs, best_val_accuracy, best_val_loss |
| **Loss History** | 5 | Last 5 validation losses |
| **Accuracy History** | 5 | Last 5 validation accuracies |
| **Model State** | 2 | total_params, seed_utilization |
| **Per-Slot State (×3)** | 12 | is_active, stage, alpha, improvement (per slot) |
| **Blueprint One-Hot (×3)** | 15 | Blueprint ID per slot |
| **Optional Telemetry** | 10 | gradient_norm, gradient_health, vanishing/exploding flags |

**Key Observation Properties:**
- LSTM processes 25-epoch sequences (one epoch = one timestep)
- Observations are normalized using Welford's online algorithm
- Per-slot observability enables anticipatory multi-slot coordination

### 2.3 Action Space (Factored)

Four independent discrete heads:

| Head | Size | Values | Causally Relevant When |
|------|------|--------|------------------------|
| **op** | 4 | WAIT, GERMINATE, CULL, FOSSILIZE | Always |
| **slot** | 3 | EARLY, MID, LATE | op ∈ {GERMINATE, CULL, FOSSILIZE} |
| **blueprint** | 5 | NOOP, CONV_LIGHT, ATTENTION, NORM, DEPTHWISE | op = GERMINATE |
| **blend** | 3 | LINEAR, SIGMOID, GATED | op = GERMINATE |

**Action Masking:**
- NOOP blueprint always masked (can't train zero parameters)
- Ops masked based on slot states (e.g., can't FOSSILIZE without PROBATIONARY seed)
- Per-head advantage masking zeros out irrelevant heads

### 2.4 Seed Lifecycle (State Machine)

```
DORMANT ──GERMINATE──► GERMINATED ──auto──► TRAINING ──G2 gate──► BLENDING
                                                                      │
                                                              G3 gate─┘
                                                                      ▼
                                                              PROBATIONARY
                                                                      │
                              ┌─────────FOSSILIZE───────────────────┬─┘
                              ▼                                     │
                         FOSSILIZED                              CULL
                         (Terminal)                                 │
                                                                    ▼
                                                                 CULLED
                                                                    │
                         ┌──────────EMBARGO──────────────────────────┘
                         ▼
                    RESETTING ──► DORMANT (slot recycled)
```

**Quality Gates:**
| Gate | Transition | Key Criteria |
|------|------------|--------------|
| G0 | → GERMINATED | seed_id and blueprint_id present |
| G1 | → TRAINING | Stage is GERMINATED |
| G2 | → BLENDING | ≥0.5% improvement, ≤10 isolation violations, ≥3 epochs, ≥0.05 gradient ratio |
| G3 | → PROBATIONARY | ≥3 blending epochs, alpha ≥ 0.95 |
| G5 | → FOSSILIZED | counterfactual ≥ 1%, healthy |

### 2.5 Episode Structure

- **Episode Length**: Fixed at `max_epochs` (typically 25 for CIFAR, 50 for TinyStories)
- **Termination**: Time limit only (no early termination on success/failure)
- **Truncation Handling**: Bootstrap value estimation for critic targets
- **Vectorization**: Multiple environments run in parallel with independent seeds

---

## 3. Current Reward Function Analysis

### 3.1 Component Breakdown

The current reward is a sum of 7 components:

```
R_total = R_bounded_attribution     (primary signal)
        + R_blending_warning        (negative trajectory penalty)
        + R_probation_warning       (indecision penalty)
        + R_pbrs                    (stage progression shaping)
        + R_rent                    (compute cost)
        + R_action_shaping          (intervention costs + bonuses)
        + R_terminal                (episode completion bonus)
```

### 3.2 Primary Signal: Bounded Attribution

**Core Formula:**
```python
if seed_contribution < 0:
    R = seed_contribution * contribution_weight  # Direct penalty
else:
    R = attributed * attribution_discount * contribution_weight
```

Where:
- `seed_contribution = real_accuracy - counterfactual_accuracy` (α=0 baseline)
- `attribution_discount = sigmoid(-10 * total_improvement)` (ransomware defense)
- `attributed = min(seed_contribution, sqrt(progress * contribution))` (geometric mean cap)

**Properties:**
- Requires extra forward pass per validation (expensive)
- Provides true causal attribution (not correlation)
- Has ransomware defense (discount when total_improvement < 0)

### 3.3 PBRS (Potential-Based Reward Shaping)

**Formula:**
```python
R_pbrs = pbrs_weight * (gamma * phi(s') - phi(s))
```

**Stage Potentials:**
```python
STAGE_POTENTIALS = {
    DORMANT: 0.0,
    GERMINATED: 1.0,
    TRAINING: 2.0,
    BLENDING: 3.5,      # Largest jump (1.5) - value creation stage
    PROBATIONARY: 5.5,
    FOSSILIZED: 6.0,    # Small jump (0.5) - anti-farming
}
```

**Properties:**
- Preserves optimal policy (Ng et al., 1999)
- Provides dense guidance toward stage advancement
- Epoch progress bonus adds within-stage shaping

> ⚠️ **EXPERT REVIEW FINDING:** The current implementation includes a **PBRS bonus for GERMINATE action** (`rewards.py:602-607`). This is **NOT valid PBRS** in the Ng et al. sense - PBRS is state-based, not action-based. Adding rewards conditioned on actions can change the optimal policy, violating PBRS's core guarantee. **Resolution:** Either remove the action-based component or reformulate as pure state potential. Similarly, the `epoch_progress_bonus * epochs_in_stage` term risks violating PBRS if `epochs_in_stage` isn't part of the successor state potential.

### 3.4 Anti-Gaming Mechanisms

| Mechanism | Target Pattern | Implementation |
|-----------|---------------|----------------|
| **Attribution Discount** | Ransomware (high contribution, negative improvement) | `sigmoid(-10 * improvement)` |
| **Ratio Penalty** | Dependency gaming (contribution >> improvement) | `-0.1 * (ratio - 5) / 5` when ratio > 5 |
| **Terminal Asymmetry** | Bad fossilization (negative improvement seeds) | No terminal bonus unless improvement ≥ 1% |
| **Attribution Inversion** | CULL farming | Negate attribution for CULL actions |
| **Legitimacy Discount** | Fossilization farming | Scale bonus by `epochs_in_stage / MIN_PROBATION_EPOCHS` |

### 3.5 Reward Normalization

- Uses Welford's online algorithm for running mean/std
- Normalizes by std only (not subtracting mean)
- Clips to [-10, +10] for stability

---

## 3A. CRITICAL: Paper's Proposed Minimal Reward (From 002-seedinteractions.md)

The paper documents that the current dense reward evolved through **three phases**, with only the minimal version working:

### Phase 1: Loss-Delta Rewards (FAILED)
- Reward based on loss improvement
- Problem: Noisy, poorly aligned with seed-specific causality
- Result: Policies failed to improve or became unstable

### Phase 2: Contribution-Based with Extensive Protections (~974 lines, FAILED)
Mechanisms included:
- Attribution discounting based on total improvement trajectories
- Ratio penalties for "high contribution / low improvement"
- Escalating blending warnings
- Anti-farming penalties for WAIT-heavy behaviour
- Stage bonuses with anti-exploitation logic
- Explicit "ransomware signature" detection

**Why it failed:**
- "Penalty stacking drowned the signal" - multiple penalties fired simultaneously
- "Interacting terms produced unstable gradients"
- "Exploration became uniformly unsafe" - policy learned "don't touch it"
- "End-task performance regressed" - conservative policy worse than no growth

### Phase 3: Minimal Counterfactual-Grounded Reward (~64 lines, WORKED)

**The working formula:**
```
r_t = Σ(m_i)                           # Sum of marginal contributions
    - λ_rent · log(1 + params_t)       # Efficiency rent
    + λ_term · E_final                 # Terminal task score
    - λ_bud · u_t² · 1[GERMINATE]      # Budget pressure (germinate only)
```

Where:
- `m_i = E(M_t) - E(M_t \ s_i)` is marginal counterfactual contribution
- `params_t` is current parameter count (or growth ratio)
- `E_final` is terminal task score
- `u_t` is budget utilization fraction

**Why it works:**
1. **Preserves signal-to-noise ratio** - counterfactual gives seed-specific credit
2. **Avoids pre-emptive moralizing** - doesn't punish exploration for imagined failures
3. **Allows learning about metamorphosis** - agent learns when turbulence pays off

---

## 3B. CRITICAL: Non-Reward Mechanisms (From 002-seedinteractions.md)

The paper argues that safety mechanisms should be **outside the reward**, not inside it:

### Interaction Decomposition (Diagnostic, O(2^n) capped to n≤3)

For n≤3 concurrent seeds, evaluate ALL subset combinations:
- `acc_∅` (host-only)
- `acc_1, acc_2, acc_3` (standalones)
- `acc_12, acc_13, acc_23` (pairs)
- `acc_123` (full)

**Interaction term:**
```
I_12 = acc_12 - acc_∅ - g_1 - g_2
     = acc_12 - acc_1 - acc_2 + acc_∅
```

Where g_i = acc_i - acc_∅ (standalone gain)

**Interpretation:**
- I > 0: **Synergy** (whole > sum of parts)
- I < 0: **Interference** (whole < sum of parts)
- I ≈ 0: **Independence**

### Interaction Derivative (Key Discriminator)

The **interaction derivative over time** is the paper's proposed discriminator between parasitism and metamorphosis:

```
ΔĨ_t = Ĩ_t - Ĩ_{t-k}
```

Where Ĩ_t is smoothed interaction (EMA).

**Hypothesis (testable):**
- **Parasitism:** I_t < 0 AND ΔĨ_t ≤ 0 (sticky interference)
- **Metamorphosis:** I_t < 0 BUT ΔĨ_t > 0 (recovering turbulence)

### Turbulence Budget (Risk Management)

Bounded tolerance for negative interaction:
- Budget consumed slowly when recovery indicators positive (ΔĨ > 0)
- Budget consumed rapidly when turbulence sticky (ΔĨ ≤ 0)
- When budget exhausted → bias toward cull/rollback

### Adaptive Checkpointing / Rollback

When early-warning signals indicate rising risk:
1. Increase checkpoint frequency
2. If recovery fails within turbulence budget → rollback
3. Try alternative branch (cull, switch blueprint, change slot)

**Key insight:** "Converts topology changes from irreversible commitments into reversible exploration steps."

### Emrakul: Model Hygiene (NOT IMPLEMENTED)

A hygiene subsystem responsible for:
1. **Pruning unreachable structure** (graph-dead)
2. **Pruning low-traffic components** (connected but unused)
3. **Cleanup after topology changes**

**Why hygiene matters for metamorphosis:**
- Without hygiene, host displacement looks like parasitism
- With hygiene, dissolved host tissue is removed, revealing coherent successor
- Improves efficiency AND attribution clarity

### Shared Energy Budget (Stability-Coupled)

Dynamic allocation between growth (Tamiyo) and pruning (Emrakul):
```
E_emrakul = E · S_t
E_tamiyo = E · (1 - S_t)
```

Where S_t ∈ [0,1] is stability estimate.

- **Stable** → Emrakul gets budget (prune, consolidate)
- **Unstable** → Tamiyo gets budget (grow, correct)

---

## 4. Research Dimensions (REVISED)

### 4.1 Primary Signal Selection

**Research Questions:**
1. Is counterfactual validation worth the compute cost?
2. Could simpler proxy signals achieve similar policy quality?
3. What is the accuracy-cost tradeoff curve?

**Candidate Primary Signals:**

| Signal | Description | Pros | Cons |
|--------|-------------|------|------|
| **Counterfactual** (current) | α=0 baseline comparison | True causation | Extra forward pass |
| **Accuracy Delta** | `val_acc - prev_val_acc` | Cheap | Conflates host + seed |
| **Gradient-Based Attribution** | Seed gradient magnitude | Cheap | Indirect signal |
| **Blending Delta** | Accuracy change during BLENDING | Targeted | Stage-specific only |
| **Improvement Since Germination** | `val_acc - acc_at_germination` | Captures full journey | Correlation not causation |

**Experimental Design:**
- Ablation study: Replace counterfactual with each proxy
- Measure: Final accuracy, learning speed, compute cost
- Control: Same hyperparameters, same seeds

### 4.2 Reward Density Spectrum

**Research Questions:**
1. Does dense shaping accelerate learning or encourage reward hacking?
2. Can sparse rewards achieve comparable performance with more epochs?
3. Is there an optimal density level between sparse and current?

**Density Levels to Test:**

| Level | Description | Components |
|-------|-------------|------------|
| **Sparse** | Terminal only | `val_acc * scale` at episode end |
| **Minimal** | Primary + terminal | Bounded attribution + terminal bonus |
| **Moderate** | + stage transitions | Minimal + PBRS only (no warnings/rent) |
| **Dense** (current) | Full stack | All 7 components |

**Experimental Design:**
- Train with each density level for extended epochs
- Compare: Learning curves, final performance, policy behavior
- Analyze: Where does dense shaping help vs. hurt?

### 4.3 PBRS Calibration

**Research Questions:**
1. Are the stage potentials well-calibrated?
2. Does PBRS actually improve learning, or just change behavior?
3. Should potentials be learned rather than hand-tuned?

**Approaches to Test:**

| Approach | Description |
|----------|-------------|
| **Current** | Hand-tuned potentials (0, 1, 2, 3.5, 5.5, 6) |
| **Uniform** | Equal spacing (0, 1, 2, 3, 4, 5) |
| **Learned** | Neural potential function φ(s) trained end-to-end |
| **Outcome-Weighted** | Potentials based on empirical success rates per stage |
| **Ablated** | No PBRS (test if it's needed at all) |

**Experimental Design:**
- Compare PBRS variants on fixed task
- Measure: Learning speed, final performance, stage distribution
- Analyze: Do seeds reach FOSSILIZED more reliably with PBRS?

### 4.4 Multi-Objective Optimization

**Research Questions:**
1. How should we trade off accuracy vs. parameter efficiency?
2. Should there be explicit Pareto optimization?
3. Is logarithmic rent the right cost function?

**Current Approach:**
```python
R_rent = -min(rent_weight * log(1 + growth_ratio), max_rent)
```

**Alternative Approaches:**

| Approach | Description |
|----------|-------------|
| **Linear Rent** | `-weight * growth_ratio` |
| **Quadratic Rent** | `-weight * growth_ratio²` |
| **Budget Constraint** | Hard limit on total parameters, no shaping |
| **Pareto Reward** | Separate reward channels for accuracy and efficiency |
| **Scalarized MOO** | Learned weighting between objectives |

### 4.5 Temporal Credit Assignment

**Research Questions:**
1. Is GAE the right algorithm for this temporal structure?
2. Would return decomposition with auxiliary value heads improve learning?
3. Can we better handle the long delay from GERMINATE to terminal bonus?

**Unique Temporal Properties:**
- Seed lifecycles span 10-20 epochs
- GERMINATE decisions have very delayed consequences (15-20 epochs to FOSSILIZE)
- CULL decisions have immediate (often negative) consequences
- Terminal bonus creates long-horizon credit assignment challenge
- With γ=0.995, γ^25 ≈ 0.88 - reasonable horizon but may lose signal for early actions

**Approaches to Test:**

| Approach | Description |
|----------|-------------|
| **Single GAE** (current) | γ=0.995, λ=0.97 for all components |
| **Return Decomposition** | Decompose G_total = G_attr + G_pbrs + G_rent + G_terminal, train auxiliary value heads V_attr(s), V_pbrs(s), etc. Policy optimizes sum, but gains interpretability |
| **TD(1) for Terminal** | Use λ=1.0 (Monte Carlo) specifically for terminal bonus component |
| **Upgoing Policy Gradient** | UPG (Nota & Thomas, 2020) for variance reduction on sparse rewards |

> ⚠️ **EXPERT REVIEW FINDING:** The original proposal of "component-specific γ" (different discount factors per reward component) is **mathematically invalid**. The value function estimates V(s) = E[Σ γ^t r_t] - if different components have different gammas, you're no longer optimizing a coherent objective. **Use return decomposition with shared γ instead.**

> ⚠️ **EXPERT REVIEW FINDING:** Hindsight Experience Replay (HER) assumes goal-conditioned policies where goals can be relabeled post-hoc. This problem doesn't have that structure - you can't "relabel" a fossilization decision. Consider instead: **counterfactual-aware hindsight** (what would reward have been if we'd culled at epoch X) using existing counterfactual infrastructure.

### 4.6 Intrinsic Motivation

**Research Questions:**
1. Would curiosity about unexplored blueprint/slot combinations help?
2. Should there be exploration bonuses for novel states?
3. Can information-theoretic rewards accelerate discovery?

**Candidate Intrinsic Rewards:**

| Type | Description | Applicability |
|------|-------------|---------------|
| **Novelty** | Bonus for visiting new (slot, blueprint) combinations | Early training |
| **Curiosity (ICM)** | Prediction error as bonus | May help with sparse rewards |
| **Empowerment** | Maximize information gain about future states | Long-term planning |
| **State Coverage** | Bonus for reaching all lifecycle stages | Exploration guarantee |

### 4.7 Factored Reward Attribution

**Research Questions:**
1. Should rewards be decomposed per action head?
2. Can per-head value functions improve credit assignment?
3. Does causal masking help or hurt?

**Current Approach:**
- Single scalar reward
- Per-head advantage masking (zero out irrelevant heads)
- Shared value function

**Alternative Approaches:**

| Approach | Description |
|----------|-------------|
| **Per-Head Rewards** | Separate reward signal for slot/blueprint/blend/op |
| **Per-Head Critics** | 4 value functions, one per head |
| **Attention-Based Attribution** | Learn which head "caused" the reward |
| **Counterfactual Heads** | Compute counterfactual per-head to attribute |

### 4.8 ★ NEW: Interaction Decomposition and Trajectory Signals

**Research Questions (from 002-seedinteractions.md):**
1. Does interaction decomposition (O(2^n) for n≤3) provide actionable regime discrimination?
2. Is the interaction derivative ΔĨ_t a reliable discriminator between parasitism and metamorphosis?
3. Can trajectory-based signals replace snapshot-based gating rules?

**Approaches to Test:**

| Approach | Description | Cost |
|----------|-------------|------|
| **No Interaction Diagnostics** (current) | Only marginal contributions | O(n+1) |
| **Full Decomposition (n≤3)** | All subset evaluations | O(2^n) |
| **Sampled Decomposition** | Periodic diagnostic windows | Variable |
| **Trajectory Features Only** | ΔĨ_t, gradient health, traffic flow | Low |

**Key Hypotheses (testable from paper):**
- Parasitism: I_t < 0 with ΔĨ_t ≤ 0 (sticky interference)
- Metamorphosis: I_t < 0 with ΔĨ_t > 0 (recovering turbulence)

### 4.9 ★ NEW: Non-Reward Mechanisms

**Research Questions:**
1. Should safety mechanisms be implemented OUTSIDE reward (as paper suggests)?
2. Does Emrakul (hygiene) make metamorphosis viable?
3. Does turbulence budgeting reduce premature culls while maintaining safety?

**Mechanisms to Implement/Test:**

| Mechanism | Purpose | Status |
|-----------|---------|--------|
| **Turbulence Budget** | Bounded tolerance for negative interaction | NOT IMPLEMENTED |
| **Adaptive Checkpointing** | Reversible exploration | NOT IMPLEMENTED |
| **Emrakul (Hygiene)** | Prune detritus, consolidate topology | NOT IMPLEMENTED |
| **Shared Energy Budget** | Stability-coupled Tamiyo/Emrakul allocation | NOT IMPLEMENTED |

**Experimental Design:**
- Compare current system with each mechanism added individually
- Measure: Collapse rate, metamorphic wins, efficiency, attribution stability

### 4.10 ★ NEW: Regime-Aware Evaluation

**Research Questions:**
1. Should we evaluate symbiosis and metamorphosis differently?
2. Is "host-only retention" the right constraint for all regimes?
3. Should Pareto metrics include efficiency, not just accuracy?

**Proposed Scorecards (from paper):**

**Symbiosis Scorecard:**
- Final net improvement vs pre-germination
- Host retention (E(M(H,∅)) vs baseline)
- Marginal stability across ablation protocols
- Efficiency rent vs contribution

**Metamorphosis Scorecard:**
- Final Pareto position vs baselines
- Seed economy (gain per seed)
- Successor robustness (variance, sensitivity)
- Hygiene delta (detritus ratio, efficiency)
- Interaction recovery statistics

**Parasitism Red Flags:**
- Net regression + host collapse
- Sticky negative interaction
- No downstream consumption
- High protocol sensitivity
- Persistent instability

---

## 5. Experimental Framework

### 5.1 Evaluation Metrics

**Primary Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| **Final Accuracy** | Val accuracy at episode end | Maximize |
| **Accuracy Under Budget** | Val accuracy with ≤X% parameter growth | Maximize |
| **Learning Speed** | Episodes to reach Y% accuracy | Minimize |
| **Policy Stability** | Variance in final accuracy across seeds | Minimize |

**Secondary Metrics:**
| Metric | Description | Purpose |
|--------|-------------|---------|
| **Fossilization Rate** | % of germinates that fossilize | Measure commitment quality |
| **Cull Rate** | % of germinates that get culled | Measure selection quality |
| **Stage Distribution** | Time spent in each stage | Analyze behavior patterns |
| **Reward Decomposition** | Per-component contribution | Debug reward design |

### 5.2 Experimental Protocol

**Phase 1: Ablations (Understand Current System)**
1. Remove each reward component individually
2. Measure impact on all metrics
3. Identify which components are essential vs. optional

**Phase 2: Alternative Primaries (Test Signal Quality)**
1. Replace counterfactual with each proxy signal
2. Compare learning curves and final performance
3. Establish accuracy-cost tradeoff

**Phase 3: Density Sweep (Find Optimal Density)**
1. Test sparse → dense spectrum
2. Vary training duration to control for sample complexity
3. Identify knee point in density-performance curve

**Phase 4: PBRS Variants (Calibrate Shaping)**
1. Test potential function variants
2. Measure stage progression reliability
3. Determine if PBRS is net-positive

**Phase 5: Advanced Techniques (Push Performance)**
1. Test intrinsic motivation additions
2. Test multi-objective formulations
3. Test temporal credit assignment improvements

### 5.3 Control Variables

To ensure valid comparisons:
- **Fixed**: Network architecture, dataset, training hyperparameters
- **Fixed**: Episode length, number of environments
- **Varied**: Reward function components only
- **Reported**: Random seeds, confidence intervals

### 5.4 Computational Budget

| Experiment Type | Estimated Cost |
|-----------------|---------------|
| Single ablation | 5 seeds × 3 runs × 25 epochs ≈ 375 episodes |
| Full sweep | 6 variants × 5 seeds × 3 runs ≈ 90 training runs |
| Statistical power | Need ≥5 seeds per condition for significance |

### 5.5 Statistical Rigor Requirements (From Expert Review)

> ⚠️ **CRITICAL:** The original document lacked statistical rigor planning. The following requirements are mandatory.

#### Power Analysis

Before running experiments, determine required sample sizes:

1. **Effect size estimate:** For 5% accuracy difference with expected std=3%:
   - Cohen's d = 5/3 ≈ 1.67 (large effect)

2. **Required samples for large effects (d > 0.8):**
   - With power=0.8 and α=0.05: ~6 samples per condition
   - Recommendation: **10 seeds minimum** for moderate effects

3. **For H7 (minimal vs dense):**
   - If expecting large effect (paper claims substantial improvement): 6-10 seeds adequate
   - If expecting moderate effect: increase to 15-20 seeds

#### Multiple Comparisons Correction

With 15 hypotheses (H1-H15), naive p<0.05 testing yields ~0.75 expected false positives.

**Required corrections:**

| Method | Threshold | Use When |
|--------|-----------|----------|
| **Bonferroni** | α = 0.05/15 = 0.0033 | Conservative, independent tests |
| **FDR (Benjamini-Hochberg)** | Adaptive | Preferred for exploratory research |
| **Pre-registration** | Original α | If hypotheses registered before experiments |

**Recommendation:** Use FDR control for exploratory phases, Bonferroni for confirmatory tests.

#### Confound Control

Ablation studies must control for **reward scale changes**:

1. After removing a component, **re-normalize reward** to original mean/std
2. Report both raw and normalized results
3. If normalized ablation performs comparably, the component's effect was scale, not structure

---

## 6. Hypotheses to Test

### Original Hypotheses (May Need Revision Based on Paper)

### H1: Counterfactual Necessity
**Hypothesis:** Counterfactual validation provides at most 5% better final accuracy than the best proxy signal.
**Rationale:** The causal precision may not be worth the 2× compute cost.
**Test:** Compare counterfactual vs. accuracy_delta, gradient_attribution, improvement_since_germination.

### H2: Dense Reward Diminishing Returns ⚠️ LIKELY FALSE
**Hypothesis:** Moderate density (primary + PBRS + terminal) achieves ≥95% of dense reward performance.
**Rationale:** Anti-gaming mechanisms may be solving imagined problems.
**Test:** Compare density levels with equal total training time.
**Paper Evidence:** Over-shaped rewards (~974 lines) FAILED, minimal (~64 lines) WORKED. This hypothesis may be BACKWARDS - minimal may outperform moderate.

### H3: PBRS Acceleration
**Hypothesis:** PBRS reduces training time to target accuracy by ≥30%.
**Rationale:** Stage progression is correlated with success, shaping should help.
**Test:** Compare with/without PBRS, measure episodes to threshold accuracy.
**Paper Note:** The paper's minimal reward does NOT include PBRS - worth testing if PBRS is even necessary.

### H4: Sparse Reward Viability
**Hypothesis:** Sparse rewards can match dense reward performance given 3× more training.
**Rationale:** LSTM should be capable of long-horizon credit assignment.
**Test:** Train sparse reward agent for extended epochs, compare final accuracy.

### H5: Intrinsic Motivation Benefit
**Hypothesis:** Curiosity bonus improves final accuracy by ≥2% in early training.
**Rationale:** Blueprint exploration is valuable, explicit bonus should help.
**Test:** Add novelty bonus for unexplored (slot, blueprint) pairs.

### H6: Rent Function Shape
**Hypothesis:** Logarithmic rent is within 1% of optimal rent function.
**Rationale:** Current function may be arbitrary, not principled.
**Test:** Compare linear, quadratic, logarithmic, learned rent functions.

---

### ★ NEW Hypotheses (From 002-seedinteractions.md)

### H7: Minimal Reward Outperforms Dense ⭐ HIGH PRIORITY
**Hypothesis:** The paper's minimal reward (~64 lines) achieves HIGHER final accuracy than current dense reward (~974 lines).
**Rationale:** Paper documents that over-shaped rewards collapsed to "do nothing" policies. Minimal rewards preserve signal-to-noise ratio.
**Test:** Implement paper's minimal formula, compare learning curves and final performance.
**Formula:** `r = Σm_i - λ_rent·log(1+params) + λ_term·E_final - λ_bud·u²·1[GERMINATE]`

### H8: Interaction Derivative Discriminates Regimes ⭐ HIGH PRIORITY
**Hypothesis:** ΔĨ_t (interaction derivative) separates parasitism from metamorphosis with AUC > 0.7 in early training.
**Rationale:** Paper proposes that parasitism = "sticky negative" (ΔĨ ≤ 0), metamorphosis = "recovering turbulence" (ΔĨ > 0).
**Test:** Implement interaction decomposition for n≤3, log ΔĨ_t, correlate with final regime labels.

### H9: Trajectory Signals Beat Snapshot Gating
**Hypothesis:** Policies trained with trajectory features (ΔĨ_t, gradient health trends) outperform policies with snapshot features only.
**Rationale:** Paper argues "snapshot gating rules are brittle" because parasitism and early metamorphosis share same phenotype.
**Test:** Add trajectory-based features to observation space, compare policy quality.

### H10: Turbulence Budget Reduces Premature Culls
**Hypothesis:** Turbulence budgeting increases metamorphic wins by ≥20% without increasing collapse rate.
**Rationale:** Paper proposes that bounded tolerance for turbulence allows productive transitions that snapshot rules would cull.
**Test:** Implement turbulence budget mechanism, compare cull timing vs. final regime.

### H11: Hygiene (Emrakul) Makes Metamorphosis Legible
**Hypothesis:** Adding hygiene (prune low-traffic components) reduces detritus ratio by ≥50% and improves attribution stability.
**Rationale:** Paper argues that without hygiene, host displacement looks like parasitism; with hygiene, successor topology becomes clear.
**Test:** Implement Emrakul, measure detritus ratio, counterfactual protocol sensitivity, Pareto efficiency.

### H12: Safety Mechanisms Outside Reward Beat Safety Inside Reward
**Hypothesis:** System with minimal reward + turbulence budget + checkpointing outperforms system with dense reward (safety in reward).
**Rationale:** Paper's central thesis: "making the reward smarter by absorbing safety concerns tended to make it less learnable."
**Test:** Compare dense-reward-only vs. minimal-reward + mechanisms on collapse rate, final performance, policy behavior.

> ⚠️ **EXPERT REVIEW NOTE:** This claim is **domain-specific**, not a general RL principle. It aligns with Constrained MDP (CMDP) literature for rare failure modes (Altman, 1999; Achiam et al., 2017), but counter-arguments exist (Hadfield-Menell et al., 2017) when constraint specification is uncertain. This hypothesis is likely correct for THIS domain because safety concerns (parasitism, ransomware) are rare failure modes where penalties dominate when they fire.

---

### ★ NEW Hypotheses (From Expert Review)

### H13: Reward Scale Sensitivity ⭐ HIGH PRIORITY
**Hypothesis:** The minimal reward's success is due to lower variance (easier optimization), not its structure (correct signal).
**Rationale:** PPO is sensitive to reward scale. Minimal reward may implicitly have better signal-to-noise ratio.
**Test:** Normalize dense reward to have same mean/std as minimal reward, re-run comparison. If normalized dense performs comparably, the issue was scale not structure.

### H14: Exploration Collapse Mechanism
**Hypothesis:** Dense reward failures are caused by policy entropy collapse before value function convergence.
**Rationale:** When multiple penalties fire simultaneously (attribution_discount + ratio_penalty + blending_warning + probation_warning), expected return for any active policy becomes negative. Gradient updates push probability mass toward WAIT.
**Test:** Track entropy and value loss curves during training. Test whether adding explicit entropy bonus rescues dense reward.
**Predictions if true:**
1. Policy entropy collapses before performance plateaus
2. Advantage estimates for all non-WAIT actions become negative
3. Value estimates converge to "do nothing" return

### H15: LSTM Necessity
**Hypothesis:** LSTM is required for credit assignment in this domain - feedforward policy cannot learn.
**Rationale:** Seed lifecycles span 10-20 epochs; GERMINATE decisions have 15-20 epoch delays to outcomes.
**Test:** Compare LSTM vs feedforward (same hidden dim) on both minimal and dense rewards. If feedforward fails on both, confirms temporal memory is essential.

---

## 7. Deliverables (REVISED - Per Expert Review)

### Phase 0: Validate Paper's Minimal Reward Claim ⭐ CRITICAL GATE
**Goal:** Determine if research should focus on minimal reward refinement or dense reward debugging.

- [ ] Implement paper's minimal reward formula (`RewardMode.PAPER_MINIMAL`)
- [ ] Add H13 control: Normalize dense reward to same scale as minimal
- [ ] Run head-to-head: SHAPED vs PAPER_MINIMAL vs SHAPED_NORMALIZED
- [ ] Track entropy curves (H14) to detect exploration collapse
- [ ] **Decision gate:**
  - If minimal >> dense: Focus on minimal reward + non-reward mechanisms
  - If minimal ≈ normalized_dense: Focus on reward scale/normalization
  - If minimal << dense: Investigate why paper's claim doesn't replicate

### Phase 1: Trajectory Features + Ablations
**Goal:** Understand which signals are informative without expensive full decomposition.

- [ ] Implement `InteractionTracker` (EMA smoothing, derivative computation)
- [ ] Add trajectory features (Ĩ_t, ΔĨ_t) to telemetry logging
- [ ] Test H8: Does interaction derivative discriminate parasitism from metamorphosis?
- [ ] Single-component ablations **with reward normalization** (H13 control)
- [ ] Test H3: Is PBRS necessary? (Paper's minimal excludes it)

### Phase 2: Turbulence Budget + Statistical Validation
**Goal:** Test non-reward safety mechanisms with proper statistical rigor.

- [ ] Implement turbulence budget mechanism (~100 lines)
- [ ] Test H10: Does turbulence budget reduce premature culls?
- [ ] Test H12: Minimal + turbulence vs dense reward
- [ ] Apply multiple comparisons correction (Bonferroni α=0.004 or FDR)
- [ ] Verify sample sizes are adequate per power analysis

### Phase 3: Conditional Extensions (Only If Earlier Phases Warrant)

**If trajectory features show value (H8 confirmed):**
- [ ] Implement full O(2^n) interaction decomposition (sampled, every 5 epochs)
- [ ] Add `@torch.compiler.disable` decorator for decomposition function

**If turbulence budget insufficient (H12 not confirmed):**
- [ ] Implement adaptive checkpointing with CPU storage
- [ ] Test rollback mechanism at episode boundaries only

**If attribution instability persists:**
- [ ] Scope Emrakul hygiene as separate research project
- [ ] Define minimal viable hygiene (reachability-based only)

### ❌ DEFERRED (Per Expert Review)
These items are explicitly deferred until earlier phases demonstrate clear need:
- Full Emrakul hygiene subsystem (2+ weeks, torch.compile incompatible)
- Shared energy budget (depends on Emrakul)
- Adaptive checkpointing mid-episode (graph breaks, complex state management)

### Final Deliverables
- [ ] Research report with statistical analysis
- [ ] Recommended reward function with implementation
- [ ] Recommended non-reward mechanisms (if any)
- [ ] Documented decision rationale at each phase gate

---

## 8. Technical Prerequisites

### 8.1 Infrastructure Requirements
- [ ] Configurable reward function (all components toggleable)
- [ ] Experiment tracking (W&B or equivalent)
- [ ] Automated metric collection
- [ ] Statistical analysis pipeline

### 8.2 Code Changes Needed
- [ ] `RewardMode` enum extension for all density levels
- [ ] Proxy signal implementations (gradient-based, blending delta)
- [ ] Per-component reward logging
- [ ] Ablation configuration system

### 8.3 ★ Code Changes with Feasibility Assessment (From Expert Reviews)

#### ✅ LOW RISK - Implement in Phase 0/1

| Change | Effort | Lines | Notes |
|--------|--------|-------|-------|
| **Minimal reward** | 1-2 days | ~50-70 | Add `RewardMode.PAPER_MINIMAL`, pure arithmetic |
| **Reward normalization control** | 0.5 days | ~20 | For H13 confound control |
| **Interaction EMA tracker** | 1 day | ~30 | Track Ĩ_t, ΔĨ_t per slot |
| **Ablation toggle flags** | 0.5 days | ~20 | Enable/disable individual components |
| **Entropy curve logging** | 0.5 days | ~15 | For H14 exploration collapse detection |

**Minimal Reward Implementation:**
```python
def compute_paper_minimal_reward(
    marginal_contributions: list[float],
    total_params: int,
    budget_utilization: float,
    action: LifecycleOp,
    terminal_accuracy: float,
    is_terminal: bool,
    config: ContributionRewardConfig,
) -> float:
    """Paper's minimal reward (~50-70 lines)."""
    r = sum(marginal_contributions)
    r -= config.rent_weight * math.log(1 + total_params / config.param_budget)
    if is_terminal:
        r += config.terminal_acc_weight * terminal_accuracy
    if action == LifecycleOp.GERMINATE:
        r -= config.budget_pressure_weight * budget_utilization ** 2
    return r
```

**Interaction Tracker Implementation:**
```python
@dataclass
class InteractionTracker:
    ema_decay: float = 0.9
    history_window: int = 20
    interaction_history: dict[str, list[float]] = field(default_factory=dict)
    smoothed: dict[str, float] = field(default_factory=dict)

    def update(self, slot_id: str, interaction: float) -> tuple[float, float]:
        """Returns (smoothed_I, derivative_dI)."""
        if slot_id not in self.smoothed:
            self.smoothed[slot_id] = interaction
        else:
            self.smoothed[slot_id] = self.ema_decay * self.smoothed[slot_id] + (1 - self.ema_decay) * interaction
        # Track history for derivative...
        # Return smoothed and derivative
```

#### ⚠️ MEDIUM RISK - Implement in Phase 2 (Conditional)

| Change | Effort | Lines | Concern |
|--------|--------|-------|---------|
| **Turbulence budget** | 2-3 days | ~100 | Integrate into action masking |
| **Full O(2^n) decomposition** | 1 week | ~200 | 4x compute; use `@torch.compiler.disable` |

**Turbulence Budget:**
```python
@dataclass
class TurbulenceBudget:
    initial_budget: float = 1.0
    recovery_rate: float = 0.1   # When dI > 0
    sticky_rate: float = 0.3     # When dI <= 0
    budget: float = field(init=False)

    def consume(self, interaction: float, derivative: float) -> bool:
        """Returns True if budget exhausted (trigger cull bias)."""
        if interaction >= 0:
            return False
        rate = self.recovery_rate if derivative > 0 else self.sticky_rate
        self.budget -= rate * abs(interaction)
        return self.budget <= 0
```

#### ❌ HIGH RISK - DEFERRED (Per Expert Review)

| Change | Effort | Lines | Why Deferred |
|--------|--------|-------|--------------|
| **Adaptive checkpointing** | 1 week | ~150 | `load_state_dict()` breaks torch.compile graphs |
| **Emrakul hygiene** | 2+ weeks | ~500+ | FX tracing incompatible with compilation |
| **Shared energy budget** | N/A | ~100 | Depends on Emrakul |

**torch.compile Compatibility Notes:**
- Minimal reward: ✅ Pure arithmetic, no graph breaks
- Interaction decomposition: ⚠️ Wrap with `@torch.compiler.disable`
- Turbulence budget: ✅ State tracking outside forward pass
- Adaptive checkpointing: ❌ Must isolate from compiled regions
- Emrakul: ❌ Requires uncompile→modify→recompile cycle

### 8.4 Existing Support
- [x] RewardMode.SPARSE exists (terminal-only)
- [x] RewardMode.SHAPED exists (dense)
- [x] Reward telemetry captures all components
- [x] Counterfactual validation implemented

---

## 9. Risk Assessment

### High Risk
| Risk | Mitigation |
|------|------------|
| Sparse rewards fail completely | Use moderate density as fallback |
| Ablations show all components essential | Focus on calibration not removal |
| Proxy signals much worse than counterfactual | Keep counterfactual, focus on efficiency |

### Medium Risk
| Risk | Mitigation |
|------|------------|
| Results don't generalize across tasks | Test on both CNN (CIFAR) and Transformer (TinyStories) |
| Statistical noise masks effects | Use more seeds, longer runs |
| Reward hacking in new configurations | Monitor fossilization quality metrics |

### Low Risk
| Risk | Mitigation |
|------|------------|
| PBRS already optimal | Document finding, move on |
| Intrinsic motivation hurts | Easy to disable |

---

## 10. Appendix: Current Implementation Details

### A. Reward Component Weights (Defaults)
```python
contribution_weight = 1.0
pbrs_weight = 0.3
rent_weight = 0.5
terminal_acc_weight = 0.05
fossilize_terminal_scale = 3.0
```

### B. Key Constants
```python
MIN_CULL_AGE = 1
MIN_PROBATION_EPOCHS = 5
MIN_FOSSILIZE_CONTRIBUTION = 1.0  # percent
```

### C. File Locations
- Reward function: `src/esper/simic/rewards.py`
- Reward config: `src/esper/simic/config.py`
- Observations: `src/esper/simic/features.py`
- Signals: `src/esper/leyline/signals.py`
- Stages: `src/esper/leyline/stages.py`

---

## 11. Next Steps

1. **Review this tasking document** with stakeholders
2. **Prioritize research dimensions** based on expected impact
3. **Set up experiment infrastructure** (tracking, configs)
4. **Begin Phase 1 ablations** to establish baseline understanding
5. **Iterate on hypotheses** based on early findings

---

*End of Research Tasking Document*
