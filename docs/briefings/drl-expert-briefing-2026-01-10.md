# DRL Expert Briefing Pack: PPO Training Analysis
## Date: 2026-01-10 | Run: `telemetry_2026-01-10_071921`

**Objective:** Peer review of training diagnostics and proposed interventions for Esper's PPO-controlled morphogenetic neural network system.

---

## Executive Summary

A DeepSeek model analyzed a screenshot of training telemetry and provided recommendations. Initial review suggested their analysis was fundamentally wrong (misinterpreting KL≈0 as "good"). However, deeper investigation into Esper's action masking system revealed a more nuanced picture: the 88% WAIT rate is largely **expected behavior** due to sequential seed development masking, but there are genuine pathologies in the sparse action heads (blueprint, slot, tempo) that the entropy floor system isn't fixing.

**Key question for peer review:** Is the entropy floor penalty failing due to (a) insufficient coefficient, (b) gradient flow death at low entropy, (c) causal masking zeroing gradients, or (d) something else entirely?

---

## Part 1: System Context

### 1.1 What is Esper?

Esper is a morphogenetic neural network framework where:
- A **host network** (e.g., ResNet for CIFAR-10) is trained normally
- **Seed modules** can be germinated, trained in isolation, blended into the host, and fossilized
- A **PPO agent (Tamiyo/Simic)** controls the seed lifecycle decisions
- The goal: learn when and how to grow the network for optimal accuracy/efficiency tradeoff

### 1.2 The Factored Action Space

Tamiyo uses a factored action space with **6 decision heads**:

| Head | Options | Purpose | Activation Frequency |
|------|---------|---------|---------------------|
| `op` | WAIT, GERMINATE, ADVANCE, FOSSILIZE, PRUNE, SET_ALPHA_TARGET | What action to take | 100% of steps |
| `slot` | r0c0, r0c1, r0c2 (3 slots) | Which slot to target | ~60% (irrelevant for WAIT) |
| `blueprint` | conv_heavy, conv_light, bottleneck, etc. (7 types) | What module type to create | ~5% (GERMINATE only) |
| `style` | SIGMOID_ADD, LINEAR, etc. | Blending algorithm | ~7% (GERMINATE + SET_ALPHA_TARGET) |
| `tempo` | 3, 5, 7, 10 epochs | Blending duration | ~5% (GERMINATE only) |
| `alpha_target` | 0.25, 0.5, 0.75, 1.0 | Target blend level | ~7% (GERMINATE + SET_ALPHA_TARGET) |

### 1.3 The Seed Lifecycle State Machine

```
DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► HOLDING ──► FOSSILIZED
                │              │            │            │
                ▼              ▼            ▼            ▼
             PRUNED ◄───────────────────────────────────────
                │
                ▼
           EMBARGOED ──► RESETTING ──► DORMANT (slot recycled)
```

**Key constraint (D3 Sequential Development):** GERMINATE is blocked when ANY seed is in:
- GERMINATED or TRAINING (active development)
- PRUNED, EMBARGOED, or RESETTING (cleanup)

This means a new seed can only start when the previous seed reaches BLENDING or later.

### 1.4 Design Intent

The user clarified: **"In most cases (like 99%) WAIT is the right move. We need to learn it's not 'the only move' — she's meant to be like a sentinel acting rarely and decisively."**

This is a sparse-action RL problem, analogous to:
- Trading systems (hold most of the time)
- Alarm systems (mostly quiet)
- Human oversight (intervene only when needed)

---

## Part 2: The DeepSeek Analysis (External Input)

DeepSeek was shown a screenshot with these metrics (episode 110):
- Grad Norm: 14.061
- KL Divergence: ~0
- Clip Fraction: 0
- Value Range: [-0.31, -0.30] with std≈0

### DeepSeek's Claims

1. "Grad Norm 14.061 is concerning but not catastrophic"
2. "In PPO, gradients typically range 0.1-5.0 when well-tuned"
3. "KL Divergence very low - good"
4. "Clip Fraction 0 - no clipping needed in surrogate loss"
5. "Value function collapsed - predictions have narrow range"
6. "Policy update is fine, value network is unstable"

### DeepSeek's Proposed Solutions

1. Increase gradient clipping (0.5 → 2.0)
2. Reduce value loss coefficient
3. Add value loss clipping
4. Value network normalization with small init
5. Gradient penalty (L2 on params)
6. Learning rate warm-up/decay
7. Per-layer gradient clipping
8. Switch optimizer (Adam → RMSprop)

---

## Part 3: Actual Telemetry Data

### 3.1 PPO Update Metrics (13 batches, episodes 10-130)

| Episode | Grad Norm | KL Divergence | Clip Frac | Explained Var | Value Std | Adv Skew | Adv Kurt | Adv Pos% |
|---------|-----------|---------------|-----------|---------------|-----------|----------|----------|----------|
| 10 | 9.81 | 2.6e-9 | 0 | -0.0006 | 0.0024 | 1.05 | 0.34 | 36.3% |
| 20 | 12.00 | 4.8e-9 | 0 | 0.0002 | 0.0021 | 0.85 | 0.11 | 38.9% |
| 30 | 12.26 | 5.0e-9 | 0 | -0.0003 | 0.0022 | 0.17 | -0.17 | 44.9% |
| 40 | 12.20 | 4.8e-9 | 0 | -0.0005 | 0.0023 | -0.75 | 2.40 | 49.5% |
| 50 | 3.32 | 5.9e-9 | 0 | -0.0001 | 0.0022 | 3.88 | 18.23 | 28.5% |
| 60 | 4.54 | 6.6e-9 | 0 | 0.0001 | 0.0018 | 1.02 | 9.22 | 53.5% |
| 70 | 16.87 | 5.5e-9 | 0 | 0.00002 | 0.0020 | 2.74 | 7.51 | 25.1% |
| 80 | 7.57 | 1.9e-9 | 0 | 0.0001 | 0.0016 | 0.51 | 1.66 | 45.2% |
| 90 | **43.27** | -7.1e-11 | 0 | 0.00003 | 0.0016 | 3.23 | 9.92 | 21.1% |
| 100 | 16.16 | -8.2e-10 | 0 | 0.00002 | 0.0016 | 3.97 | 15.69 | **11.3%** |
| 110 | 14.06 | -1.7e-9 | 0 | 0.0003 | 0.0016 | 3.26 | 11.01 | 20.5% |
| 120 | 21.28 | -8.3e-10 | 0 | 0.0002 | 0.0017 | 3.72 | 15.19 | 19.1% |
| 130 | 23.40 | 8.9e-10 | 0 | 0.0001 | 0.0018 | 2.45 | 5.31 | 25.5% |

### 3.2 Per-Head Entropy Trajectory

| Episode | slot | blueprint | style | tempo | alpha_target | op |
|---------|------|-----------|-------|-------|--------------|-----|
| 10 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.515 |
| 20 | 0.120 | 0.120 | 0.120 | 0.120 | 0.119 | 0.528 |
| 30 | 0.105 | 0.105 | 0.105 | 0.104 | 0.105 | 0.559 |
| 40 | 0.080 | 0.082 | 0.087 | 0.080 | 0.088 | 0.527 |
| 50 | 0.051 | 0.055 | 0.069 | 0.054 | 0.069 | 0.458 |
| 60 | 0.030 | 0.038 | 0.048 | 0.038 | 0.048 | 0.353 |
| 70 | 0.016 | 0.027 | 0.043 | 0.029 | 0.044 | 0.275 |
| 80 | 0.007 | 0.018 | 0.034 | 0.021 | 0.038 | 0.204 |
| 90 | 0.003 | 0.012 | 0.048 | 0.016 | 0.053 | 0.192 |
| 100 | 0.001 | 0.008 | 0.022 | 0.011 | 0.024 | 0.157 |
| 110 | **0.0006** | **0.007** | 0.042 | **0.010** | 0.048 | 0.184 |
| 120 | **0.0003** | **0.007** | 0.046 | **0.009** | 0.055 | 0.176 |
| 130 | **0.0002** | **0.009** | 0.056 | **0.011** | 0.074 | 0.167 |

**Configured Entropy Floors (from leyline):**
- slot: 0.20 → actual 0.0002 (1000x below!)
- blueprint: 0.40 → actual 0.009 (44x below!)
- tempo: 0.40 → actual 0.011 (36x below!)
- style: 0.30 → actual 0.056 (5x below)
- op: 0.15 → actual 0.167 (above floor ✓)

### 3.3 Action Distribution (20,443 decisions)

| Action | Count | Percentage | Avg Reward |
|--------|-------|------------|------------|
| WAIT | 18,063 | **88.4%** | +0.20 |
| GERMINATE | 997 | 4.9% | +0.17 |
| PRUNE | 905 | 4.4% | -1.06 |
| SET_ALPHA_TARGET | 438 | 2.1% | +1.70 |
| FOSSILIZE | 40 | 0.2% | +1.30 |

### 3.4 Seed Lifecycle Statistics

- Seeds germinated: 997
- Seeds fossilized: 40 (4.0% success rate)
- Seeds pruned: 905 (90.8% prune rate)
- Currently active: 52

Blueprint distribution when germinating:
| Blueprint | Germinated | Fossilized | Success Rate |
|-----------|------------|------------|--------------|
| conv_heavy | 182 | 16 | 8.8% |
| depthwise | 158 | 10 | 6.3% |
| conv_light | 109 | 6 | 5.5% |
| bottleneck | 109 | 4 | 3.7% |
| norm | 102 | 4 | 3.9% |
| conv_small | 252 | 0 | 0% |
| attention | 85 | 0 | 0% |

---

## Part 4: Investigation Journey (Learning Process)

### Phase 1: Initial Assessment (Standard PPO Diagnostics)

**Hypothesis:** DeepSeek misinterpreted KL≈0 and clip_fraction=0 as positive signs when they indicate policy collapse.

**Evidence gathered:**
- KL divergence consistently ~1e-9 (essentially zero)
- Clip fraction always 0
- Explained variance ~0.0001 (value function not predictive)
- Head entropies collapsing across all heads

**Initial conclusion:** Classic entropy collapse → policy frozen → KL=0 because policy isn't changing.

### Phase 2: User Clarification (Sentinel Design)

**User input:** "In most cases (like 99%) WAIT is the right move. She's meant to be like a sentinel."

**Reframe:** This is a sparse-action RL problem. High WAIT percentage might be correct.

**New question:** Is 88% WAIT too much, too little, or about right?

### Phase 3: Action Masking Deep Dive

**Discovery:** Esper has aggressive action masking via D3 sequential development.

**Key code from `tamiyo/policy/action_masks.py`:**

```python
# D3: Stages that block GERMINATE - enforces sequential seed development
_BLOCKS_GERMINATION = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
})

# D3: Cleanup stages that also block GERMINATE
_CLEANUP_BLOCKS_GERMINATION = frozenset({
    SeedStage.PRUNED.value,
    SeedStage.EMBARGOED.value,
    SeedStage.RESETTING.value,
})

can_germinate = (
    has_empty_enabled_slot
    and not seed_limit_reached
    and not has_developing_seed  # Blocked if ANY seed in above stages
)
```

**Implication:** GERMINATE is blocked for most of the lifecycle. The 88% WAIT isn't policy collapse—it's action masking.

### Phase 4: Entropy Floor System Analysis

**Discovery:** Esper has per-head entropy floors with quadratic penalty.

**Code from `leyline/__init__.py`:**

```python
ENTROPY_FLOOR_PER_HEAD: dict[str, float] = {
    "op": 0.15,           # Always active - can exploit more
    "slot": 0.20,         # Usually active (~60%)
    "blueprint": 0.40,    # GERMINATE only (~18%) - CRITICAL: needs high floor
    "style": 0.30,        # GERMINATE + SET_ALPHA_TARGET (~22%)
    "tempo": 0.40,        # GERMINATE only (~18%) - needs high floor
    "alpha_target": 0.25,
    "alpha_speed": 0.20,
    "alpha_curve": 0.20,
}

ENTROPY_FLOOR_PENALTY_COEF: dict[str, float] = {
    "op": 0.05,
    "slot": 0.15,
    "blueprint": 0.50,    # CRITICAL: prone to collapse
    "style": 0.20,
    "tempo": 0.50,        # CRITICAL: prone to collapse
    # ...
}
```

**Penalty formula:** `loss += coef * max(0, floor - entropy)²`

**The puzzle:** Blueprint entropy is 0.009 with floor 0.40 and penalty coefficient 0.50. The penalty should be:
```
penalty = 0.50 * (0.40 - 0.009)² = 0.50 * 0.153 = 0.077
```

That's a significant penalty! Why isn't it working?

### Phase 5: Gradient Flow Hypothesis

**Hypothesis:** Entropy floor penalty creates a loss signal but gradients can't flow.

**The chain rule problem:**
```
∂penalty/∂θ = ∂penalty/∂entropy × ∂entropy/∂θ
```

When entropy is near-zero (deterministic policy):
- `∂penalty/∂entropy` is large (quadratic penalty pushes hard)
- BUT `∂entropy/∂θ` is near-zero (small weight changes don't affect a deterministic distribution)

**The death spiral:**
1. Early training: policy becomes overconfident on one blueprint
2. Entropy drops → `∂entropy/∂θ` shrinks
3. Floor penalty exists but gradients don't flow
4. Entropy continues dropping → even smaller gradients
5. Terminal state: entropy ≈ 0, no gradient pathway to recover

---

## Part 5: Revised Diagnosis

### What's Actually Wrong

| Issue | Severity | Root Cause |
|-------|----------|------------|
| 88% WAIT rate | **Expected** | D3 sequential masking (not policy collapse) |
| slot entropy 0.0002 | **Ambiguous** | Often only 1 slot valid per op; may be expected |
| blueprint entropy 0.009 | **CRITICAL** | Gradient death at low entropy; floor penalty ineffective |
| tempo entropy 0.011 | **CRITICAL** | Same gradient death problem |
| KL ≈ 0 | **Expected** | Policy frozen on sparse heads + action masking on op |
| Clip fraction = 0 | **Expected** | Natural consequence of above |
| Value std ≈ 0.002 | **Concerning** | Value network may have issues too |

### What DeepSeek Got Right
- Value network has collapsed predictions (correct observation)
- Gradient norms are elevated (correct observation)

### What DeepSeek Got Wrong
- Interpreted KL≈0 as "good" (wrong—it's pathological for learning heads)
- Interpreted clip=0 as "fine" (wrong—indicates no meaningful updates)
- Blamed value network as root cause (wrong—value is downstream symptom)
- 0/8 proposed solutions address entropy collapse

### What We Missed Initially
- Action masking means 88% WAIT is structural, not learned
- The REAL problem is sparse heads (blueprint, tempo) not exploring when they CAN act
- Entropy floor penalty exists but gradient flow is broken

---

## Part 6: Questions for Peer Review

### Q1: Is the Gradient Flow Hypothesis Correct?

The entropy floor penalty should be creating a loss signal of ~0.08 per batch, but entropy continues to fall. Is this because:

a) Gradient of entropy w.r.t. policy params is near-zero at low entropy?
b) Causal masking is zeroing gradients for sparse heads too aggressively?
c) The penalty coefficient is still too low despite being 0.50?
d) Something else entirely?

### Q2: Is KL≈0 Actually Expected Here?

Given that:
- 88% of actions are WAIT (mostly forced by masking)
- Sparse heads are collapsed (always same choice)
- Non-masked decisions happen rarely

Should we expect KL≈0, or is this still pathological?

### Q3: Proposed Interventions

Which of these would you recommend?

**Option A: Temperature scaling during training**
```python
# Prevent confidence lock-in by dividing logits by T > 1
blueprint_logits = blueprint_logits / temperature
```

**Option B: Logit noise injection**
```python
if training:
    logits = logits + torch.randn_like(logits) * noise_scale
```

**Option C: Minimum probability enforcement**
```python
probs = softmax(logits)
probs = torch.clamp(probs, min=min_prob)  # e.g., 0.05
probs = probs / probs.sum()
```

**Option D: Scheduled entropy coefficient increase for sparse heads**
```python
# Late-training entropy boost when collapse detected
if head_entropy < floor:
    entropy_coef = base_coef * (1 + collapse_boost)
```

**Option E: Entropy gradient injection (REINFORCE-style)**
```python
# Direct gradient on entropy using log-derivative trick
entropy_grad_loss = -entropy_coef * (log_prob * (entropy - target_entropy).detach())
```

**Option F: Accept that slot/blueprint will collapse given masking**
- Only ensure op_entropy stays healthy
- Let other heads collapse if they only have 1 valid choice most of the time

### Q4: Is the Run Salvageable?

Given current state:
- 130 episodes completed of 2000 planned
- Blueprint/tempo heads are collapsed
- Accuracy plateaued at ~21%

Should we:
a) Continue and see if exploration recovers?
b) Stop and restart with intervention?
c) Hot-patch entropy coefficients mid-run?

### Q5: Sparse Head Metric Validity

Current metrics average head entropy across ALL timesteps. But:
- Blueprint head is only relevant ~5% of the time (when GERMINATE is valid)
- Averaging over WAIT steps dilutes the signal

Should we track **conditional entropy** (entropy when the head actually matters)?

---

## Part 7: Relevant Code Snippets

### 7.1 Entropy Floor Penalty Computation

From `simic/agent/ppo_update.py`:

```python
def compute_entropy_floor_penalty(
    entropy: dict[str, torch.Tensor],
    head_masks: dict[str, torch.Tensor],
    entropy_floor: dict[str, float],
    penalty_coef: dict[str, float],
) -> torch.Tensor:
    """Compute penalty for heads whose entropy falls below floor.

    Uses quadratic penalty: loss += coef * max(0, floor - entropy)^2
    """
    total_penalty = torch.tensor(0.0, device=next(iter(entropy.values())).device)

    for head in entropy_floor:
        if head in entropy:
            floor = entropy_floor[head]
            coef = penalty_coef[head]
            head_ent = entropy[head]

            # Quadratic penalty when below floor
            shortfall = torch.clamp(floor - head_ent, min=0)
            total_penalty = total_penalty + coef * (shortfall ** 2)

    return total_penalty
```

### 7.2 Causal Masking for Sparse Heads

The PPO update applies causal masks to zero gradients when heads aren't decision-relevant:

```python
# From ppo_update.py compute_losses()
for key in entropy:
    head_coef = entropy_coef_per_head[key]
    mask = head_masks[key]

    # D1: Combine causal mask with forced-action mask for entropy
    effective_mask = mask  # ... complex masking logic

    n_valid = effective_mask.sum().clamp(min=1)
    masked_ent = (entropy[key] * effective_mask).sum() / n_valid
    entropy_loss = entropy_loss - head_coef * masked_ent
```

### 7.3 MaskedCategorical Entropy Calculation

From `tamiyo/policy/action_masks.py`:

```python
def entropy(self) -> torch.Tensor:
    """Compute normalized entropy over valid actions.

    Returns entropy normalized to [0, 1] by dividing by max entropy.
    When only one action is valid, entropy is exactly 0.
    """
    probs = self._dist.probs
    log_probs = self._dist.logits - self._dist.logits.logsumexp(dim=-1, keepdim=True)
    raw_entropy = -(probs * log_probs * self.mask).sum(dim=-1)
    num_valid = self.mask.sum(dim=-1).clamp(min=1)
    max_entropy = torch.log(num_valid.float())

    # Single valid action = zero entropy (no choice)
    normalized = raw_entropy / safe_max_entropy.clamp(min=1e-8)
    return torch.where(num_valid == 1, torch.zeros_like(normalized), normalized)
```

### 7.4 Per-Head Entropy Coefficients

From `simic/agent/ppo_agent.py`:

```python
# Sparse heads need higher entropy coefficients
ENTROPY_COEF_PER_HEAD: dict[str, float] = {
    "op": 1.0,       # Always active (100% of steps)
    "slot": 1.0,     # Usually active (~60%)
    "blueprint": 1.3, # GERMINATE only (~18%) — needs boost
    "style": 1.2,    # GERMINATE + SET_ALPHA_TARGET (~22%)
    "tempo": 1.3,    # GERMINATE only (~18%) — needs boost
    "alpha_target": 1.2,
    "alpha_speed": 1.2,
    "alpha_curve": 1.2,
}
```

---

## Part 8: Summary for Peer Review

**The situation:**
1. Esper's PPO agent controls seed lifecycle with a sparse-action policy
2. Action masking enforces sequential development (WAIT is often forced)
3. Sparse heads (blueprint, tempo) have collapsed despite entropy floor penalties
4. The penalty exists but gradients appear unable to flow at low entropy
5. DeepSeek's analysis focused on value network; real issue is sparse head gradient death

**What we need:**
1. Validation or refutation of the gradient flow hypothesis
2. Recommendation on which intervention to implement
3. Assessment of whether to continue or restart the run
4. Guidance on proper metrics for sparse-action policies

**Constraints:**
- Sentinel design is intentional (WAIT-heavy is correct)
- Can't remove action masking (D3 sequential development is load-bearing)
- Need to preserve exploration in sparse heads without disrupting op head

---

---

## Part 9: DRL Expert Peer Review Results

**Review completed:** 2026-01-10

### Gradient Flow Hypothesis: VALIDATED ✓

The expert confirmed the mathematical basis:

```
∂H/∂z_i = π(a_i) × [H(π) - log π(a_i)]
```

When entropy collapses (π approaches one-hot):
- For dominant action: π(a*) → 1, log π(a*) → 0, so ∂H/∂z* → 0
- For suppressed actions: π(a) → 0, and π(a) × |log π(a)| → 0 (L'Hôpital)

**Result:** Quadratic penalty creates loss but NO gradient pathway.

### Contributing Factors Identified

1. **Primary:** ∂entropy/∂θ ≈ 0 at low entropy (gradient death)
2. **Secondary:** Causal masking zeros gradients 95% of the time for blueprint/tempo
3. **Tertiary:** Reward signal CORRECTLY reinforces conv_heavy (exploitation wins)

### KL≈0 Verdict: PATHOLOGICAL

Even with 88% forced WAIT and sparse heads, KL should be measurable. The KL=0 indicates:
- Policy ratios π_new/π_old ≈ 1.0 across ALL heads, ALL timesteps
- This is policy stagnation, not masking effects

### Recommended Intervention: Probability Floor (Option C)

```python
PROBABILITY_FLOOR_PER_HEAD = {
    "op": 0.02,        # Low — high-frequency head
    "slot": 0.03,      # Low — usually few valid choices
    "blueprint": 0.10, # HIGH — critical sparse head
    "style": 0.05,
    "tempo": 0.10,     # HIGH — critical sparse head
    "alpha_target": 0.05,
    "alpha_speed": 0.05,
    "alpha_curve": 0.05,
}
```

Implementation in `MaskedCategorical`:
```python
def _apply_entropy_floor(self, logits: torch.Tensor, min_prob: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    floored_probs = torch.where(self.mask, torch.clamp(probs, min=min_prob), probs)
    floored_probs = floored_probs / (floored_probs * self.mask).sum(dim=-1, keepdim=True)
    return torch.log(floored_probs + 1e-8)
```

### Run Salvageability: YES

- Only 8% through training (130/2000 episodes)
- Value network not completely collapsed
- Reward signal coherent
- 40 fossilizations achieved (learning something)

**Recommended action:**
1. Stop run
2. Implement probability floor
3. Optionally reset blueprint/tempo head weights
4. Resume from checkpoint
5. Monitor conditional entropy

### Rejected Interventions

- **Option B (logit noise):** Violates policy gradient theory
- **Option D (higher entropy coef):** Same gradient death problem
- **Option F (accept collapse):** Blueprint has 6-7 valid choices; collapse is pathology

### New Metric Recommended: Conditional Entropy

```python
conditional_blueprint_entropy = mean(H[blueprint] WHERE op=GERMINATE)
```

Current metrics dilute sparse head entropy 20x by averaging over WAIT steps.

---

*End of briefing pack*
