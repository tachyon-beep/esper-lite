# Simic Subsystem: Complete Reward Function Documentation

> **Audit Scope**: Full documentation of all reward functions, modes, and shaping mechanisms in `src/esper/simic/rewards/`
>
> **Audit Date**: 2026-01-11
>
> **Reviewer**: DRL Expert + Reward Shaping Engineering Skills

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Reward Family: CONTRIBUTION](#2-reward-family-contribution)
   - [Mode: SHAPED](#21-mode-shaped-default)
   - [Mode: ESCROW](#22-mode-escrow)
   - [Mode: BASIC](#23-mode-basic)
   - [Mode: SPARSE](#24-mode-sparse)
   - [Mode: MINIMAL](#25-mode-minimal)
   - [Mode: SIMPLIFIED](#26-mode-simplified)
3. [Reward Family: LOSS](#3-reward-family-loss)
4. [PBRS: Potential-Based Reward Shaping](#4-pbrs-potential-based-reward-shaping)
5. [Anti-Gaming Mechanisms](#5-anti-gaming-mechanisms)
6. [Intervention Costs](#6-intervention-costs)
7. [Reward Component Breakdown](#7-reward-component-breakdown)
8. [Mathematical Reference](#8-mathematical-reference)

---

## 1. Architecture Overview

The Simic reward system implements a **two-family dispatch architecture** for the Tamiyo seed lifecycle controller:

```
RewardFamily (top-level)
├── CONTRIBUTION → Uses counterfactual validation (seed_contribution)
│   ├── SHAPED (default) - Dense shaping with PBRS, attribution, anti-gaming
│   ├── ESCROW - Dense, reversible attribution (anti-peak/anti-thrash)
│   ├── BASIC - Accuracy improvement minus parameter rent only
│   ├── SPARSE - Terminal-only (accuracy - param_cost)
│   ├── MINIMAL - Sparse + early-prune penalty
│   └── SIMPLIFIED - DRL Expert recommended (PBRS + intervention + terminal)
│
└── LOSS → Uses loss delta as primary signal
    └── (single mode) - Normalized loss delta + rent + PBRS + terminal
```

### Key Design Principles

1. **Counterfactual Attribution**: The CONTRIBUTION family uses `seed_contribution` from counterfactual validation to isolate seed impact from host drift.

2. **PBRS Compliance**: All shaping uses the Ng et al. (1999) potential-based formula `F(s,s') = γΦ(s') - Φ(s)` to preserve optimal policy invariance.

3. **Anti-Gaming**: Multiple mechanisms prevent reward hacking (ransomware signatures, ratio penalties, timing discounts).

4. **Configurable Ablation**: Each component can be disabled independently for systematic experimentation.

### Core Constants (from `leyline`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_GAMMA` | 0.995 | Discount factor for PBRS |
| `MIN_PRUNE_AGE` | 1 | Minimum epochs before pruning allowed |
| `MIN_HOLDING_EPOCHS` | 5 | Epochs required in HOLDING for legitimacy |
| `DEFAULT_MIN_FOSSILIZE_CONTRIBUTION` | 1.0 | Threshold for "contributing" fossilized seed |

---

## 2. Reward Family: CONTRIBUTION

**Entry Point**: `compute_contribution_reward()` in `contribution.py`

**Primary Signal**: `seed_contribution` (counterfactual validation result)

**Config Class**: `ContributionRewardConfig`

---

### 2.1 Mode: SHAPED (Default)

The most sophisticated reward mode with dense shaping, anti-gaming, and full PBRS.

#### Formula (Conceptual)

```
R_total = bounded_attribution
        + blending_warning
        + holding_warning
        + pbrs_bonus
        + synergy_bonus
        - rent_penalty
        + alpha_shock
        - occupancy_rent
        - fossilized_rent
        + first_germinate_bonus
        + action_shaping
        + terminal_bonus
```

#### Bounded Attribution

The core reward signal, computed from counterfactual `seed_contribution`:

```python
# For positive contribution with progress:
if seed_contribution >= 0 and progress > 0:
    if seed_contribution >= progress:
        attributed = formula(progress, seed_contribution)  # geometric/harmonic/minimum
    else:
        attributed = seed_contribution

    attributed *= attribution_discount  # Sigmoid discount for negative total_improvement
    attributed *= timing_discount       # D3: Early germination penalty
    bounded_attribution = contribution_weight * attributed + ratio_penalty

# For negative contribution (seed hurting):
elif seed_contribution < 0:
    bounded_attribution = contribution_weight * seed_contribution  # Pass through negative
```

**Attribution Formula Variants** (`attribution_formula` config):

| Formula | Equation | Character |
|---------|----------|-----------|
| `geometric` | `√(progress × contribution)` | Rewards host drift (legacy) |
| `harmonic` | `2pc/(p+c)` | Dominated by smaller value (recommended) |
| `minimum` | `min(progress, contribution)` | Very conservative |

**Proxy Fallback**: When `seed_contribution` is unavailable but `acc_delta > 0`:
```python
bounded_attribution = proxy_contribution_weight * acc_delta
# where proxy_contribution_weight = contribution_weight * proxy_confidence_factor (0.3)
```

#### Blending Warning

Escalating penalty for seeds showing negative trajectory during BLENDING stage:

```python
if stage == BLENDING and total_improvement < 0:
    escalation = min(epochs_in_stage * 0.05, 0.3)
    blending_warning = -0.1 - escalation
```

#### Holding Indecision Penalty

Penalizes non-terminal actions in HOLDING stage (prevents "turntabling"):

```python
if stage == HOLDING and action not in (FOSSILIZE, PRUNE):
    if epochs_in_stage >= 2 and bounded_attribution > 0:
        base_penalty = 0.1
        ramp_penalty = max(0, epochs_waiting - 1) * 0.05
        holding_warning = -min(base_penalty + ramp_penalty, 0.3)
```

#### Compute Rent (Parameter Cost)

Logarithmic cost for seed parameters:

```python
growth_ratio = effective_seed_params / max(host_params, rent_host_params_floor)
scaled_cost = log(1.0 + growth_ratio)
rent_penalty = min(rent_weight * scaled_cost, max_rent)
```

**Config Defaults**:
- `rent_weight`: 0.5
- `max_rent`: 1.5
- `rent_host_params_floor`: 200

#### D2: Capacity Economics (Slot Saturation Prevention)

```python
# Occupancy rent for slots above free threshold
n_occupied = n_active_seeds + num_fossilized_seeds
excess_occupied = max(0, n_occupied - free_slots)
occupancy_rent = seed_occupancy_cost * excess_occupied  # 0.01 per excess slot

# Fossilized maintenance rent
fossilized_rent = fossilized_maintenance_cost * num_fossilized_seeds  # 0.002 per fossil

# First germination bonus (breaks "do nothing" symmetry)
if action == GERMINATE and seeds_germinated_this_episode == 0:
    first_germ_bonus = 0.2
```

#### Alpha Shock Penalty

Convex penalty on rapid alpha changes (prevents oscillation):

```python
raw_shock = -alpha_shock_coef * alpha_delta_sq_sum  # coef = 0.1958
alpha_shock = max(raw_shock, -alpha_shock_cap)  # cap = 1.0
```

#### Terminal Bonus

Episode completion incentive:

```python
if epoch == max_epochs:
    terminal_bonus = val_acc * terminal_acc_weight  # 0.05

    # Quality-weighted fossil bonus (tanh saturation)
    if num_contributing_fossilized > 0:
        fossilize_terminal_bonus = fossilize_terminal_scale * tanh(
            num_contributing_fossilized / fossilize_quality_ceiling
        )
        # With ceiling=3: 1 fossil ≈ 32%, 3 ≈ 76%, 5 ≈ 93% of max
```

---

### 2.2 Mode: ESCROW

Dense attribution with **reversible credit** to prevent peak-chasing and thrashing.

#### Key Difference from SHAPED

Instead of paying immediate attribution, ESCROW tracks an "escrow credit target" and pays only the **delta** from previous credit:

```python
# Credit target is based on stable accuracy (min of recent window)
escrow_credit_target = max(0.0, contribution_weight * attributed + ratio_penalty)

# Pay only the change
escrow_delta = escrow_credit_target - escrow_credit_prev

# Clip large swings for PPO stability
if escrow_delta_clip > 0:
    escrow_delta = clip(escrow_delta, -escrow_delta_clip, escrow_delta_clip)

bounded_attribution = escrow_delta
```

**Why ESCROW Exists**: Standard SHAPED mode rewards transient accuracy spikes immediately. If accuracy later regresses, the agent already received credit. ESCROW uses `stable_val_acc = min(recent_k_accuracies)` and only pays for sustained improvements.

**Config Defaults**:
- `escrow_stable_window`: 3 epochs
- `escrow_delta_clip`: 2.0 (max per-step swing)

#### Special Cases

- **PRUNE**: Zeroes out escrow credit target (forfeit unrealized gains)
- **FOSSILIZE**: Freezes escrow credit (no further changes)
- **Negative contribution**: Passes through immediately (no escrow delay for harm)

---

### 2.3 Mode: BASIC

Minimal reward using only accuracy improvement and parameter rent.

#### Formula

```python
accuracy_improvement = basic_acc_delta_weight * (acc_delta / 100.0)  # weight = 5.0
rent_penalty = param_penalty_weight * (effective_overhead / param_budget)  # 0.1, 500k

reward = accuracy_improvement - rent_penalty
```

**Returns**: `(reward, rent_penalty, growth_ratio)` tuple

**Use Case**: Ablation experiments to isolate the contribution of dense shaping components.

---

### 2.4 Mode: SPARSE

Terminal-only reward (no intermediate signals).

#### Formula

```python
if epoch != max_epochs:
    return 0.0  # Zero reward until terminal

accuracy_reward = committed_val_acc / 100.0
param_cost = param_penalty_weight * (fossilized_seed_params / param_budget)

base_reward = accuracy_reward - param_cost
return sparse_reward_scale * clamp(base_reward, -1.0, 1.0)
```

**Config Defaults**:
- `param_budget`: 500,000
- `param_penalty_weight`: 0.1
- `sparse_reward_scale`: 1.0

**Use Case**: Ground truth baseline for comparing shaping variants.

---

### 2.5 Mode: MINIMAL

Sparse reward + early-prune penalty only.

#### Formula

```python
reward = compute_sparse_reward(...)  # Terminal only

# Add penalty for pruning too young
if action == PRUNE and seed_age < early_prune_threshold:
    reward += early_prune_penalty  # -0.1
```

**Config Defaults**:
- `early_prune_threshold`: 5 epochs
- `early_prune_penalty`: -0.1

**Use Case**: Minimal shaping to prevent degenerate "prune everything" policies.

---

### 2.6 Mode: SIMPLIFIED

DRL Expert recommended 3-component reward.

#### Formula

```python
reward = 0.0

# 1. PBRS stage progression
if seed_info and not disable_pbrs:
    reward += pbrs_bonus

# 2. Intervention cost (small friction)
if action != WAIT:
    reward -= 0.01

# 3. Terminal bonus (main signal)
if epoch == max_epochs:
    accuracy_bonus = (val_acc / 100.0) * 3.0
    fossilize_bonus = num_contributing_fossilized * 2.0
    reward += accuracy_bonus + fossilize_bonus
```

**Use Case**: Baseline for systematic reward engineering experiments. Minimal components, clear signal, easy to reason about.

---

## 3. Reward Family: LOSS

**Entry Point**: `compute_loss_reward()` in `loss_primary.py`

**Primary Signal**: `loss_delta` (validation loss change)

**Config Class**: `LossRewardConfig` (in `leyline/reward_config.py`)

### Formula

```python
# 1. Normalized loss delta (primary signal)
normalized_delta = loss_delta / typical_loss_delta_std
clipped = clip(normalized_delta, -max_loss_delta, max_loss_delta)
if clipped > 0:
    clipped *= regression_penalty_scale  # Asymmetric: punish regression harder
reward = -clipped * loss_delta_weight

# 2. Compute rent (with grace period)
if not in_grace_period:
    growth_ratio = (total_params - host_params) / host_params
    rent_penalty = compute_rent_weight * log(1.0 + growth_ratio)
    reward -= min(rent_penalty, max_rent_penalty)

# 3. PBRS stage bonus
if seed_info:
    reward += pbrs_stage_bonus

# 4. Terminal bonus
if epoch == max_epochs:
    improvement = baseline_loss - val_loss
    normalized = clamp(improvement / achievable_range, 0, 1)
    reward += normalized * terminal_loss_weight
```

### Config Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `loss_delta_weight` | 5.0 | Scale for loss delta signal |
| `max_loss_delta` | 5.0 | Clipping bound (after normalization) |
| `regression_penalty_scale` | 0.5 | Asymmetric penalty for loss increase |
| `typical_loss_delta_std` | 0.1 | Task-specific normalization |
| `compute_rent_weight` | 0.05 | Rent coefficient |
| `max_rent_penalty` | 5.0 | Rent cap |
| `grace_epochs` | 3 | Rent-free period for new seeds |
| `stage_potential_weight` | 0.1 | PBRS weight |
| `baseline_loss` | 2.3 | Initial loss (ln(10) for CIFAR-10) |
| `target_loss` | 0.3 | Achievable loss |
| `terminal_loss_weight` | 1.0 | Terminal bonus scale |

### Task-Specific Presets

```python
# CIFAR-10
LossRewardConfig.for_cifar10()
# baseline_loss=2.3, target_loss=0.3, typical_loss_delta_std=0.05

# TinyStories
LossRewardConfig.for_tinystories()
# baseline_loss=10.8, target_loss=3.5, typical_loss_delta_std=0.15, compute_rent_weight=0.01
```

---

## 4. PBRS: Potential-Based Reward Shaping

**Location**: `shaping.py`

### Theoretical Foundation

From Ng et al. (1999), shaping reward of form `F(s,s') = γΦ(s') - Φ(s)` preserves optimal policy invariance.

### Stage Potentials

```python
STAGE_POTENTIALS = {
    SeedStage.UNKNOWN: 0.0,      # Fallback
    SeedStage.DORMANT: 0.0,      # Baseline
    SeedStage.GERMINATED: 1.0,   # +1.0 for initiating growth
    SeedStage.TRAINING: 2.0,     # +1.0 for G1 passage
    SeedStage.BLENDING: 3.5,     # +1.5 (LARGEST) - value creation phase
    SeedStage.HOLDING: 5.5,      # +2.0 for stability validation
    SeedStage.FOSSILIZED: 6.0,   # +0.5 (SMALLEST) - anti-farming
    SeedStage.PRUNED: 0.0,       # Recycled
    SeedStage.EMBARGOED: 0.0,    # Recycled
    SeedStage.RESETTING: 0.0,    # Recycled
}
```

### PBRS Bonus Computation

```python
def _contribution_pbrs_bonus(seed_info, config):
    # Current potential (stage + epoch progress)
    phi_current = STAGE_POTENTIALS[stage]
    phi_current += min(epochs_in_stage * epoch_progress_bonus, max_progress_bonus)

    # Previous potential (handles stage transitions)
    if epochs_in_stage == 0:  # Just transitioned
        phi_prev = STAGE_POTENTIALS[previous_stage]
        phi_prev += min(previous_epochs_in_stage * epoch_progress_bonus, max_progress_bonus)
    else:  # Same stage
        phi_prev = STAGE_POTENTIALS[stage]
        phi_prev += min((epochs_in_stage - 1) * epoch_progress_bonus, max_progress_bonus)

    return pbrs_weight * (gamma * phi_current - phi_prev)
```

**Config Defaults**:
- `pbrs_weight`: 0.3
- `epoch_progress_bonus`: 0.3
- `max_progress_bonus`: 2.0
- `gamma`: 0.995

### Auxiliary Potential Functions

```python
# Accuracy-based potential (for general use)
def compute_potential(val_acc, epoch, max_epochs):
    time_factor = (max_epochs - epoch) / max_epochs
    return val_acc * time_factor * 0.1

# Seed-state potential (for observation-based computation)
def compute_seed_potential(obs):
    base = STAGE_POTENTIALS[stage]
    progress = min(epochs_in_stage * 0.3, 2.0)
    return base + progress
```

---

## 5. Anti-Gaming Mechanisms

### 5.1 Attribution Discount (Sigmoid)

Discounts reward for seeds with negative total improvement:

```python
if total_improvement < 0:
    exp_arg = min(-attribution_sigmoid_steepness * total_improvement, 700.0)
    attribution_discount = 1.0 / (1.0 + exp(exp_arg))

    # With steepness=3:
    # -0.1% regression → 43% credit
    # -0.5% regression → 18% credit
```

### 5.2 Ratio Penalty

Penalizes high contribution with low improvement (ransomware signature):

```python
if seed_contribution > 1.0 and attribution_discount >= 0.5:
    if total_improvement > safe_threshold:
        ratio = seed_contribution / total_improvement
        if ratio > hacking_ratio_threshold:  # 5.0
            ratio_penalty = -min(
                -prune_good_seed_penalty,
                0.1 * (ratio - threshold) / threshold
            )
    elif total_improvement <= safe_threshold:
        ratio_penalty = prune_good_seed_penalty * min(1.0, seed_contribution / 10.0)
```

### 5.3 Ransomware Signature Detection

Emits warning when seed shows high contribution + negative improvement:

```python
def _check_ransomware_signature(
    seed_contribution,
    total_improvement,
    contribution_threshold=0.1,
    degradation_threshold=-0.2,
):
    """
    Pattern: Seed claims "I'm helping!" while model gets worse.
    This indicates the seed created a dependency (making itself seem valuable
    while corrupting the model to need it).
    """
    if seed_contribution > contribution_threshold and total_improvement < degradation_threshold:
        emit_warning("ransomware_signature", severity="critical" if seed_contribution >= 1.0 else "warning")
```

### 5.4 Timing Discount (D3)

Discounts attribution for early germination to prevent "claim host drift" gaming:

```python
def _compute_timing_discount(germination_epoch, warmup_epochs, discount_floor):
    if germination_epoch >= warmup_epochs:
        return 1.0  # Full credit after warmup

    # Linear interpolation: epoch 0 = floor, epoch warmup = 1.0
    progress = germination_epoch / warmup_epochs
    return discount_floor + (1.0 - discount_floor) * progress
```

**Config Defaults**:
- `germination_warmup_epochs`: 10
- `germination_discount_floor`: 0.4

### 5.5 Prune Age Gate (C3)

Prevents "germinate → hurt → prune" farming:

```python
if action == PRUNE and seed_contribution < prune_hurting_threshold:
    if seed_age < min_prune_bonus_age:  # 3 epochs
        return -0.1  # Small penalty for quick-cycle farming
    return prune_hurting_bonus  # 0.15 for legitimately removing bad seed
```

---

## 6. Intervention Costs

Fixed costs for lifecycle actions (action friction):

| Action | Cost | Purpose |
|--------|------|---------|
| `WAIT` | 0.0 | No cost for observation |
| `GERMINATE` | -0.15 | D3: Makes early germinate net-negative |
| `FOSSILIZE` | -0.01 | Small friction |
| `PRUNE` | -0.005 | Small friction |
| `SET_ALPHA_TARGET` | -0.005 | Small friction |
| `ADVANCE` | 0.0 | No cost (environment-driven) |

### Action-Specific Shaping

Beyond flat costs, some actions have conditional shaping:

```python
# FOSSILIZE: Bonus for contributing seeds, penalty for non-contributing
if action == FOSSILIZE:
    if seed in HOLDING and contribution >= MIN_FOSSILIZE_CONTRIBUTION:
        shaping = fossilize_base_bonus + fossilize_contribution_scale * contribution
        shaping *= legitimacy_discount  # Based on epochs in HOLDING
    elif contribution < MIN_FOSSILIZE_CONTRIBUTION:
        shaping = fossilize_noncontributing_penalty  # -0.2

# PRUNE: Bonus for removing harmful seeds, penalty for removing good ones
if action == PRUNE:
    if contribution < prune_hurting_threshold:
        shaping = prune_hurting_bonus  # 0.15 (with age gate)
    elif contribution < 0:
        shaping = prune_acceptable_bonus  # 0.1
    elif contribution > 0:
        shaping = prune_good_seed_penalty - 0.05 * contribution  # -0.3 base

# GERMINATE: PBRS bonus for starting growth (with timing discount)
if action == GERMINATE and no_existing_seed:
    pbrs = gamma * phi(GERMINATED) - phi(no_seed)
    shaping = pbrs_weight * pbrs * timing_discount
```

---

## 7. Reward Component Breakdown

The `RewardComponentsTelemetry` dataclass captures all reward components for debugging:

### Primary Signal Components
| Field | Description |
|-------|-------------|
| `seed_contribution` | Counterfactual validation result |
| `bounded_attribution` | Final attributed reward after discounts |
| `progress_since_germination` | `val_acc - acc_at_germination` |
| `attribution_discount` | Sigmoid discount for negative improvement |
| `timing_discount` | Early germination discount |

### Escrow Components (ESCROW mode)
| Field | Description |
|-------|-------------|
| `escrow_credit_prev` | Credit from previous step |
| `escrow_credit_target` | Target credit based on stable accuracy |
| `escrow_delta` | Change in credit (clipped) |
| `escrow_credit_next` | Credit for next step |

### Penalty Components
| Field | Description |
|-------|-------------|
| `compute_rent` | Parameter cost (negative) |
| `alpha_shock` | Alpha oscillation penalty |
| `blending_warning` | Negative trajectory during BLENDING |
| `holding_warning` | Indecision during HOLDING |
| `ratio_penalty` | High contribution / low improvement |
| `occupancy_rent` | Slot saturation cost |
| `fossilized_rent` | Fossil maintenance cost |

### Bonus Components
| Field | Description |
|-------|-------------|
| `pbrs_bonus` | Stage progression shaping |
| `synergy_bonus` | Scaffold interaction credit |
| `action_shaping` | Per-action bonuses/penalties |
| `terminal_bonus` | Episode completion reward |
| `fossilize_terminal_bonus` | Fossil count reward (tanh-saturated) |
| `first_germinate_bonus` | Symmetry-breaking for first germination |

### Diagnostic Metric

```python
@property
def shaped_reward_ratio(self):
    """Fraction of reward from shaping terms (high values → potential hacking)"""
    shaped = sum(all_shaping_terms)
    return abs(shaped) / abs(total_reward) if total_reward != 0 else 0.0
```

---

## 8. Mathematical Reference

### Bounded Attribution (SHAPED)

```
R_attr = w_c × f(progress, contribution) × d_sigmoid × d_timing + r_ratio

where:
- w_c = contribution_weight (1.0)
- f() = attribution formula (geometric/harmonic/minimum)
- d_sigmoid = 1 / (1 + exp(-k × total_improvement)) for negative improvement
- d_timing = floor + (1 - floor) × (epoch / warmup) for early germination
- r_ratio = ratio penalty for hacking patterns
```

### PBRS Stage Progression

```
R_pbrs = w × (γ × Φ(s') - Φ(s))

where:
- w = pbrs_weight (0.3)
- γ = DEFAULT_GAMMA (0.995)
- Φ(s) = STAGE_POTENTIALS[stage] + min(epochs × 0.3, 2.0)
```

### Compute Rent

```
R_rent = -min(w_r × log(1 + growth_ratio), max_rent)

where:
- w_r = rent_weight (0.5)
- growth_ratio = effective_seed_params / max(host_params, 200)
- max_rent = 1.5
```

### Terminal Bonus

```
R_term = val_acc × w_t + scale × tanh(n_fossils / ceiling)

where:
- w_t = terminal_acc_weight (0.05)
- scale = fossilize_terminal_scale (3.0)
- ceiling = fossilize_quality_ceiling (3.0)
```

### Escrow Delta

```
credit_target = max(0, w_c × attributed + ratio_penalty)
delta = clip(credit_target - credit_prev, -clip_bound, clip_bound)
R_escrow = delta
```

---

## Appendix: Config Parameter Reference

### ContributionRewardConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contribution_weight` | 1.0 | Scale for counterfactual attribution |
| `proxy_confidence_factor` | 0.3 | Proxy = contribution_weight × this |
| `escrow_stable_window` | 3 | Epochs for stable accuracy min |
| `escrow_delta_clip` | 2.0 | Max per-step escrow swing |
| `pbrs_weight` | 0.3 | PBRS bonus scale |
| `epoch_progress_bonus` | 0.3 | Per-epoch potential increase |
| `max_progress_bonus` | 2.0 | Epoch bonus cap |
| `rent_weight` | 0.5 | Parameter rent coefficient |
| `max_rent` | 1.5 | Per-step rent cap |
| `rent_host_params_floor` | 200 | Min host params for rent calc |
| `base_slot_rent_ratio` | 0.0039 | Alpha-weighted rent floor |
| `alpha_shock_coef` | 0.1958 | Alpha oscillation penalty |
| `alpha_shock_cap` | 1.0 | Max alpha shock penalty |
| `invalid_fossilize_penalty` | -1.0 | Wrong-stage fossilize |
| `prune_fossilized_penalty` | -1.0 | Pruning fossilized seed |
| `germinate_with_seed_penalty` | -0.3 | Germinate with existing seed |
| `germinate_cost` | -0.15 | Action friction |
| `fossilize_cost` | -0.01 | Action friction |
| `prune_cost` | -0.005 | Action friction |
| `set_alpha_target_cost` | -0.005 | Action friction |
| `fossilize_base_bonus` | 0.5 | Base fossilize bonus |
| `fossilize_contribution_scale` | 0.1 | Per-unit contribution bonus |
| `fossilize_noncontributing_penalty` | -0.2 | Low-contribution fossilize |
| `prune_hurting_bonus` | 0.15 | Removing harmful seed |
| `prune_acceptable_bonus` | 0.1 | Removing non-contributing |
| `prune_good_seed_penalty` | -0.3 | Removing contributing seed |
| `prune_hurting_threshold` | -0.5 | Definition of "hurting" |
| `min_prune_bonus_age` | 3 | Age gate for prune bonus |
| `improvement_safe_threshold` | 0.1 | Ratio penalty threshold |
| `hacking_ratio_threshold` | 5.0 | Contribution/improvement limit |
| `attribution_sigmoid_steepness` | 3.0 | Discount curve for regression |
| `terminal_acc_weight` | 0.05 | Accuracy terminal bonus |
| `fossilize_terminal_scale` | 3.0 | Fossil count bonus scale |
| `fossilize_quality_ceiling` | 3.0 | Tanh saturation ceiling |
| `gamma` | 0.995 | Discount factor (from leyline) |
| `reward_mode` | SHAPED | Active reward mode |
| `disable_pbrs` | False | Ablation flag |
| `disable_terminal_reward` | False | Ablation flag |
| `disable_anti_gaming` | False | Ablation flag |
| `param_budget` | 500,000 | Parameter budget (sparse/minimal) |
| `param_penalty_weight` | 0.1 | Param penalty (sparse/minimal) |
| `sparse_reward_scale` | 1.0 | Sparse reward multiplier |
| `early_prune_threshold` | 5 | Minimal mode prune gate |
| `early_prune_penalty` | -0.1 | Minimal mode prune penalty |
| `auto_prune_penalty` | -0.2 | Environment auto-prune penalty |
| `seed_occupancy_cost` | 0.01 | D2: Per-slot excess cost |
| `free_slots` | 1 | D2: Rent-free slots |
| `fossilized_maintenance_cost` | 0.002 | D2: Per-fossil maintenance |
| `first_germinate_bonus` | 0.2 | D2: First germ bonus |
| `germination_warmup_epochs` | 10 | D3: Warmup period |
| `germination_discount_floor` | 0.4 | D3: Min discount factor |
| `disable_timing_discount` | False | D3: Ablation flag |
| `attribution_formula` | "harmonic" | D3: Formula variant |

### LossRewardConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_delta_weight` | 5.0 | Loss delta scale |
| `max_loss_delta` | 5.0 | Clipping bound |
| `regression_penalty_scale` | 0.5 | Asymmetric loss increase penalty |
| `typical_loss_delta_std` | 0.1 | Normalization factor |
| `compute_rent_weight` | 0.05 | Rent coefficient |
| `max_rent_penalty` | 5.0 | Rent cap |
| `grace_epochs` | 3 | Rent-free period |
| `stage_potential_weight` | 0.1 | PBRS weight |
| `baseline_loss` | 2.3 | Initial loss |
| `target_loss` | 0.3 | Achievable loss |
| `terminal_loss_weight` | 1.0 | Terminal bonus scale |

---

*Document generated by Simic Subsystem Audit (Reward Function Scope)*
