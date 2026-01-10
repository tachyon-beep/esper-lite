# Esper Reward Function Specification

**Version:** 1.0
**Date:** 2026-01-10
**Scope:** Complete documentation of all reward functions in the Tamiyo/Simic training system

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [RewardMode Enum](#2-rewardmode-enum)
3. [SHAPED Reward Function](#3-shaped-reward-function)
4. [ESCROW Reward Function](#4-escrow-reward-function)
5. [SPARSE Reward Function](#5-sparse-reward-function)
6. [MINIMAL Reward Function](#6-minimal-reward-function)
7. [BASIC Reward Function](#7-basic-reward-function)
8. [SIMPLIFIED Reward Function](#8-simplified-reward-function)
9. [LOSS Family Reward Function](#9-loss-family-reward-function)
10. [PBRS Shaping System](#10-pbrs-shaping-system)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Architecture Overview

The Esper reward system is organized into two **reward families**:

```
RewardFamily
├── CONTRIBUTION (default)
│   ├── SHAPED (default mode)
│   ├── ESCROW
│   ├── SPARSE
│   ├── MINIMAL
│   ├── BASIC
│   └── SIMPLIFIED
└── LOSS
    └── Loss-primary reward
```

### Key Files

| File | Purpose |
|------|---------|
| `simic/rewards/rewards.py` | Public dispatch API and exports |
| `simic/rewards/contribution.py` | Contribution-family implementations |
| `simic/rewards/loss_primary.py` | Loss-family implementation |
| `simic/rewards/shaping.py` | PBRS utilities and stage potentials |
| `simic/rewards/types.py` | Input dataclasses (`ContributionRewardInputs`, `LossRewardInputs`) |
| `leyline/reward_config.py` | `LossRewardConfig` dataclass |

### Dispatch Flow

```python
# Entry point: compute_reward(inputs: ContributionRewardInputs)
#   -> Reads config.reward_mode
#   -> Dispatches to appropriate function

# Entry point: compute_reward_for_family(family, inputs)
#   -> RewardFamily.CONTRIBUTION -> compute_reward()
#   -> RewardFamily.LOSS -> compute_loss_reward()
```

---

## 2. RewardMode Enum

```python
class RewardMode(Enum):
    SHAPED = "shaped"       # Dense counterfactual + PBRS + warnings (default)
    ESCROW = "escrow"       # Dense with reversible attribution (anti-peak/anti-thrash)
    BASIC = "basic"         # Accuracy delta minus parameter rent only
    SPARSE = "sparse"       # Terminal-only ground truth
    MINIMAL = "minimal"     # Sparse + early-prune penalty
    SIMPLIFIED = "simplified"  # DRL Expert recommended: PBRS + intervention cost + terminal
```

### Mode Selection Guide

| Mode | Density | Primary Signal | Use Case |
|------|---------|----------------|----------|
| **SHAPED** | Dense | Counterfactual contribution | Default, full feature set |
| **ESCROW** | Dense | Stable (sustained) accuracy | Anti-peak gaming |
| **SPARSE** | Terminal | Final accuracy - param cost | Ground truth baseline |
| **MINIMAL** | Terminal+shaping | Sparse + prune penalty | Minimal viable reward |
| **BASIC** | Dense | acc_delta - rent | Ablation studies |
| **SIMPLIFIED** | Semi-sparse | PBRS + terminal | Clean gradients, recommended |

---

## 3. SHAPED Reward Function

**Function:** `compute_contribution_reward()` (contribution.py:261-653)
**Default:** Yes (when `reward_mode == RewardMode.SHAPED`)

### Formula (Simplified)

```
R_shaped = bounded_attribution
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

### Component Details

#### 3.1 Bounded Attribution

The core signal measuring seed contribution to accuracy improvement.

```python
# When seed_contribution is available (from counterfactual validation):
if seed_contribution < 0:
    bounded_attribution = contribution_weight * seed_contribution
else:
    # Combine progress and contribution using configurable formula
    attributed = _compute_attributed_value(progress, seed_contribution, formula)
    attributed *= attribution_discount  # Sigmoid for regressing seeds
    bounded_attribution = (contribution_weight * attributed) + ratio_penalty
    bounded_attribution *= timing_discount  # D3: early germination discount
```

**Attribution Formulas** (`attribution_formula` config):

| Formula | Calculation | Behavior |
|---------|-------------|----------|
| `geometric` | `sqrt(progress * contribution)` | Rewards host drift (legacy) |
| `harmonic` | `2*p*c/(p+c)` | Dominated by smaller value (recommended) |
| `minimum` | `min(progress, contribution)` | Very conservative |

**Attribution Discount** (anti-gaming):
```python
# Seeds with negative total_improvement get discounted credit
if total_imp < 0:
    exp_arg = min(-attribution_sigmoid_steepness * total_imp, 700.0)
    attribution_discount = 1.0 / (1.0 + exp_arg)
```

**Ratio Penalty** (anti-gaming):
```python
# High contribution/improvement ratio triggers penalty
if ratio > hacking_ratio_threshold:
    ratio_penalty = -min(max_penalty, scale * excess_ratio)
```

**Timing Discount** (D3: anti-timing-gaming):
```python
# Seeds germinated early get discounted attribution
# Linear from discount_floor (epoch 0) to 1.0 (epoch >= warmup)
timing_discount = discount_floor + (1.0 - discount_floor) * (germination_epoch / warmup_epochs)
```

**Default Config Values:**
- `contribution_weight = 1.0`
- `proxy_confidence_factor = 0.3`
- `attribution_sigmoid_steepness = 3.0`
- `hacking_ratio_threshold = 5.0`
- `germination_warmup_epochs = 10`
- `germination_discount_floor = 0.4`
- `attribution_formula = "harmonic"`

#### 3.2 Blending Warning

Escalating penalty for seeds in BLENDING stage with negative total improvement.

```python
if seed_info.stage == STAGE_BLENDING and total_imp < 0:
    escalation = min(epochs_in_stage * 0.05, 0.3)
    blending_warning = -0.1 - escalation
```

**Range:** `[-0.4, 0]`

#### 3.3 Holding Warning

Penalty for indecision in HOLDING stage (non-terminal actions).

```python
if seed_info.stage == STAGE_HOLDING and action not in (FOSSILIZE, PRUNE):
    if epochs_in_stage >= 2 and bounded_attribution > 0:
        base_penalty = 0.1
        ramp_penalty = max(0, epochs_waiting - 1) * 0.05
        holding_warning = -min(base_penalty + ramp_penalty, 0.3)
```

**Purpose:** Prevents "turntabling" (SET_ALPHA_TARGET spam to avoid decision).

#### 3.4 PBRS Bonus

Potential-based reward shaping for stage progression (see [Section 10](#10-pbrs-shaping-system)).

```python
pbrs_bonus = pbrs_weight * (gamma * phi_current - phi_prev)
```

**Default:** `pbrs_weight = 0.3`

#### 3.5 Synergy Bonus

Reward for scaffolding behavior (seed interactions).

```python
if interaction_sum > 0 and attribution_discount >= 0.5 and bounded_attribution > 0:
    raw_bonus = math.tanh(interaction_sum * 0.5)
    synergy_bonus = raw_bonus * 0.1
```

#### 3.6 Compute Rent

Logarithmic parameter cost to prevent model bloat.

```python
if effective_overhead > 0:
    growth_ratio = effective_overhead / max(host_params, rent_host_params_floor)
    scaled_cost = math.log(1.0 + growth_ratio)
    rent_penalty = min(rent_weight * scaled_cost, max_rent)
```

**Default Values:**
- `rent_weight = 0.5`
- `max_rent = 1.5`
- `rent_host_params_floor = 200`

#### 3.7 Alpha Shock

Convex penalty for rapid alpha changes (anti-oscillation).

```python
if alpha_delta_sq_sum > 0:
    raw_shock = -alpha_shock_coef * alpha_delta_sq_sum
    alpha_shock = max(raw_shock, -alpha_shock_cap)
```

**Default Values:**
- `alpha_shock_coef = 0.1958`
- `alpha_shock_cap = 1.0`

#### 3.8 Capacity Economics (D2)

Threshold-based occupancy rent to prevent slot saturation.

```python
# Occupancy rent for slots above threshold
n_occupied = n_active_seeds + num_fossilized_seeds
excess_occupied = max(0, n_occupied - free_slots)
occupancy_rent = seed_occupancy_cost * excess_occupied

# Fossilized maintenance
fossilized_rent = fossilized_maintenance_cost * num_fossilized_seeds

# First germination bonus (breaks "do nothing" symmetry)
if action == GERMINATE and seeds_germinated_this_episode == 0:
    first_germ_bonus = first_germinate_bonus
```

**Default Values:**
- `seed_occupancy_cost = 0.01`
- `free_slots = 1`
- `fossilized_maintenance_cost = 0.002`
- `first_germinate_bonus = 0.2`

#### 3.9 Action Shaping

Intervention costs and action-specific modifiers.

| Action | Components |
|--------|------------|
| **GERMINATE** | `germinate_cost` + PBRS bonus (if no seed) + `germinate_with_seed_penalty` (if seed exists) |
| **FOSSILIZE** | `fossilize_cost` + fossilize shaping (legitimacy/ransomware checks) |
| **PRUNE** | `prune_cost` + prune shaping (contribution-based) + PBRS penalty |
| **SET_ALPHA_TARGET** | `set_alpha_target_cost` |
| **WAIT** | 0 |

**Default Intervention Costs:**
- `germinate_cost = -0.15`
- `fossilize_cost = -0.01`
- `prune_cost = -0.005`
- `set_alpha_target_cost = -0.005`

**Fossilize Shaping:**
```python
# Valid only from HOLDING stage
if seed_info.stage != STAGE_HOLDING:
    return invalid_fossilize_penalty  # -1.0

# Negative improvement = damage penalty + ransomware check
if total_delta < 0:
    penalty = -0.5 - min(|total_delta| * 0.2, 1.0)
    if ransomware_signature:  # High contribution + negative improvement
        penalty -= 0.3

# Positive contribution = bonus scaled by legitimacy
if seed_contribution >= MIN_FOSSILIZE_CONTRIBUTION:
    bonus = (fossilize_base_bonus + fossilize_contribution_scale * seed_contribution) * legitimacy
```

**Prune Shaping:**
```python
# Age gate prevents germinate→hurt→prune farming
if seed_age < MIN_PRUNE_AGE:
    return -0.3 * (MIN_PRUNE_AGE - seed_age)

if seed_contribution < prune_hurting_threshold:  # -0.5
    if seed_age < min_prune_bonus_age:  # Age gate (C3 fix)
        return -0.1  # Young hurting seed: small penalty
    return prune_hurting_bonus  # 0.15

if seed_contribution < 0:
    return prune_acceptable_bonus  # 0.1

# Good seed: scaled penalty
return max(prune_good_seed_penalty - 0.05 * contribution, 3 * prune_good_seed_penalty)
```

#### 3.10 Terminal Bonus

Episode-end reward for accuracy and fossilized seeds.

```python
if epoch == max_epochs:
    terminal_bonus = val_acc * terminal_acc_weight  # 0.05
    terminal_bonus += num_contributing_fossilized * fossilize_terminal_scale  # 3.0
```

---

## 4. ESCROW Reward Function

**Function:** `compute_contribution_reward()` with `reward_mode == RewardMode.ESCROW`
**Purpose:** Dense attribution with reversible credit (anti-peak/anti-thrash)

### Key Difference from SHAPED

Instead of paying attribution directly, ESCROW tracks a "credit target" and pays the **delta**:

```python
# ESCROW mode requires:
# - stable_val_acc: min(last_k_accuracies) for sustained improvement proof
# - return_components=True: for escrow state tracking

# Credit target based on attributed value
escrow_credit_target = max(0.0, contribution_weight * attributed + ratio_penalty)

# Pay the delta (can be negative = clawback)
escrow_delta = escrow_credit_target - escrow_credit_prev
bounded_attribution = escrow_delta
```

### Escrow Mechanics

| Event | Credit Target | Delta |
|-------|---------------|-------|
| Sustained improvement | Increases | Positive payout |
| Accuracy drops | Decreases | Negative (clawback) |
| PRUNE action | Reset to 0 | Forfeit accumulated credit |
| FOSSILIZE | Frozen | No further changes |

**Config Values:**
- `escrow_stable_window = 3` (epochs for min() calculation)
- `escrow_delta_clip = 0.0` (0 = no clip; positive = per-step cap)

### Anti-Gaming Properties

1. **Anti-Peak:** Transient spikes are clawed back when accuracy drops
2. **Anti-Thrash:** Stable accuracy (min of window) required for credit
3. **Prune Forfeit:** PRUNE resets credit to 0 (no escrow farming)

---

## 5. SPARSE Reward Function

**Function:** `compute_sparse_reward()` (contribution.py:656-675)
**Purpose:** Terminal-only ground truth signal

### Formula

```python
def compute_sparse_reward(committed_val_acc, fossilized_seed_params, epoch, max_epochs, config):
    if epoch != max_epochs:
        return 0.0  # Zero reward until terminal

    accuracy_reward = committed_val_acc / 100.0
    param_cost = param_penalty_weight * (fossilized_seed_params / param_budget)

    base_reward = accuracy_reward - param_cost
    clamped_base = max(-1.0, min(1.0, base_reward))

    return sparse_reward_scale * clamped_base
```

### Properties

- **Density:** Zero for all non-terminal steps
- **Signal:** `(accuracy - param_cost) * scale`, clamped to [-1, 1]
- **Use Case:** Baseline for credit assignment studies

**Default Values:**
- `param_budget = 500,000`
- `param_penalty_weight = 0.1`
- `sparse_reward_scale = 1.0`

---

## 6. MINIMAL Reward Function

**Function:** `compute_minimal_reward()` (contribution.py:678-700)
**Purpose:** Sparse + early-prune penalty

### Formula

```python
def compute_minimal_reward(...):
    reward = compute_sparse_reward(...)  # Terminal signal

    # Add early-prune penalty
    if action == PRUNE and seed_age < early_prune_threshold:
        reward += early_prune_penalty

    return reward
```

### Properties

- **Density:** Terminal + early-prune shaping
- **Signal:** Sparse ground truth + degenerate policy prevention
- **Use Case:** Minimal viable reward for "prune-all" policy prevention

**Default Values:**
- `early_prune_threshold = 5` (epochs)
- `early_prune_penalty = -0.1`

---

## 7. BASIC Reward Function

**Function:** `compute_basic_reward()` (contribution.py:703-730)
**Purpose:** Accuracy improvement minus parameter rent only

### Formula

```python
def compute_basic_reward(acc_delta, effective_seed_params, total_params, host_params, config):
    # Accuracy improvement (scaled to stable range)
    accuracy_improvement = basic_acc_delta_weight * (acc_delta / 100.0)

    # Parameter rent
    effective_overhead = effective_seed_params or max(total_params - host_params, 0)
    rent_penalty = param_penalty_weight * (effective_overhead / param_budget)

    return accuracy_improvement - rent_penalty
```

### Properties

- **Density:** Dense (per-step)
- **Signal:** `acc_delta * weight - param_rent`
- **No Lifecycle Shaping:** No PBRS, no action costs, no anti-gaming
- **Use Case:** Ablation studies, baseline comparisons

**Default Value:**
- `basic_acc_delta_weight = 5.0`

---

## 8. SIMPLIFIED Reward Function

**Function:** `compute_simplified_reward()` (contribution.py:733-759)
**Purpose:** DRL Expert recommended clean reward

### Formula

```python
def compute_simplified_reward(action, seed_info, epoch, max_epochs, val_acc, num_contributing_fossilized, config):
    reward = 0.0

    # 1. PBRS bonus only
    if seed_info is not None and not disable_pbrs:
        reward += _contribution_pbrs_bonus(seed_info, config)

    # 2. Universal intervention cost
    if action != WAIT:
        reward -= 0.01

    # 3. Terminal bonus
    if epoch == max_epochs:
        accuracy_bonus = (val_acc / 100.0) * 3.0
        fossilize_bonus = num_contributing_fossilized * 2.0
        reward += accuracy_bonus + fossilize_bonus

    return reward
```

### Properties

- **Density:** Semi-sparse (PBRS + terminal)
- **3 Components Only:**
  1. PBRS stage progression (policy-invariant shaping)
  2. Uniform intervention cost (-0.01 for non-WAIT)
  3. Terminal accuracy + fossilization bonus
- **No Attribution:** No counterfactual signal in dense reward
- **Clean Gradients:** Minimal reward hacking surface
- **Use Case:** Recommended starting point for new experiments

---

## 9. LOSS Family Reward Function

**Function:** `compute_loss_reward()` (loss_primary.py:28-66)
**Config:** `LossRewardConfig` (leyline/reward_config.py)

### Formula

```python
def compute_loss_reward(inputs: LossRewardInputs):
    config = inputs.config or LossRewardConfig.default()
    reward = 0.0

    # 1. Normalized loss delta (improvement = positive reward)
    normalized_delta = loss_delta / typical_loss_delta_std
    clipped = clip(normalized_delta, -max_loss_delta, max_loss_delta)
    if clipped > 0:  # Regression
        clipped *= regression_penalty_scale  # Asymmetric
    reward += (-clipped) * loss_delta_weight

    # 2. Compute rent (same as contribution family)
    if not in_grace_period:
        growth_ratio = (total_params - host_params) / host_params
        rent = min(compute_rent_weight * log(1 + growth_ratio), max_rent_penalty)
        reward -= rent

    # 3. PBRS stage bonus
    if seed_info is not None:
        reward += compute_pbrs_stage_bonus(seed_info, config)

    # 4. Terminal loss bonus
    if epoch == max_epochs:
        improvement = baseline_loss - val_loss
        normalized = clip(improvement / achievable_range, 0, 1)
        reward += normalized * terminal_loss_weight

    return reward
```

### LossRewardConfig Defaults

```python
@dataclass
class LossRewardConfig:
    # Loss delta scaling
    loss_delta_weight: float = 5.0
    max_loss_delta: float = 5.0
    regression_penalty_scale: float = 0.5  # Asymmetric: regression penalized less
    typical_loss_delta_std: float = 0.1

    # Compute rent
    compute_rent_weight: float = 0.05
    max_rent_penalty: float = 5.0
    grace_epochs: int = 3

    # PBRS
    stage_potential_weight: float = 0.1

    # Terminal
    baseline_loss: float = 2.3  # ln(10) for CIFAR-10
    target_loss: float = 0.3
    terminal_loss_weight: float = 1.0
```

### Task-Specific Presets

```python
LossRewardConfig.for_cifar10()
# baseline_loss=2.3, target_loss=0.3, typical_loss_delta_std=0.05

LossRewardConfig.for_tinystories()
# baseline_loss=10.8, target_loss=3.5, typical_loss_delta_std=0.15, compute_rent_weight=0.01
```

---

## 10. PBRS Shaping System

**File:** `simic/rewards/shaping.py`

### Theoretical Foundation

Potential-Based Reward Shaping (Ng et al., 1999):
```
F(s, s') = gamma * phi(s') - phi(s)
```

**Guarantees:**
1. **Policy Invariance:** Optimal policy unchanged by PBRS
2. **Bounded Effect:** Discounted sum affects value by `gamma^T * phi(s_T) - phi(s_0)`
3. **Telescoping:** Per-step bonuses sum to potential difference

### Stage Potentials

```python
STAGE_POTENTIALS = {
    SeedStage.UNKNOWN: 0.0,
    SeedStage.DORMANT: 0.0,
    SeedStage.GERMINATED: 1.0,   # +1.0 for initiating growth
    SeedStage.TRAINING: 2.0,     # +1.0 for G1 passage
    SeedStage.BLENDING: 3.5,     # +1.5 (LARGEST) - value creation phase
    SeedStage.HOLDING: 5.5,      # +2.0 for stability validation
    SeedStage.FOSSILIZED: 6.0,   # +0.5 (SMALLEST) - anti-farming
    SeedStage.PRUNED: 0.0,
    SeedStage.EMBARGOED: 0.0,
    SeedStage.RESETTING: 0.0,
}
```

**Design Rationale:**
- BLENDING has largest increment (+1.5) because this is where value is created
- FOSSILIZED has smallest increment (+0.5) to prevent "fossilization farming"
- Failure stages (PRUNED, EMBARGOED, RESETTING) have zero potential

### PBRS Bonus Calculation

```python
def _contribution_pbrs_bonus(seed_info, config):
    # Current potential = stage potential + epoch progress
    phi_current = STAGE_POTENTIALS[seed_info.stage]
    phi_current += min(epochs_in_stage * epoch_progress_bonus, max_progress_bonus)

    # Previous potential (for telescoping)
    if epochs_in_stage == 0:  # Just transitioned
        phi_prev = STAGE_POTENTIALS[previous_stage]
        phi_prev += min(previous_epochs * epoch_progress_bonus, max_progress_bonus)
    else:  # Same stage, time passed
        phi_prev = STAGE_POTENTIALS[seed_info.stage]
        phi_prev += min((epochs_in_stage - 1) * epoch_progress_bonus, max_progress_bonus)

    return pbrs_weight * (gamma * phi_current - phi_prev)
```

### Supporting Functions

```python
# General potential (accuracy + time)
compute_potential(val_acc, epoch, max_epochs) -> float

# Standard PBRS bonus
compute_pbrs_bonus(potential_prev, potential_next, gamma) -> float

# Observation-based potential (for policy)
compute_seed_potential(obs: dict) -> float
```

---

## 11. Configuration Reference

### ContributionRewardConfig (Full)

```python
@dataclass
class ContributionRewardConfig:
    # === Primary Signal ===
    contribution_weight: float = 1.0
    proxy_confidence_factor: float = 0.3

    # === Escrow Attribution ===
    escrow_stable_window: int = 3
    escrow_delta_clip: float = 0.0

    # === PBRS ===
    pbrs_weight: float = 0.3
    epoch_progress_bonus: float = 0.3
    max_progress_bonus: float = 2.0
    gamma: float = 0.995

    # === Compute Rent ===
    rent_weight: float = 0.5
    max_rent: float = 1.5
    rent_host_params_floor: int = 200
    base_slot_rent_ratio: float = 0.0039
    alpha_shock_coef: float = 0.1958
    alpha_shock_cap: float = 1.0

    # === Enforcement Penalties ===
    invalid_fossilize_penalty: float = -1.0
    prune_fossilized_penalty: float = -1.0
    germinate_with_seed_penalty: float = -0.3

    # === Intervention Costs ===
    germinate_cost: float = -0.15
    fossilize_cost: float = -0.01
    prune_cost: float = -0.005
    set_alpha_target_cost: float = -0.005

    # === Fossilize Shaping ===
    fossilize_base_bonus: float = 0.5
    fossilize_contribution_scale: float = 0.1
    fossilize_noncontributing_penalty: float = -0.2

    # === Prune Shaping ===
    prune_hurting_bonus: float = 0.15
    prune_acceptable_bonus: float = 0.1
    prune_good_seed_penalty: float = -0.3
    prune_hurting_threshold: float = -0.5
    min_prune_bonus_age: int = 3

    # === Anti-Gaming ===
    improvement_safe_threshold: float = 0.1
    hacking_ratio_threshold: float = 5.0
    attribution_sigmoid_steepness: float = 3.0

    # === Terminal ===
    terminal_acc_weight: float = 0.05
    fossilize_terminal_scale: float = 3.0

    # === Basic Mode ===
    basic_acc_delta_weight: float = 5.0

    # === Mode Selection ===
    reward_mode: RewardMode = RewardMode.SHAPED

    # === Ablation Flags ===
    disable_pbrs: bool = False
    disable_terminal_reward: bool = False
    disable_anti_gaming: bool = False

    # === Sparse/Minimal ===
    param_budget: int = 500_000
    param_penalty_weight: float = 0.1
    sparse_reward_scale: float = 1.0
    early_prune_threshold: int = 5
    early_prune_penalty: float = -0.1

    # === Action Shaping ===
    advance_from_training_penalty: float = -0.1
    auto_prune_penalty: float = -0.2

    # === D2: Capacity Economics ===
    seed_occupancy_cost: float = 0.01
    free_slots: int = 1
    fossilized_maintenance_cost: float = 0.002
    first_germinate_bonus: float = 0.2

    # === D3: Timing Discount ===
    germination_warmup_epochs: int = 10
    germination_discount_floor: float = 0.4
    disable_timing_discount: bool = False

    # === D3: Attribution Formula ===
    attribution_formula: Literal["geometric", "harmonic", "minimum"] = "harmonic"
```

### LossRewardConfig (Full)

```python
@dataclass
class LossRewardConfig:
    loss_delta_weight: float = 5.0
    max_loss_delta: float = 5.0
    regression_penalty_scale: float = 0.5
    typical_loss_delta_std: float = 0.1

    compute_rent_weight: float = 0.05
    max_rent_penalty: float = 5.0
    grace_epochs: int = 3

    stage_potential_weight: float = 0.1

    baseline_loss: float = 2.3
    target_loss: float = 0.3
    terminal_loss_weight: float = 1.0
```

---

## Appendix A: Reward Component Summary

| Component | Modes | Range | Purpose |
|-----------|-------|-------|---------|
| bounded_attribution | SHAPED, ESCROW | (-∞, +∞) | Primary seed quality signal |
| blending_warning | SHAPED, ESCROW | [-0.4, 0] | Escalating penalty for regressing BLENDING seeds |
| holding_warning | SHAPED, ESCROW | [-0.3, 0] | Penalty for indecision in HOLDING |
| pbrs_bonus | SHAPED, ESCROW, SIMPLIFIED | ~[-2, +2] | Stage progression shaping |
| synergy_bonus | SHAPED, ESCROW | [0, 0.1] | Scaffolding interaction reward |
| compute_rent | SHAPED, ESCROW, BASIC, LOSS | [0, max_rent] | Parameter efficiency cost |
| alpha_shock | SHAPED, ESCROW | [-cap, 0] | Alpha oscillation penalty |
| occupancy_rent | SHAPED, ESCROW | [0, ∞) | Slot saturation prevention |
| fossilized_rent | SHAPED, ESCROW | [0, ∞) | Frozen compute cost |
| first_germinate_bonus | SHAPED, ESCROW | 0 or 0.2 | Break "do nothing" symmetry |
| action_shaping | SHAPED, ESCROW | varies | Action-specific costs/bonuses |
| terminal_bonus | SHAPED, ESCROW, SIMPLIFIED | [0, ∞) | Episode-end accuracy reward |
| sparse_reward | SPARSE, MINIMAL | [-scale, +scale] | Terminal-only ground truth |
| early_prune_penalty | MINIMAL | penalty or 0 | Prune-all policy prevention |

---

## Appendix B: Anti-Gaming Mechanisms

| Mechanism | Target Exploit | Implementation |
|-----------|----------------|----------------|
| Attribution Discount | Regressing seeds claiming credit | Sigmoid discount based on total_improvement |
| Ratio Penalty | contribution >> improvement | Penalty when ratio exceeds threshold |
| Ransomware Check | High contribution + negative improvement | Telemetry alert + penalty |
| Timing Discount | Early germinate to claim host drift | Linear discount before warmup |
| Holding Warning | WAIT farming in HOLDING | Escalating penalty for indecision |
| Min Prune Bonus Age | Germinate→hurt→prune farming | Age gate before prune bonus |
| Legitimacy Discount | Rush to FOSSILIZE | Scale bonus by epochs in HOLDING |
| Escrow Clawback | Peak gaming | Credit based on stable (min window) accuracy |

---

## Appendix C: Telemetry Integration

All reward components are tracked via `RewardComponentsTelemetry`:

```python
@dataclass
class RewardComponentsTelemetry:
    total_reward: float
    action_name: str
    epoch: int
    seed_stage: int | None
    val_acc: float

    # Attribution
    seed_contribution: float | None
    bounded_attribution: float
    progress_since_germination: float | None
    attribution_discount: float
    ratio_penalty: float
    timing_discount: float

    # Escrow
    escrow_credit_prev: float
    escrow_credit_target: float
    escrow_delta: float
    escrow_credit_next: float

    # Warnings
    blending_warning: float
    holding_warning: float

    # Shaping
    pbrs_bonus: float
    synergy_bonus: float
    action_shaping: float

    # Economics
    compute_rent: float
    growth_ratio: float
    alpha_shock: float
    occupancy_rent: float
    fossilized_rent: float
    first_germinate_bonus: float
    n_active_seeds: int

    # Terminal
    terminal_bonus: float
    fossilize_terminal_bonus: float
    num_fossilized_seeds: int
    num_contributing_fossilized: int
```

Available in Karn via `reward_components` view for debugging and analysis.
