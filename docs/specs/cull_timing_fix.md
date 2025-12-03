# Cull Timing Fix Specification

**Status:** Revised based on DRL + PyTorch expert review

## Problem Statement

After adding action masking to enforce Kasmina state machine rules, the PPO agent exhibits poor sample efficiency (~200 episodes to learn basic behavior). The root cause:

**Before masking:** Invalid actions were no-ops (implicit WAITs). Random exploration had ~14% chance of CULL per step, allowing accidental lifecycle completion ~10% of the time.

**After masking:** In TRAINING stage, only WAIT and CULL are valid. Random exploration has 50% CULL probability per step. Probability of surviving to PROBATIONARY (15 steps) = 0.5^12 ≈ 0.02%.

The agent cannot learn that patience pays off because it almost never experiences lifecycle completion rewards.

Secondary issue: `MIN_CULL_AGE = 3` allows culling before there's enough signal to evaluate seed quality. A seed at -0.5% improvement in epoch 3 might be +3% by epoch 7.

## Solution Overview (Revised)

Three-layer fix based on expert review:

1. **Layer 1 (Structural):** Increase `MIN_CULL_AGE` from 3 to **10** epochs (was 7 in v1)
2. **Layer 2 (PBRS):** Add information potential to reward shaping
3. **Layer 3 (Exploration):** Policy initialization bias toward WAIT

**Key revision:** DRL expert noted 0.4% survival (MIN_CULL_AGE=7) is marginal. MIN_CULL_AGE=10 gives ~3% survival (0.5^5), which is 60x improvement and sufficient for learning.

## Expert Review Findings

### DRL Expert
- MIN_CULL_AGE=7 gives 0.4% survival - marginal for learning
- Recommend MIN_CULL_AGE=10 + policy init bias for ~3% survival
- info_potential overlaps with progress_bonus - scale to 1.0 instead of 2.0
- PBRS formulation is correct and composes cleanly
- Need diagnostics: survival curves, stage completion rates, CULL timing histograms
- Verify GERMINATE-WAIT-CULL cycle is net negative (no farming exploit)

### PyTorch Expert
- **BUG:** rewards.py line 514 has hardcoded `MIN_CULL_AGE = 3` - won't sync!
- Need to import from features.py or consolidate
- Need `prev_obs` to include `seed_age_epochs - 1` for correct PBRS
- Export FULL_EVALUATION_AGE from features.py for consistency

## Detailed Changes (Revised)

### 1. features.py - Increase MIN_CULL_AGE and add FULL_EVALUATION_AGE

```python
# src/esper/simic/features.py

# CHANGE: 3 -> 10 (revised from 7 based on DRL expert recommendation)
# Minimum seed age before CULL is allowed (structural fix for uninformed culls)
# 10 epochs gives ~3% survival rate (0.5^5), sufficient for learning.
MIN_CULL_AGE = 10

# NEW: Epochs needed for confident seed quality assessment (for info potential)
FULL_EVALUATION_AGE = 10  # Aligned with MIN_CULL_AGE

# Update __all__ to export FULL_EVALUATION_AGE
__all__ = [
    "safe",
    "obs_to_base_features",
    "compute_action_mask",
    "MIN_CULL_AGE",
    "FULL_EVALUATION_AGE",  # NEW
    "TaskConfig",
    "normalize_observation",
]
```

**Rationale:** MIN_CULL_AGE=10 gives ~3% survival vs 0.4% at MIN_CULL_AGE=7. This is 60x improvement over current and provides sufficient signal for learning.

### 2. rewards.py - Import constants and add Information Potential

**BUG FIX:** Import MIN_CULL_AGE from features.py instead of hardcoding locally.

```python
# src/esper/simic/rewards.py

# At top of file, add import:
from esper.simic.features import MIN_CULL_AGE, FULL_EVALUATION_AGE

# REMOVE the local MIN_CULL_AGE = 3 on line 514 (now imported)

# Modify compute_seed_potential function (around line 642)
def compute_seed_potential(obs: dict) -> float:
    """Compute potential value Phi(s) based on seed state.

    The potential captures:
    1. Stage progression value (existing) - seeds closer to FOSSILIZED have higher potential
    2. Information accumulation value (NEW) - older seeds have more evaluation data

    Potential-based reward shaping: r' = r + gamma*Phi(s') - Phi(s)
    This preserves optimal policy (PBRS guarantee) while improving learning.

    Args:
        obs: Observation dictionary with has_active_seed, seed_stage,
             seed_epochs_in_stage, seed_age_epochs

    Returns:
        Potential value for the current state
    """
    has_active = obs.get("has_active_seed", 0)
    seed_stage = obs.get("seed_stage", 0)
    epochs_in_stage = obs.get("seed_epochs_in_stage", 0)
    seed_age = obs.get("seed_age_epochs", 0)  # NEW: total age for info potential

    # No potential for inactive seeds or DORMANT (stage 1)
    if not has_active or seed_stage <= 1:
        return 0.0

    # Stage-based potential values (existing)
    stage_potentials = {
        2: 2.0,   # GERMINATED
        3: 4.0,   # TRAINING
        4: 5.5,   # BLENDING
        5: 6.5,   # SHADOWING
        6: 7.0,   # PROBATIONARY
        7: 7.5,   # FOSSILIZED
    }

    base_potential = stage_potentials.get(seed_stage, 0.0)
    progress_bonus = min(epochs_in_stage * 0.5, 3.0)

    # NEW: Information accumulation potential (scaled to 1.0 per DRL expert)
    # Seeds become more valuable as we gain evaluation data.
    # Scaled to 1.0 (not 2.0) to avoid overlap with progress_bonus.
    info_fraction = min(1.0, seed_age / FULL_EVALUATION_AGE)
    info_potential = info_fraction * 1.0  # Max 1.0 additional potential

    return base_potential + progress_bonus + info_potential
```

**Rationale:**
- Import fixes sync bug between features.py and rewards.py
- info_potential scaled to 1.0 (not 2.0) to avoid overlap with progress_bonus
- PBRS-compliant: potential on state, CULL destroys it naturally

### 3. rewards.py - Pass seed_age to potential calculation

Update the PBRS section in `compute_shaped_reward` to include seed_age:

```python
# In compute_shaped_reward, around line 340
    if seed_info is not None:
        if seed_info.epochs_in_stage == 0:
            prev_stage = seed_info.previous_stage
            prev_epochs = 0
        else:
            prev_stage = seed_info.stage
            prev_epochs = seed_info.epochs_in_stage - 1

        # NEW: Compute prev_age for info potential PBRS
        prev_age = max(0, seed_info.seed_age_epochs - 1)

        current_obs = {
            "has_active_seed": 1,
            "seed_stage": seed_info.stage,
            "seed_epochs_in_stage": seed_info.epochs_in_stage,
            "seed_age_epochs": seed_info.seed_age_epochs,  # NEW
        }
        prev_obs = {
            "has_active_seed": 1,
            "seed_stage": prev_stage,
            "seed_epochs_in_stage": prev_epochs,
            "seed_age_epochs": prev_age,  # NEW: required for correct PBRS telescoping
        }
        phi_t = compute_seed_potential(current_obs)
        phi_t_prev = compute_seed_potential(prev_obs)
        pb_bonus = compute_pbrs_bonus(phi_t_prev, phi_t, gamma=0.99)
        reward += config.seed_potential_weight * pb_bonus
```

### 4. networks.py - Policy Initialization Bias (Layer 3)

Add WAIT bias to policy network initialization:

```python
# src/esper/simic/networks.py

# In PolicyNetwork.__init__ or ActorCritic.__init__, after creating output layer:
def _init_action_bias(self):
    """Initialize policy bias to favor WAIT over CULL.

    This provides a reasonable starting policy without hard constraints.
    The bias will be learned away if CULL is actually optimal.

    Action layout: [WAIT, GERMINATE_*, FOSSILIZE, CULL]
    """
    with torch.no_grad():
        # Get the output layer (last linear layer in actor)
        output_layer = self.actor[-1] if hasattr(self, 'actor') else self.policy_head
        if hasattr(output_layer, 'bias') and output_layer.bias is not None:
            output_layer.bias[0] = 0.5   # WAIT - slight positive bias
            output_layer.bias[-1] = -0.5  # CULL - slight negative bias
```

**Rationale:** Shifts initial softmax toward WAIT (~62%) without hard constraints. The bias is learned away if CULL is genuinely better in some states.

### 5. rewards.py - Update _cull_shaping terminal PBRS

Update the terminal PBRS correction in `_cull_shaping` to include seed_age:

```python
# In _cull_shaping, around line 537
    # Terminal PBRS correction: account for potential loss from destroying the seed
    current_obs = {
        "has_active_seed": 1,
        "seed_stage": stage,
        "seed_epochs_in_stage": seed_info.epochs_in_stage,
        "seed_age_epochs": seed_info.seed_age_epochs,  # NEW: include info potential
    }
    phi_current = compute_seed_potential(current_obs)
    # PBRS: gamma * phi(next) - phi(current) where next = no seed (phi=0)
    pbrs_correction = 0.99 * 0.0 - phi_current  # = -phi_current
    terminal_pbrs = config.seed_potential_weight * pbrs_correction
```

### 6. rewards.py - Remove redundant age penalty from _cull_shaping

The age penalty at the top of `_cull_shaping` is now redundant because:
1. Action masking prevents CULL before MIN_CULL_AGE = 7
2. Information potential provides graduated penalty for early culls

Remove or comment out:

```python
# In _cull_shaping, REMOVE this block (lines 511-517):
#    # Age penalty: culling a very young seed wastes the germination investment.
#    # Scale: -0.3 per epoch missing from minimum age (3 epochs)
#    MIN_CULL_AGE = 3
#    if seed_age < MIN_CULL_AGE and stage in (STAGE_GERMINATED, STAGE_TRAINING):
#        age_penalty = -0.3 * (MIN_CULL_AGE - seed_age)
#        return age_penalty
```

### 7. Test Updates

**test_action_masking.py:**
- Update all references to MIN_CULL_AGE (3 → 10)
- Test that young seeds (age < 10) cannot be culled
- Test that mature seeds (age >= 10) can be culled

**test_simic_rewards.py:**
- Add test for information potential calculation
- Test that compute_seed_potential increases with seed_age
- Test that CULL penalty is larger for younger seeds (via PBRS)
- Verify GERMINATE-WAIT-CULL cycle is net negative (no farming exploit)

**test_simic_networks.py:**
- Test policy initialization bias is applied correctly
- Test WAIT logit starts higher than CULL logit

## Expected Behavior (Revised)

### Before Fix
- Agent has 50% chance of CULL every step after age 3
- P(survive to PROBATIONARY) ≈ 0.003%
- Takes ~200 episodes to learn WAIT is better

### After Fix
- Agent cannot CULL until age 10 (structural)
- Policy initialization biases toward WAIT (~62% initial probability)
- When CULL becomes available, younger seeds have higher information potential
- Culling at age 10 costs more (PBRS) than culling at age 15
- P(survive to PROBATIONARY) ≈ 0.5^5 ≈ 3.1% (1000x improvement)
- Agent learns lifecycle value much faster due to:
  - Higher survival rate
  - Clearer PBRS signal
  - Initial policy bias

## Interaction with Existing Rewards

This fix composes cleanly with existing reward structure:
- `training_bonus`, `blending_bonus`, `fossilized_bonus` unchanged
- `cull_promising_penalty` still applies based on improvement
- Stage-based PBRS still applies
- Information potential is additive and PBRS-compliant

## Rollback Plan

If this creates new exploits:
1. Keep MIN_CULL_AGE = 10 (specification fix, unlikely to cause issues)
2. Remove policy initialization bias (revert networks.py)
3. Reduce `info_potential` scaling factor from 1.0 to 0.5
4. As last resort, remove info_potential entirely, keep only structural fix

## Files Modified

1. `src/esper/simic/features.py` - MIN_CULL_AGE (3→10), add FULL_EVALUATION_AGE
2. `src/esper/simic/rewards.py` - Import constants, compute_seed_potential, _cull_shaping
3. `src/esper/simic/networks.py` - Add policy initialization bias
4. `tests/test_action_masking.py` - Update for new MIN_CULL_AGE
5. `tests/test_simic_rewards.py` - Add info potential tests, farming exploit test
6. `tests/test_simic_networks.py` - Add policy bias tests

## Diagnostics (Per DRL Expert)

Add logging to validate the fix:

1. **Survival curves:** Log `P(seed survives to age N)` per episode
2. **Stage completion rates:** Track `episodes_reaching_FOSSILIZED / total_episodes`
3. **CULL timing distribution:** Histogram of `seed_age at CULL`
4. **PBRS sanity check:** For complete episodes, verify PBRS telescoping
