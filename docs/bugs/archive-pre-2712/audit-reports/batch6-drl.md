# Batch 6 Deep RL Review: Simic Rewards

**Reviewer:** DRL Expert
**Date:** 2025-12-27
**Files Reviewed:**
- `/home/john/esper-lite/src/esper/simic/rewards/__init__.py`
- `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`
- `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`

---

## Executive Summary

This is **the most critical module for RL correctness** in the Esper system. The reward engineering here is sophisticated and addresses many known failure modes in RL training. The code demonstrates deep understanding of reward shaping theory (PBRS), anti-gaming mechanisms, and credit assignment challenges.

**Overall Assessment:** Strong implementation with correct PBRS theory, comprehensive anti-gaming defenses, and good test coverage. A few edge cases and theoretical concerns warrant attention.

### Key Strengths
1. **Correct PBRS implementation** with runtime gamma consistency check
2. **Comprehensive anti-ransomware defenses** (attribution discount, ratio penalty, fossilize shaping)
3. **Multi-modal reward support** (SHAPED, SPARSE, MINIMAL, SIMPLIFIED) for ablation studies
4. **Excellent telemetry integration** for reward debugging
5. **Property-based test coverage** for PBRS guarantees

### Areas of Concern
1. **P2:** Potential Goodhart risk in terminal bonus scaling
2. **P2:** PBRS telescoping approximation with gamma < 1
3. **P3:** Synergy bonus not included in anti-stacking protection
4. **P3:** `compute_loss_reward` action parameter is int, not LifecycleOp

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/simic/rewards/__init__.py`

**Purpose:** Public API for the rewards module. Clean re-export of all reward functions, configs, and internal helpers (for testing).

**Observations:**
- Exports internal helpers prefixed with `_` for testing - good practice
- Complete `__all__` declaration prevents namespace pollution
- Imports from both submodules correctly

**Findings:** None

---

### 2. `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`

**Purpose:** Core reward computation for Tamiyo seed lifecycle RL training. Implements PBRS-based dense shaping, counterfactual attribution, anti-gaming mechanisms, and multiple reward modes for experimentation.

#### 2.1 PBRS Implementation Analysis

**STAGE_POTENTIALS (lines 116-125):**
```python
STAGE_POTENTIALS = {
    0: 0.0,   # UNKNOWN
    1: 0.0,   # DORMANT
    2: 1.0,   # GERMINATED
    3: 2.0,   # TRAINING
    4: 3.5,   # BLENDING (largest increment)
    6: 5.5,   # HOLDING
    7: 6.0,   # FOSSILIZED (smallest increment)
}
```

**Verdict:** Well-designed. The comments (lines 70-114) demonstrate understanding of:
- Ng et al. (1999) policy invariance guarantees
- Anti-farming through small FOSSILIZED increment (+0.5)
- Value creation emphasis at BLENDING (+1.5)

**Concern (P2):** The comment claims "telescoping" but with gamma < 1 (0.995), the sum of PBRS bonuses does NOT exactly equal `gamma^T * phi(s_T) - phi(s_0)`. The per-step formula `gamma * phi(s') - phi(s)` has the gamma applied asymmetrically. The property tests use a relaxed tolerance (line 280: `tolerance = 2.0 + 1.0 * T`) acknowledging this limitation. This is mathematically correct behavior for gamma-discounted PBRS, but the comments should clarify that perfect telescoping only holds for gamma=1.

#### 2.2 Gamma Consistency Check (P0-class defense)

**Lines 498-502:**
```python
if config.gamma != DEFAULT_GAMMA:
    raise ValueError(
        f"PBRS gamma mismatch: config.gamma={config.gamma} != DEFAULT_GAMMA={DEFAULT_GAMMA}. "
        "This breaks policy invariance guarantees (Ng et al., 1999). Use DEFAULT_GAMMA from leyline."
    )
```

**Verdict:** Excellent. Runtime validation prevents accidental misconfiguration that would break PBRS guarantees. Using `ValueError` instead of `assert` ensures check runs even with `python -O`.

#### 2.3 Ransomware Detection Mechanisms

The code implements a sophisticated multi-layer defense against "ransomware seeds" - seeds that create structural dependencies without adding value:

1. **Attribution Discount (lines 534-537):**
   ```python
   if total_imp < 0:
       exp_arg = min(-config.attribution_sigmoid_steepness * total_imp, 700.0)
       attribution_discount = 1.0 / (1.0 + math.exp(exp_arg))
   ```
   Uses sigmoid discount for negative total_improvement. The steepness=3 is well-calibrated per comments (lines 217-223) to avoid penalizing normal training variance (0.1-0.3%).

2. **Ratio Penalty (lines 542-559):**
   Catches high contribution/improvement ratios (> 5x) as suspicious. Only fires when attribution_discount >= 0.5 to avoid penalty stacking.

3. **Fossilize Shaping (lines 1272-1330):**
   Ransomware check at lines 1295-1312 - seeds with negative total_improvement AND high contribution get extra penalty.

4. **Telemetry Integration (lines 1370-1457):**
   `_check_reward_hacking` and `_check_ransomware_signature` emit telemetry events for monitoring.

**Verdict:** Comprehensive defense against Goodhart's Law. The anti-stacking logic (ratio_penalty skipped when attribution_discount < 0.5) is particularly thoughtful.

#### 2.4 Credit Assignment Analysis

**PRUNE Attribution Inversion (lines 640-641):**
```python
if action == LifecycleOp.PRUNE:
    bounded_attribution = -bounded_attribution
```

**Verdict:** Correct. Without this, the policy would learn "PRUNE everything for +attribution rewards". The test at line 195 (`test_prune_good_seed_inverts_attribution`) validates this.

**FOSSILIZE Zero Attribution for Negative Delta (lines 632-634):**
```python
if action == LifecycleOp.FOSSILIZE and seed_info is not None:
    if seed_info.total_improvement < 0:
        bounded_attribution = 0.0
```

**Verdict:** Correct. Prevents double-counting where fossilize shaping already penalizes, avoiding reward leakage.

**Fossilized Seeds Zero Attribution (lines 523-524):**
```python
seed_is_fossilized = seed_info is not None and seed_info.stage == STAGE_FOSSILIZED
```

Used to skip attribution for permanent seeds. **Verdict:** Correct design - no decisions to be made for fossilized seeds.

#### 2.5 Terminal Bonus Analysis

**Lines 790-802:**
```python
if epoch == max_epochs and not config.disable_terminal_reward:
    terminal_bonus = val_acc * config.terminal_acc_weight
    fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
    terminal_bonus += fossilize_terminal_bonus
    reward += terminal_bonus
```

**Concern (P2 - Goodhart Risk):** The `fossilize_terminal_scale=3.0` creates an incentive to maximize fossilized count. While `num_contributing_fossilized` filters for seeds with `total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION` (1.0%), this threshold may be too low. A seed that barely passes the threshold (1.0% improvement) gets the same +3.0 terminal bonus as a seed with 10% improvement.

**Recommendation:** Consider scaling terminal bonus proportionally to total_improvement, e.g.:
```python
fossilize_terminal_bonus = sum(
    min(seed.total_improvement * config.fossilize_contribution_scale, config.fossilize_terminal_scale)
    for seed in contributing_fossilized
)
```

This would make the terminal bonus proportional to actual value added.

#### 2.6 Sparse/Minimal/Simplified Modes

**Lines 812-975:**
Well-implemented ablation modes for reward function experiments:

- **SPARSE:** Terminal-only reward. Correctly clamps base reward before scaling (H10 fix, lines 850-858).
- **MINIMAL:** Sparse + early-prune penalty. Simple but may be useful for baseline.
- **SIMPLIFIED:** DRL Expert recommended 3-component design (PBRS + intervention cost + terminal).

**Verdict:** Good experimentation infrastructure. The `RewardMode` enum provides clean switching.

#### 2.7 Loss-Primary Reward

**Lines 1570-1619:**
```python
def compute_loss_reward(
    action: int,  # <-- P3: Should be LifecycleOp
    ...
)
```

**Finding (P3):** The `action` parameter is typed as `int` while `compute_contribution_reward` uses `LifecycleOp`. This inconsistency could cause confusion. The function doesn't use the action parameter anyway (it's unused in the body).

**Lines 1602-1606:**
```python
growth_ratio = (total_params - host_params) / host_params
scaled_cost = math.log(1.0 + growth_ratio)
rent_penalty = config.compute_rent_weight * scaled_cost
```

**Verdict:** Logarithmic scaling is appropriate for rent penalty - prevents excessive penalty for large seeds while still penalizing bloat.

#### 2.8 Synergy Bonus

**Lines 1210-1233:**
```python
def _compute_synergy_bonus(
    interaction_sum: float,
    boost_received: float,
    synergy_weight: float = 0.1,
) -> float:
```

**Concern (P3):** The synergy bonus uses `interaction_sum` but the `boost_received` parameter is unused. Either:
1. It should be incorporated into the bonus, or
2. It should be removed from the signature

Also, the synergy bonus is added to reward (lines 714-720) but is NOT subject to the anti-stacking protection. A ransomware seed could potentially still receive synergy bonus even when attribution is zeroed.

**Lines 1227-1228:**
```python
if interaction_sum <= 0:
    return 0.0
```

Correctly gates on positive interaction only.

#### 2.9 Holding Indecision Penalty

**Lines 678-698:**
```python
if seed_info is not None and seed_info.stage == STAGE_HOLDING:
    if action == LifecycleOp.WAIT:
        if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
            has_counterfactual = (...)
            if has_counterfactual:
                epochs_waiting = seed_info.epochs_in_stage - 1
                holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
                holding_warning = max(holding_warning, -10.0)
```

**Verdict:** Exponential escalation (-1, -3, -9, -10) is aggressive but justified per the comment "to overcome +7.5 attribution". The `bounded_attribution > 0` gate correctly prevents stacking on ransomware seeds.

---

### 3. `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`

**Purpose:** Dataclass for reward component breakdown, enabling reward debugging and hacking detection.

#### 3.1 RewardComponentsTelemetry Dataclass

**Lines 13-58:**
Well-designed telemetry capturing all reward components:
- Primary signal (`seed_contribution`, `bounded_attribution`)
- Anti-gaming signals (`attribution_discount`, `ratio_penalty`)
- Penalties (`compute_rent`, `alpha_shock`, `blending_warning`, `holding_warning`)
- Bonuses (`pbrs_bonus`, `synergy_bonus`, `action_shaping`, `terminal_bonus`)
- Context fields for debugging

#### 3.2 shaped_reward_ratio Property

**Lines 60-100:**
```python
@property
def shaped_reward_ratio(self) -> float:
    """Fraction of total reward from shaping terms."""
    if abs(self.total_reward) < 1e-8:
        return 0.0
    shaped = (
        self.stage_bonus + self.pbrs_bonus + self.synergy_bonus + ...
    )
    return abs(shaped) / abs(self.total_reward)
```

**Verdict:** Excellent diagnostic metric. High values (> 0.5) indicate potential reward hacking where agent optimizes shaping bonuses rather than actual value.

**Minor (P4):** The `stage_bonus` field is always 0.0 (no code sets it), but it's included in the sum. Either:
1. Remove `stage_bonus` from the dataclass, or
2. Set it somewhere in `compute_contribution_reward`

#### 3.3 Performance Optimization

**Lines 102-138:**
```python
def to_dict(self) -> dict[str, float | int | str | None]:
    """Uses explicit dict construction instead of asdict() for 3-5x performance."""
    return {
        "base_acc_delta": self.base_acc_delta,
        ...
    }
```

**Verdict:** Good optimization for hot path. Manual dict construction avoids `dataclasses.asdict()` overhead.

---

## Cross-Cutting Integration Risks

### 1. PBRS Telescoping with Stage Transitions

The `_contribution_pbrs_bonus` function (lines 1165-1207) reconstructs previous state for PBRS calculation:

```python
if seed_info.epochs_in_stage == 0:
    # Just transitioned - use actual previous epoch count
    if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
        _logger.warning(
            "PBRS telescoping risk: transition from stage %d with previous_epochs_in_stage=0.",
            seed_info.previous_stage,
        )
```

**Risk:** If `SeedInfo` is constructed incorrectly (e.g., `previous_epochs_in_stage=0` when the seed actually spent time in previous stage), PBRS will underestimate `phi_prev` and over-reward the transition.

**Mitigation:** The warning log helps detect this. The property tests use relaxed tolerance acknowledging this limitation.

### 2. Counterfactual Signal Availability

The reward function handles three signal availability scenarios:
1. **Counterfactual available** (`seed_contribution is not None`): BLENDING+ stages
2. **Proxy signal** (`seed_contribution is None`, `seed_info is not None`): Pre-blending
3. **No signal** (`seed_info is None`): Seedless states

**Risk:** Transition from proxy to counterfactual at BLENDING entry could cause reward discontinuity if the signals disagree significantly.

**Mitigation:** The proxy uses lower weight (0.3x) and doesn't penalize negative deltas, providing softer gradient.

### 3. Reward Scale Consistency

Different reward modes have different scales:
- **SHAPED:** Can reach -40 to +40 (per property test bounds)
- **SPARSE:** [-scale, +scale] where scale defaults to 1.0
- **SIMPLIFIED:** PBRS (bounded) + terminal (up to ~9.0)

**Risk:** Switching reward modes mid-experiment could cause value function instability.

**Mitigation:** The ablation flags (`disable_pbrs`, `disable_terminal_reward`, `disable_anti_gaming`) allow incremental changes rather than full mode switch.

---

## Severity-Tagged Findings

### P0: Critical / Blocking
None identified.

### P1: Correctness Bugs
None identified.

### P2: Significant Issues

| ID | Location | Description | Recommendation |
|----|----------|-------------|----------------|
| P2-1 | rewards.py:790-802 | Terminal bonus gives same +3.0 for 1.0% improvement seed as 10% improvement seed | Scale terminal bonus proportionally to total_improvement |
| P2-2 | rewards.py:70-114 | PBRS "telescoping" comment implies exact cancellation but gamma < 1 breaks this | Clarify that gamma < 1 introduces bounded error; property tests acknowledge this |

### P3: Code Quality

| ID | Location | Description | Recommendation |
|----|----------|-------------|----------------|
| P3-1 | rewards.py:1570 | `compute_loss_reward` action param typed as `int` instead of `LifecycleOp` | Update type annotation for consistency |
| P3-2 | rewards.py:1210-1233 | `boost_received` parameter unused in `_compute_synergy_bonus` | Either incorporate into bonus or remove |
| P3-3 | rewards.py:714-720 | Synergy bonus not protected by anti-stacking logic | Consider gating synergy bonus on positive attribution_discount |
| P3-4 | reward_telemetry.py:37 | `stage_bonus` field never set | Either set it or remove from dataclass |

### P4: Style/Minor

| ID | Location | Description | Recommendation |
|----|----------|-------------|----------------|
| P4-1 | rewards.py:33 | `cast` imported but only used once (line 1131) | Minor, but could use `# type: ignore` instead |
| P4-2 | rewards.py:1623-1627 | `boost_received` unused in function signature | Remove or document why reserved |

---

## Theoretical Analysis

### PBRS Policy Invariance

The implementation correctly follows Ng et al. (1999):

**Theorem:** For any MDP M, adding potential-based shaping F(s, a, s') = gamma * phi(s') - phi(s) to the reward function preserves the optimal policy.

**Verification:**
1. `STAGE_POTENTIALS` defines phi(s) as state-dependent potential
2. `compute_pbrs_bonus` computes gamma * phi(s') - phi(s)
3. Runtime check enforces PBRS gamma == PPO gamma

**Limitation (acknowledged):** The epoch progress bonus adds a time-dependent component to potential. This is still valid PBRS (time is part of state), but the comment should clarify this.

### Credit Assignment Horizon

With 25-epoch episodes and gamma=0.995:
- gamma^25 = 0.882
- Effective horizon ~= 1/(1-gamma) = 200 timesteps

This is appropriate for the episode length. The sparse reward mode tests whether shaped rewards are necessary.

### Goodhart's Law Defenses

The code addresses multiple reward hacking vectors:

| Attack Vector | Defense | Effectiveness |
|---------------|---------|---------------|
| Ransomware seeds | Attribution discount sigmoid | Strong |
| Dependency gaming | Ratio penalty (> 5x) | Strong |
| Fossilize farming | Small FOSSILIZED increment, legitimacy discount | Moderate |
| WAIT farming | Holding indecision penalty (exponential) | Strong |
| Terminating action evasion | Terminal bonus for fossilized count | Moderate |

The "moderate" ratings reflect that determined adversarial training could potentially find edge cases. The telemetry hooks (`_check_reward_hacking`, `_check_ransomware_signature`) enable detection of such patterns.

---

## Test Coverage Assessment

The reward module has excellent property-based test coverage:

| Test File | Coverage Focus |
|-----------|----------------|
| `test_pbrs_properties.py` | Telescoping, monotonicity, gamma=1 exact cancellation |
| `test_reward_properties.py` | Bounds, finiteness, monotonicity |
| `test_reward_antigaming.py` | Ransomware detection, fossilize farming, ratio penalty |
| `test_reward_invariants.py` | Mathematical invariants |
| `test_reward_semantics.py` | Semantic correctness |
| `test_rewards.py` | Unit tests for all reward components |
| `test_reward_modes.py` | SPARSE/MINIMAL/SIMPLIFIED modes |

**Gap:** No explicit test for synergy bonus integration with anti-stacking logic.

---

## Recommendations Summary

1. **P2-1:** Scale terminal fossilize bonus by contribution magnitude to prevent marginal fossilizations being NPV-equivalent to high-value ones.

2. **P2-2:** Clarify PBRS telescoping comment to note gamma < 1 introduces bounded error (which is acceptable and tested).

3. **P3-3:** Add synergy bonus to anti-stacking protection (gate on `attribution_discount >= 0.5` or `bounded_attribution > 0`).

4. **P3-4:** Remove unused `stage_bonus` field from telemetry or set it appropriately.

5. **Future work:** Consider adding property test specifically for synergy bonus behavior with ransomware seeds.

---

## Conclusion

The reward engineering in this module is **production-quality**. The implementation demonstrates sophisticated understanding of RL reward design challenges including PBRS theory, credit assignment, and Goodhart-resistant shaping. The multi-modal reward support enables rigorous ablation studies.

The identified issues are all addressable without major refactoring. The comprehensive test suite provides confidence in correctness.

**Approval Status:** Ready for production with minor fixes to P3 items.
