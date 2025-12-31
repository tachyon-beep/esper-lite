# Proposed Fix: FOSSILIZE Action Incentive Rebalancing

**Date:** 2025-12-10
**Status:** APPROVED - DRL Expert Sign-off Received

## Problem Summary

The PPO policy has stopped using FOSSILIZE entirely because WAIT-farming in PROBATIONARY is more profitable:

| Strategy | Reward |
|----------|--------|
| WAIT × 4 epochs in PROBATIONARY | +28 to +32 (repeatable) |
| FOSSILIZE (good seed) | +0.8 one-shot, then nothing |

The current PROB penalty (-0.10 to -0.30) is noise compared to +7.5 attribution per step.

## Approved Changes

### Change 1: Add Terminal Bonus for Fossilized Seeds

**Status:** APPROVED (with scale adjustment 2.0 → 3.0)

**Rationale:** Make FOSSILIZE NPV-positive by deferring reward to episode end. The value function learns that states with more fossilized seeds have higher terminal value.

**Config addition in `ContributionRewardConfig`:**
```python
# Terminal bonus for fossilized seeds (incentivizes completion over farming)
# DRL Expert: increased from 2.0 to 3.0 to compete with post-scale attribution
fossilize_terminal_scale: float = 3.0
```

**Function signature update:**
```python
def compute_contribution_reward(
    action: IntEnum,
    seed_contribution: float | None,
    val_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: ContributionRewardConfig | None = None,
    acc_at_germination: float | None = None,
    acc_delta: float | None = None,
    return_components: bool = False,
    num_fossilized_seeds: int = 0,  # NEW: for terminal bonus
) -> float | tuple[float, RewardComponentsTelemetry]:
```

**Terminal bonus modification (section 5):**
```python
# === 5. TERMINAL BONUS ===
terminal_bonus = 0.0
if epoch == max_epochs:
    # Base accuracy bonus
    terminal_bonus = val_acc * config.terminal_acc_weight
    # Bonus per fossilized seed to incentivize completion over farming
    # This makes FOSSILIZE NPV-positive vs WAIT-farming in PROBATIONARY
    # DRL Expert review 2025-12-10: addresses H4 (terminating action problem)
    terminal_bonus += num_fossilized_seeds * config.fossilize_terminal_scale
    reward += terminal_bonus
```

### Change 2: Exponential PROB Penalty Escalation

**Status:** APPROVED (with steeper curve)

**Rationale:** Create urgency without punishing early information-gathering epochs. Must overcome +7.5 attribution to force decisions.

**DRL Expert recommended penalty schedule:**
- Epoch 1: 0 (grace period)
- Epoch 2: -1.0
- Epoch 3: -3.0
- Epoch 4: -6.0
- Epoch 5+: -10.0 (capped at clip boundary)

This ensures epoch 4+ has **net negative** reward for WAITing, forcing a decision.

**Code change (section 1c):**
```python
# === 1c. PROBATIONARY INDECISION PENALTY ===
# Exponential escalation for WAITing too long in PROBATIONARY
# Creates urgency to make FOSSILIZE/CULL decision before timeout
# DRL Expert review 2025-12-10: steepened to overcome +7.5 attribution
# Note: Orthogonal to blending_warning (different stage, different pathology)
probation_warning = 0.0
if seed_info is not None and seed_info.stage == STAGE_PROBATIONARY:
    if action_name == "WAIT":
        # Only penalize if counterfactual data is available (agent has info to act)
        # Grace period: epoch 1 is free (information gathering)
        if seed_info.epochs_in_stage >= 2:
            has_counterfactual = (
                seed_info.total_improvement is not None
                or seed_info.improvement_since_stage_start is not None
            )
            if has_counterfactual:
                # Exponential: epoch 2 -> -1.0, epoch 3 -> -3.0, epoch 4 -> -6.0
                epochs_waiting = seed_info.epochs_in_stage - 1
                probation_warning = -1.0 * (3 ** (epochs_waiting - 1))
                # Cap at -10.0 (clip boundary) to avoid extreme penalties
                probation_warning = max(probation_warning, -10.0)
                reward += probation_warning
```

### Change 3: Add Telemetry Fields

**Status:** APPROVED

**In `RewardComponentsTelemetry`:**
```python
num_fossilized_seeds: int = 0  # Current count for debugging
fossilize_terminal_bonus: float = 0.0  # Terminal bonus from fossilized seed count
```

## Expected Impact

### NPV Comparison After Fix

**WAIT-farm strategy (after both fixes):**
- Epoch 1: +7.5 attr - 0 penalty = +7.5
- Epoch 2: +7.5 attr - 1.0 penalty = +6.5
- Epoch 3: +7.5 attr - 3.0 penalty = +4.5
- Epoch 4: +7.5 attr - 6.0 penalty = +1.5
- Epoch 5: +7.5 attr - 10.0 penalty = -2.5 (NET NEGATIVE)
- Timeout: seed lost, no terminal bonus
- **Net per seed cycle: ~+17.5 (down from +28-32)**

**FOSSILIZE strategy:**
- FOSSILIZE action: +0.8 immediate (base + contribution scale)
- Terminal bonus: +3.0 at episode end
- Seed contributes to accuracy
- **Net per fossilization: ~+3.8 + accuracy contribution**

**With 5 fossilizations per episode:** +15.0 terminal bonus + accuracy improvement

### Credit Assignment

Handled naturally by existing mechanisms:
- **Compute rent:** Bad seeds add parameters → rent penalty
- **Accuracy terminal bonus:** Bad fossilizations hurt final accuracy
- **Fossilize shaping:** Already penalizes negative-improvement seeds
- **Ransomware mitigations:** Attribution discount + ratio penalty reduce rewards for gaming

## Interaction Analysis

### Terminal Bonus + Existing Fossilize Shaping
Combined for a 5% contribution seed fossilized at full legitimacy:
- Immediate: 0.5 + 0.5 = 1.0
- Terminal: 3.0
- **Net: +4.0** (competitive with farming)

### Exponential Penalty + Blending Warning
No problematic interaction - different stages, different triggers:
- `blending_warning`: "Your seed is hurting, consider CULL"
- `probation_warning`: "You have counterfactual data, make a decision"

### Changes + Ransomware Mitigations
Orthogonal fixes that compose correctly:
| Seed Type | Ransomware Signal | PROB Penalty | Outcome |
|-----------|-------------------|--------------|---------|
| Legitimate, slow | None | Full penalty | Forced decision |
| Ransomware | High discount + ratio penalty | Reduced (low attr) | Culled earlier |
| Legitimate, good | None | Forces fossilize | Correct |

## Monitoring Metrics

**Primary (must track):**
- `fossilize_action_rate`: Should increase from ~0% to 10-30%
- `avg_fossilizations_per_episode`: Target 3-5
- `prob_wait_epochs_before_decision`: Should decrease

**Red Flags:**
- `fossilize_rate > 50%`: Premature fossilization
- `avg_prob_epochs > 4`: Penalty not working
- `terminal_bonus = 0`: Not fossilizing at all

## Complementary Change (Training Config)

Increase entropy coefficient temporarily to ensure exploration:
```python
entropy_coef = 0.10  # Up from 0.05, decay to 0.02 after convergence
```

## DRL Expert Sign-off

| Change | Status | Notes |
|--------|--------|-------|
| **Change 1: Terminal Bonus** | APPROVED | Scale adjusted to 3.0 |
| **Change 2: Exponential PROB Penalty** | APPROVED | Curve steepened to (0, -1.0, -3.0, -6.0, -10.0) |
| **Change 3: Telemetry Fields** | APPROVED | Added num_fossilized_seeds field |

**Implementation Order:**
1. Implement reward scale fix first (if not done)
2. Implement this fossilize incentive fix with adjusted parameters
3. Monitor for 50-100 updates before further tuning

---
**Signed:** DRL Expert Review, 2025-12-10
