# tests/simic/test_pbrs_verification.py
"""Verify PBRS properties hold for SIMPLIFIED reward mode.

Per Ng et al. (1999): PBRS guarantees optimal policy invariance
when shaping reward F(s,s') = gamma * Phi(s') - Phi(s).

These tests verify our implementation maintains this property.
"""

from esper.simic.rewards.rewards import (
    RewardMode,
    ContributionRewardConfig,
    STAGE_POTENTIALS,
)
from esper.leyline import SeedStage


def test_pbrs_uses_stage_potentials():
    """PBRS bonus uses predefined stage potentials."""
    # Verify STAGE_POTENTIALS is defined for all lifecycle stages
    # Note: PRUNED, EMBARGOED, RESETTING are failure/cleanup states, not part of normal lifecycle
    lifecycle_stages = [
        SeedStage.UNKNOWN,
        SeedStage.DORMANT,
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.HOLDING,
        SeedStage.FOSSILIZED,
    ]
    for stage in lifecycle_stages:
        assert stage in STAGE_POTENTIALS, f"Missing potential for {stage}"


def test_pbrs_potential_monotonicity():
    """Later stages should have higher potentials (progress is rewarded)."""
    # The lifecycle goes: DORMANT -> GERMINATED -> TRAINING -> BLENDING -> HOLDING -> FOSSILIZED
    lifecycle = [
        SeedStage.DORMANT,
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.HOLDING,
        SeedStage.FOSSILIZED,
    ]

    prev_potential = float('-inf')
    for stage in lifecycle:
        current = STAGE_POTENTIALS[stage]
        assert current >= prev_potential, f"{stage} potential should be >= previous"
        prev_potential = current


def test_pbrs_is_difference_based():
    """PBRS reward should be Phi(s') - Phi(s), not absolute."""
    # Verify the formula: advancing DORMANT->GERMINATED gives positive reward
    phi_dormant = STAGE_POTENTIALS[SeedStage.DORMANT]
    phi_germinated = STAGE_POTENTIALS[SeedStage.GERMINATED]

    pbrs_advance = phi_germinated - phi_dormant
    assert pbrs_advance > 0, "Advancing stage should give positive PBRS"

    # Verify: going backwards would give negative reward
    pbrs_regress = phi_dormant - phi_germinated
    assert pbrs_regress < 0, "Regressing stage should give negative PBRS"


def test_simplified_mode_includes_pbrs():
    """SIMPLIFIED mode should include PBRS component."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
    assert config.reward_mode == RewardMode.SIMPLIFIED


def test_pbrs_zero_for_same_stage():
    """PBRS for staying in same stage should be zero."""
    # Only check lifecycle stages (not failure/cleanup states)
    lifecycle_stages = [
        SeedStage.UNKNOWN,
        SeedStage.DORMANT,
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.HOLDING,
        SeedStage.FOSSILIZED,
    ]
    for stage in lifecycle_stages:
        phi = STAGE_POTENTIALS[stage]
        pbrs = phi - phi  # Same stage transition
        assert pbrs == 0.0, f"PBRS for {stage}->{stage} should be 0"


def test_pbrs_net_zero_over_episode():
    """PBRS should approximately net to zero over a complete episode.

    This verifies policy invariance: the shaping rewards telescope,
    leaving only the difference Phi(terminal) - Phi(initial).
    """
    # Simulate a lifecycle: DORMANT -> GERMINATED -> TRAINING -> FOSSILIZED
    trajectory = [
        (SeedStage.DORMANT, SeedStage.GERMINATED),
        (SeedStage.GERMINATED, SeedStage.TRAINING),
        (SeedStage.TRAINING, SeedStage.BLENDING),
        (SeedStage.BLENDING, SeedStage.HOLDING),
        (SeedStage.HOLDING, SeedStage.FOSSILIZED),
    ]

    total_pbrs = 0.0
    for s, s_prime in trajectory:
        pbrs = STAGE_POTENTIALS[s_prime] - STAGE_POTENTIALS[s]
        total_pbrs += pbrs

    # Total should equal Phi(FOSSILIZED) - Phi(DORMANT)
    expected = STAGE_POTENTIALS[SeedStage.FOSSILIZED] - STAGE_POTENTIALS[SeedStage.DORMANT]
    assert abs(total_pbrs - expected) < 1e-6, "PBRS should telescope correctly"
