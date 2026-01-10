"""Potential-based shaping utilities for reward computation."""

from __future__ import annotations

from typing import Any

from esper.leyline import DEFAULT_GAMMA, SeedStage

# =============================================================================
# POTENTIAL-BASED REWARD SHAPING (PBRS) - DESIGN RATIONALE
# =============================================================================
#
# These values implement Ng et al. (1999) potential-based shaping:
#   F(s, s') = gamma * phi(s') - phi(s)
#
# KEY PROPERTIES MAINTAINED:
# 1. Bounded Effect: Adding PBRS to discounted returns affects value by exactly
#    gamma^T * phi(s_T) - phi(s_0), ensuring the optimal policy is preserved.
#    (The undiscounted sum of per-step PBRS bonuses differs from this value
#    when gamma < 1, but the effect on optimal actions is unchanged.)
#
# 2. Policy Invariance: Optimal policy unchanged by shaping (Ng et al., 1999).
#    Adding PBRS to any reward function preserves the optimal policy
#    because the shaping is purely potential-based.
#
# VALUE RATIONALE (actual values):
# - UNKNOWN (0.0): Fallback/error state - no reward
# - DORMANT (0.0): Baseline state before germination - no reward
# - GERMINATED (1.0): +1.0 for initiating growth
# - TRAINING (2.0): +1.0 for successful G1 gate passage
# - BLENDING (3.5): +1.5 (LARGEST delta) - critical integration phase
#   This is where value is actually created; alpha ramp merges seed contribution
# - HOLDING (5.5): +2.0 for stability validation (value 5 skipped)
# - FOSSILIZED (6.0): +0.5 (SMALLEST delta) - terminal bonus
#   Small to prevent "fossilization farming" (rushing to completion)
#
# TUNING HISTORY:
# - v1: Linear progression (1.0 increments each stage)
#       Problem: Insufficient BLENDING incentive; seeds stalled at TRAINING
# - v2: Current values with BLENDING emphasis (+1.5)
#       Result: Improved seed integration success rate
#
# VALIDATION:
# Property-based tests in tests/simic/properties/test_pbrs_properties.py verify:
# - Telescoping property holds for arbitrary stage sequences
# - Potentials are monotonically increasing toward FOSSILIZED
# - BLENDING has largest increment (value creation phase)
# - FOSSILIZED has smallest increment (anti-farming)
# - DORMANT/UNKNOWN have zero potential
#
# All reward functions MUST use this single STAGE_POTENTIALS dictionary.
# Using different potentials across reward functions breaks telescoping.
# =============================================================================

STAGE_POTENTIALS: dict[SeedStage, float] = {
    SeedStage.UNKNOWN: 0.0,
    SeedStage.DORMANT: 0.0,
    SeedStage.GERMINATED: 1.0,
    SeedStage.TRAINING: 2.0,
    SeedStage.BLENDING: 3.5,  # Largest increment - this is where value is created
    # Value 5 intentionally skipped (was SHADOWING, removed)
    SeedStage.HOLDING: 5.5,
    SeedStage.FOSSILIZED: 6.0,  # Smallest increment - not a farming target
    # Failure/recycling stages have zero potential (same as DORMANT)
    SeedStage.PRUNED: 0.0,
    SeedStage.EMBARGOED: 0.0,
    SeedStage.RESETTING: 0.0,
}


def compute_potential(val_acc: float, epoch: int, max_epochs: int) -> float:
    """Compute potential function for PBRS.

    Potential-based reward shaping (PBRS) guarantees that shaping
    does not change the optimal policy. The potential function
    should reflect how "good" a state is for future returns.

    Args:
        val_acc: Current validation accuracy
        epoch: Current epoch
        max_epochs: Maximum epochs

    Returns:
        Potential value
    """
    # Simple potential: higher accuracy = higher potential
    # Discounted by time remaining to encourage early improvement
    time_factor = (max_epochs - epoch) / max_epochs
    return float(val_acc * time_factor * 0.1)


def compute_pbrs_bonus(
    potential_prev: float,
    potential_next: float,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Compute potential-based reward shaping bonus.

    PBRS: F(s, s') = gamma * potential(s') - potential(s)

    This bonus can be added to any base reward without
    affecting the optimal policy (Ng et al., 1999).
    """
    return gamma * potential_next - potential_prev


def compute_seed_potential(obs: dict[str, Any]) -> float:
    """Compute potential value Phi(s) based on seed state.

    The potential captures the expected future value of having an active seed
    in various stages. This helps bridge the temporal gap where GERMINATE
    has negative immediate reward but high future value.

    Potential-based reward shaping: r' = r + gamma*Phi(s') - Phi(s)
    This preserves optimal policy (PBRS guarantee) while improving learning.

    Args:
        obs: Observation dictionary with REQUIRED keys:
            - has_active_seed: 0 or 1, whether a seed is active
            - seed_stage: int, current seed stage (see Note)
            - seed_epochs_in_stage: int, epochs in current stage

    Returns:
        Potential value for the current state

    Raises:
        KeyError: If any required key is missing from obs

    Note:
        seed_stage values match SeedStage enum from leyline:
        - DORMANT=1, GERMINATED=2, TRAINING=3, BLENDING=4
        - HOLDING=6, FOSSILIZED=7
    """
    has_active = obs["has_active_seed"]
    seed_stage = obs["seed_stage"]
    epochs_in_stage = obs["seed_epochs_in_stage"]

    # No potential for inactive seeds or DORMANT (stage 1)
    if not has_active or seed_stage <= 1:
        return 0.0

    # Use unified STAGE_POTENTIALS for PBRS consistency across all reward functions
    base_potential = STAGE_POTENTIALS[SeedStage(seed_stage)]

    # Progress bonus matches ContributionRewardConfig defaults for PBRS consistency
    # epoch_progress_bonus=0.3, max_progress_bonus=2.0
    progress_bonus = min(epochs_in_stage * 0.3, 2.0)

    return float(base_potential + progress_bonus)


__all__ = [
    "STAGE_POTENTIALS",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_seed_potential",
]
