"""Reward Computation for Tamiyo Seed Lifecycle Controller.

This module consolidates reward functions used across:
- Online PPO training (simic/ppo.py)
- Offline data generation (datagen/generate.py)
- Offline RL (simic/iql.py)

The reward design follows these principles:
1. Accuracy improvement is the primary signal
2. Lifecycle progression bonuses encourage exploration
3. Action-specific shaping guides decision making
4. Potential-based shaping maintains optimal policy invariance

Usage:
    from esper.simic.rewards import compute_shaped_reward, RewardConfig

    reward = compute_shaped_reward(
        action=ActionEnum.GERMINATE_CONV,  # Pass Enum member
        acc_delta=0.5,
        val_acc=65.0,
        seed_info=SeedInfo(...),
        epoch=10,
        max_epochs=25,
        action_enum=ActionEnum, # Required context
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from esper.leyline import SeedStage, MIN_CULL_AGE, MIN_PROBATION_EPOCHS
from esper.leyline.actions import is_germinate_action
from esper.simic.reward_telemetry import RewardComponentsTelemetry


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
# - SHADOWING (4.5): +1.0 for surviving blending without regression
# - PROBATIONARY (5.5): +1.0 for stability validation
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
# Property-based tests in tests/properties/test_pbrs_telescoping.py verify:
# - Telescoping property holds for arbitrary stage sequences
# - Potentials are monotonically increasing toward FOSSILIZED
# - BLENDING has largest increment (value creation phase)
# - FOSSILIZED has smallest increment (anti-farming)
# - DORMANT/UNKNOWN have zero potential
#
# All reward functions MUST use this single STAGE_POTENTIALS dictionary.
# Using different potentials across reward functions breaks telescoping.
# =============================================================================

STAGE_POTENTIALS = {
    0: 0.0,   # UNKNOWN
    1: 0.0,   # DORMANT
    2: 1.0,   # GERMINATED
    3: 2.0,   # TRAINING
    4: 3.5,   # BLENDING (largest increment - this is where value is created)
    5: 4.5,   # SHADOWING
    6: 5.5,   # PROBATIONARY
    7: 6.0,   # FOSSILIZED (smallest increment - not a farming target)
}

# Default discount factor for PBRS. All reward configs should use this value
# to ensure consistent telescoping properties across the codebase.
DEFAULT_GAMMA = 0.99


# =============================================================================
# Reward Configuration
# =============================================================================


@dataclass(slots=True)
class RewardConfig:
    """Configuration for reward computation.

    All weights are tunable hyperparameters. Default values
    are optimized for balanced exploration of seed lifecycle.
    """

    # Accuracy delta scaling
    acc_delta_weight: float = 2.0

    # Stage presence bonuses
    training_bonus: float = 0.2
    blending_bonus: float = 0.3
    fossilized_bonus: float = 0.5

    # Improvement-based bonuses
    stage_improvement_weight: float = 0.1
    blending_improvement_bonus: float = 0.2

    # Action-specific weights (immediate bonuses removed to prevent churn exploitation)
    germinate_no_seed_bonus: float = 0.0  # Was 0.3
    germinate_early_bonus: float = 0.0  # Was 0.2
    germinate_with_seed_penalty: float = -0.3

    advance_good_bonus: float = 0.5
    advance_premature_penalty: float = -0.2
    advance_blending_bonus: float = 0.4
    advance_no_seed_penalty: float = -0.2

    cull_failing_bonus: float = 0.3
    cull_acceptable_bonus: float = 0.15
    cull_promising_penalty: float = -0.3
    cull_no_seed_penalty: float = -0.2
    cull_param_recovery_weight: float = 0.1  # Bonus per 10K params recovered

    wait_plateau_penalty: float = -0.1
    wait_patience_bonus: float = 0.1
    wait_stagnant_penalty: float = -0.1

    # Terminal bonus
    terminal_acc_weight: float = 0.05

    # Thresholds
    early_epoch_fraction: float = 1 / 3  # "Early" = first third of training
    cull_failing_threshold: float = -1.0
    wait_patience_threshold: float = 0.5
    wait_stagnant_epochs: int = 5

    # Compute cost penalty (per-step rent on excess params)
    # Uses logarithmic scaling: rent = weight * log(1 + growth_ratio)
    # This provides diminishing marginal penalty, avoiding the cliff between
    # small seeds (2K params) and large seeds (74K params) that quadratic caused.
    compute_rent_weight: float = 0.5
    max_rent_penalty: float = 8.0  # Cap to prevent runaway negatives

    # PBRS scaling for seed-based potential shaping
    seed_potential_weight: float = 0.3

    @staticmethod
    def default() -> "RewardConfig":
        """Return default configuration."""
        return RewardConfig()


# =============================================================================
# Loss-Primary Reward Configuration (Phase 2)
# =============================================================================


# =============================================================================
# Contribution-Primary Reward Configuration (uses counterfactual validation)
# =============================================================================


@dataclass(slots=True)
class ContributionRewardConfig:
    """Configuration for contribution-primary reward computation.

    Uses counterfactual validation (seed_contribution) as the primary signal,
    eliminating heuristics that conflate host drift with seed impact.

    This is the recommended reward function when counterfactual validation
    is enabled in vectorized training.
    """

    # Primary signal: seed contribution weight
    contribution_weight: float = 3.0

    # PBRS stage progression
    pbrs_weight: float = 0.3
    epoch_progress_bonus: float = 0.3
    max_progress_bonus: float = 2.0

    # Compute rent (logarithmic scaling)
    rent_weight: float = 0.5
    max_rent: float = 8.0

    # Enforcement penalties (state machine compliance)
    invalid_fossilize_penalty: float = -1.0
    cull_fossilized_penalty: float = -1.0
    germinate_with_seed_penalty: float = -0.3

    # Intervention costs (action friction)
    germinate_cost: float = -0.02
    fossilize_cost: float = -0.01
    cull_cost: float = -0.005

    # Fossilize shaping
    fossilize_base_bonus: float = 0.5
    fossilize_contribution_scale: float = 0.1
    fossilize_noncontributing_penalty: float = -0.2

    # Cull shaping
    cull_hurting_bonus: float = 0.3
    cull_acceptable_bonus: float = 0.1
    cull_good_seed_penalty: float = -0.3
    cull_hurting_threshold: float = -0.5

    # Terminal bonus
    terminal_acc_weight: float = 0.05

    # Gamma for PBRS (uses module constant for consistency)
    gamma: float = DEFAULT_GAMMA

    @staticmethod
    def default() -> "ContributionRewardConfig":
        """Return default configuration."""
        return ContributionRewardConfig()


@dataclass(slots=True)
class LossRewardConfig:
    """Configuration for loss-primary reward computation.

    All weights are tunable hyperparameters optimized for
    cross-task comparability using normalized loss delta.
    """

    # Loss delta scaling
    loss_delta_weight: float = 5.0
    max_loss_delta: float = 5.0  # After normalization
    regression_penalty_scale: float = 0.5  # Asymmetric clipping
    typical_loss_delta_std: float = 0.1  # Task-specific normalization

    # Compute rent (logarithmic scaling like RewardConfig)
    compute_rent_weight: float = 0.05
    max_rent_penalty: float = 5.0
    grace_epochs: int = 3  # Rent-free grace period for new seeds

    # Stage bonuses (PBRS-compatible)
    stage_potential_weight: float = 0.1

    # Terminal bonus
    baseline_loss: float = 2.3  # Task-specific (random init loss)
    target_loss: float = 0.3  # Task-specific (achievable loss)
    terminal_loss_weight: float = 1.0

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss

    @staticmethod
    def default() -> "LossRewardConfig":
        return LossRewardConfig()

    @staticmethod
    def for_cifar10() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=2.3,  # ln(10)
            target_loss=0.3,
            typical_loss_delta_std=0.05,
        )

    @staticmethod
    def for_tinystories() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,
            typical_loss_delta_std=0.15,
            compute_rent_weight=0.01,
        )


# =============================================================================
# Lightweight Seed Info (for fast path)
# =============================================================================


class SeedInfo(NamedTuple):
    """Minimal seed information for reward computation.

    Designed to avoid importing heavy classes in the hot path.
    Stage values match SeedStage IntEnum:
        0=UNKNOWN, 1=DORMANT, 2=GERMINATED, 3=TRAINING,
        4=BLENDING, 5=SHADOWING, 6=PROBATIONARY, 7=FOSSILIZED, etc.
    """

    stage: int  # SeedStage.value
    improvement_since_stage_start: float
    total_improvement: float  # Since germination (for G5 gate alignment)
    epochs_in_stage: int
    seed_params: int = 0  # Trainable params of active seed
    previous_stage: int = 0  # For PBRS stage bonus calculation
    previous_epochs_in_stage: int = 0  # Epochs in previous stage at transition (for PBRS telescoping)
    seed_age_epochs: int = 0  # Total epochs since germination (for rent grace)

    @staticmethod
    def from_seed_state(seed_state, seed_params: int = 0) -> "SeedInfo | None":
        """Convert from kasmina.SeedState to SeedInfo.

        Args:
            seed_state: The seed state from kasmina, or None
            seed_params: Trainable parameter count of the active seed module

        Returns:
            SeedInfo or None if no seed state
        """
        if seed_state is None:
            return None
        metrics = seed_state.metrics
        improvement = 0.0
        total_improvement = 0.0
        seed_age = 0
        if metrics:
            improvement = metrics.current_val_accuracy - metrics.accuracy_at_stage_start
            total_improvement = metrics.total_improvement
            seed_age = metrics.epochs_total
        return SeedInfo(
            stage=seed_state.stage.value,
            improvement_since_stage_start=improvement,
            total_improvement=total_improvement,
            epochs_in_stage=seed_state.epochs_in_stage,
            seed_params=seed_params,
            previous_stage=seed_state.previous_stage.value,
            previous_epochs_in_stage=seed_state.previous_epochs_in_stage,
            seed_age_epochs=seed_age,
        )


# Stage constants from leyline contract
STAGE_GERMINATED = SeedStage.GERMINATED.value
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
STAGE_SHADOWING = SeedStage.SHADOWING.value
STAGE_PROBATIONARY = SeedStage.PROBATIONARY.value


# =============================================================================
# Core Reward Functions
# =============================================================================


def compute_shaped_reward(
    action: IntEnum,  # dynamic action enum member (e.g., WAIT, GERMINATE_X, FOSSILIZE, CULL)
    acc_delta: float,
    val_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: RewardConfig | None = None,
    action_enum: type | None = None,  # <--- NEW: Added this argument to fix TypeError
    return_components: bool = False,
) -> float | tuple[float, "RewardComponentsTelemetry"]:
    """Compute shaped reward for seed lifecycle control.

    This function is designed for high-throughput use:
    - Uses primitive types and NamedTuples (no heavy objects)
    - All configuration is explicit (no global state)
    - Zero allocations in hot path

    Args:
        action: Action taken (dynamic IntEnum member)
        acc_delta: Accuracy improvement this step
        val_acc: Current validation accuracy
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Extra params added (fossilized + active seed)
        host_params: Baseline host model params (for normalization)
        config: Reward configuration (uses default if None)
        action_enum: Optional Enum class for validating integer actions (legacy compat)
        return_components: If True, return (reward, RewardComponentsTelemetry) tuple

    Returns:
        Shaped reward value, or (reward, components) if return_components=True
    """
    from esper.simic.reward_telemetry import RewardComponentsTelemetry

    if config is None:
        config = _DEFAULT_CONFIG

    # Track components if requested
    components = RewardComponentsTelemetry() if return_components else None
    if components:
        # Populate diagnostic fields (same as compute_contribution_reward)
        components.action_name = action.name
        components.epoch = epoch
        components.seed_stage = seed_info.stage if seed_info else None
        components.val_acc = val_acc

    reward = 0.0

    # Base: accuracy improvement
    base_acc = acc_delta * config.acc_delta_weight
    reward += base_acc
    if components:
        components.base_acc_delta = base_acc

    # Compute rent: penalize excess params with logarithmic scaling
    # log(1 + growth_ratio) provides diminishing marginal penalty:
    # - Still penalizes large param counts
    # - But removes the 50x cliff between small (2K) and large (74K) seeds
    # - Allows conv_enhance seeds to be viable despite higher param count
    rent_penalty = 0.0
    growth_ratio = 0.0
    if host_params > 0 and total_params > 0:
        # total_params currently passed as added params; convert to growth ratio
        growth_ratio = total_params / host_params
        # Guard against negative ratios (defensive - shouldn't happen in practice)
        scaled_cost = math.log(1.0 + max(0.0, growth_ratio))
        rent_penalty = config.compute_rent_weight * scaled_cost
        rent_penalty = min(rent_penalty, config.max_rent_penalty)
        reward -= rent_penalty
    if components:
        components.compute_rent = -rent_penalty
        components.growth_ratio = growth_ratio

    # Lifecycle stage rewards
    stage_bonus = 0.0
    if seed_info is not None:
        stage = seed_info.stage
        improvement = seed_info.improvement_since_stage_start

        if stage == STAGE_TRAINING:
            stage_bonus += config.training_bonus
            if improvement > 0:
                stage_bonus += improvement * config.stage_improvement_weight

        elif stage == STAGE_BLENDING:
            stage_bonus += config.blending_bonus
            if acc_delta > 0:
                stage_bonus += config.blending_improvement_bonus

        elif stage == STAGE_FOSSILIZED:
            stage_bonus += config.fossilized_bonus

    reward += stage_bonus
    if components:
        components.stage_bonus = stage_bonus

    # Potential-based reward shaping for lifecycle progression
    # PBRS telescoping: F(s,a,s') = gamma * phi(s') - phi(s)
    # We must ensure phi(s') at timestep t equals phi(s) at timestep t+1.
    pbrs_bonus = 0.0
    if seed_info is not None:
        # Reconstruct previous timestep state using actual epoch counts
        if seed_info.epochs_in_stage == 0:
            # Just transitioned - use actual previous epoch count for correct telescoping
            prev_stage = seed_info.previous_stage
            prev_epochs = seed_info.previous_epochs_in_stage
        else:
            # Same stage, one fewer epoch
            prev_stage = seed_info.stage
            prev_epochs = seed_info.epochs_in_stage - 1

        current_obs = {
            "has_active_seed": 1,
            "seed_stage": seed_info.stage,
            "seed_epochs_in_stage": seed_info.epochs_in_stage,
        }
        prev_obs = {
            "has_active_seed": 1,
            "seed_stage": prev_stage,
            "seed_epochs_in_stage": prev_epochs,
        }
        phi_t = compute_seed_potential(current_obs)
        phi_t_prev = compute_seed_potential(prev_obs)
        pb_bonus = compute_pbrs_bonus(phi_t_prev, phi_t, gamma=DEFAULT_GAMMA)
        pbrs_bonus = config.seed_potential_weight * pb_bonus
        reward += pbrs_bonus
    if components:
        components.pbrs_bonus = pbrs_bonus

    # Action-specific shaping (semantic, enum-based)
    # We rely on action.name (IntEnum member)
    action_shaping = 0.0
    action_name = action.name
    if is_germinate_action(action):
        action_shaping = _germinate_shaping(seed_info, epoch, max_epochs, config)
    elif action_name == "FOSSILIZE":
        action_shaping = _advance_shaping(seed_info, config)
    elif action_name == "CULL":
        action_shaping = _cull_shaping(seed_info, config)
    elif action_name == "WAIT":
        action_shaping = _wait_shaping(seed_info, acc_delta, config)
    reward += action_shaping
    if components:
        components.action_shaping = action_shaping

    # Terminal bonus
    terminal_bonus = 0.0
    if epoch == max_epochs:
        terminal_bonus = val_acc * config.terminal_acc_weight
        reward += terminal_bonus
    if components:
        components.terminal_bonus = terminal_bonus
        components.total_reward = reward

    if return_components:
        return reward, components
    return reward


def _germinate_shaping(
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    config: RewardConfig,
) -> float:
    """Compute shaping for GERMINATE action.

    Includes PBRS bonus for successful germination to balance the
    PBRS penalty applied when culling seeds. Without this, germination
    has a net negative bias because:
    - Culling pays PBRS penalty: gamma * 0 - phi_current = -phi_current
    - Germinating had no PBRS bonus: 0 (asymmetric)

    Now germination receives: gamma * phi_germinated - 0 ≈ 1.98
    """
    if seed_info is None:
        # Bonus for germinating when no active seed
        bonus = config.germinate_no_seed_bonus
        # Extra bonus for germinating early
        if epoch < max_epochs * config.early_epoch_fraction:
            bonus += config.germinate_early_bonus

        # PBRS bonus for germination: gamma * phi(GERMINATED) - phi(no_seed)
        # This balances the PBRS penalty applied when culling seeds.
        # phi(no_seed) = 0, phi(GERMINATED) = 1.0 (from STAGE_POTENTIALS)
        # PBRS: 0.99 * 1.0 - 0.0 = 0.99
        germinated_obs = {
            "has_active_seed": 1,
            "seed_stage": STAGE_GERMINATED,
            "seed_epochs_in_stage": 0,
        }
        phi_germinated = compute_seed_potential(germinated_obs)
        phi_no_seed = 0.0
        pbrs_bonus = compute_pbrs_bonus(phi_no_seed, phi_germinated, gamma=DEFAULT_GAMMA)
        bonus += config.seed_potential_weight * pbrs_bonus

        return bonus
    else:
        # Penalty for trying to germinate with existing seed
        return config.germinate_with_seed_penalty


def _advance_shaping(seed_info: SeedInfo | None, config: RewardConfig) -> float:
    """Compute shaping for FOSSILIZE action.

    FOSSILIZE only succeeds from PROBATIONARY → FOSSILIZED (Leyline contract).
    Additionally, G5 gate requires total_improvement > 0 (since germination).

    Attempting FOSSILIZE at other stages is an INVALID action that will fail.
    We heavily penalize invalid attempts to teach the agent the state machine.
    """
    if seed_info is None:
        return config.advance_no_seed_penalty

    stage = seed_info.stage
    # Use total_improvement (since germination) to align with G5 gate criteria.
    # G5 gate checks: total_improvement > 0 AND is_healthy (currently always True).
    # This prevents rewarding attempts that will fail the gate.
    total_improvement = seed_info.total_improvement

    # Only reward FOSSILIZE at PROBATIONARY (the only valid source state)
    if stage == STAGE_PROBATIONARY:
        if total_improvement > 0:
            # Large immediate bonus for successful fossilization
            # Base bonus + improvement-scaled bonus
            fossilize_bonus = config.advance_good_bonus + 1.5  # Base 2.0 total
            improvement_bonus = 0.1 * total_improvement  # +0.1 per % improvement
            return fossilize_bonus + improvement_bonus
        # Failed fossilize: seed has no net improvement since germination.
        # G5 gate will reject this. Heavy penalty (-1.0) to discourage spam.
        # DRL expert recommendation: match INVALID action penalty magnitude.
        return -1.0

    # FOSSILIZE at earlier stages is INVALID - action will fail.
    # Heavy penalty to teach agent the state machine constraints.
    # Must be more expensive than CULL penalty so agent doesn't spam FOSSILIZE.
    if stage in (STAGE_TRAINING, STAGE_BLENDING, STAGE_SHADOWING):
        return -1.0  # Wasted action penalty (same as CULL from FOSSILIZED)

    # FOSSILIZED: FOSSILIZE is a no-op (already fossilized)
    if stage == STAGE_FOSSILIZED:
        return -0.5  # Mild penalty for redundant action

    return 0.0


def _cull_shaping(seed_info: SeedInfo | None, config: RewardConfig) -> float:
    """Compute shaping for CULL action.

    CULL is incentivized for failing seeds but FOSSILIZED seeds cannot be culled.
    Attempting to cull a FOSSILIZED seed is a wasted action (no-op) and penalized.

    Age penalty prevents "germinate then immediately cull" anti-pattern.
    PBRS penalty is scaled by seed health FOR LATE STAGES ONLY - this preserves
    full PBRS incentives for early-stage decisions while allowing exits from
    failing late-stage seeds.

    Note on PBRS Deviation (DRL Expert review):
        The health_factor scaling intentionally deviates from pure potential-based
        reward shaping (Ng et al., 1999) which would preserve optimal policy guarantees.
        We accept this deviation because:
        1. Failing seeds trapped in late stages represent a pathological case
        2. The 0.3 floor prevents gaming by intentionally tanking seeds
        3. Other shaping terms (base_shaping, param_recovery) are already non-PBRS
        4. Early stages (< BLENDING) retain full PBRS to preserve learning signal
    """
    if seed_info is None:
        return config.cull_no_seed_penalty

    improvement = seed_info.improvement_since_stage_start
    stage = seed_info.stage
    seed_params = seed_info.seed_params
    seed_age = seed_info.seed_age_epochs

    # FOSSILIZED seeds cannot be culled - they are permanent by design.
    # Attempting to cull is a wasted action. Heavy penalty to discourage.
    if stage == STAGE_FOSSILIZED:
        return -1.0  # Wasted action penalty

    # Age penalty: culling a very young seed wastes the germination investment.
    # This prevents the "germinate then immediately cull" anti-pattern.
    # Scale: -0.3 per epoch missing from minimum age
    if seed_age < MIN_CULL_AGE and stage in (STAGE_GERMINATED, STAGE_TRAINING):
        age_penalty = -0.3 * (MIN_CULL_AGE - seed_age)  # -0.9 at age 0, -0.6 at age 1
        return age_penalty  # Return early - don't add other bonuses to young culls

    # Base shaping: reward culling failing seeds, penalize culling promising ones
    if improvement < config.cull_failing_threshold:
        base_shaping = config.cull_failing_bonus
    elif improvement < 0:
        base_shaping = config.cull_acceptable_bonus
    else:
        # Scale penalty with improvement - culling +14% seed should hurt more than +1%
        improvement_penalty = -0.1 * max(0, improvement)
        # Scale penalty with stage - culling at SHADOWING hurts more than at TRAINING
        stage_penalty = -0.5 * max(0, stage - 3)  # TRAINING=3, so penalty starts at BLENDING
        base_shaping = config.cull_promising_penalty + improvement_penalty + stage_penalty

    # Param recovery bonus: incentivize culling bloated seeds to free resources
    # +0.1 per 10K params, capped at 0.5
    param_recovery_bonus = min(0.5, (seed_params / 10_000) * config.cull_param_recovery_weight)

    # Terminal PBRS correction: account for potential loss from destroying the seed
    current_obs = {
        "has_active_seed": 1,
        "seed_stage": stage,
        "seed_epochs_in_stage": seed_info.epochs_in_stage,
    }
    phi_current = compute_seed_potential(current_obs)

    # Health discount: failing seeds in LATE STAGES get reduced PBRS penalty
    # (DRL Expert recommendation: only apply for stage >= BLENDING to preserve
    # early-stage PBRS incentives where the penalty is smaller anyway)
    health_factor = 1.0
    if improvement < 0 and stage >= STAGE_BLENDING:
        # Scale from 1.0 (improvement=0) to 0.3 (improvement=-3 or worse)
        health_factor = max(0.3, 1.0 + improvement / 3.0)

    # PBRS: gamma * phi(next) - phi(current) where next = no seed (phi=0)
    pbrs_correction = 0.99 * 0.0 - phi_current  # = -phi_current
    terminal_pbrs = config.seed_potential_weight * pbrs_correction * health_factor

    return base_shaping + param_recovery_bonus + terminal_pbrs


def _wait_shaping(
    seed_info: SeedInfo | None,
    acc_delta: float,
    config: RewardConfig,
) -> float:
    """Compute shaping for WAIT action.

    WAIT is the correct default action during mechanical stages (TRAINING,
    BLENDING, SHADOWING) because Kasmina auto-advances through these stages.
    Tamiyo's job is to "not cull" unless something is clearly failing.
    """
    if seed_info is None:
        # Penalize waiting when plateauing with no seed
        if acc_delta < 0.5:
            return config.wait_plateau_penalty
        return 0.0

    stage = seed_info.stage
    improvement = seed_info.improvement_since_stage_start
    epochs_in_stage = seed_info.epochs_in_stage

    # MECHANICAL STAGES: WAIT is the correct default action.
    # Auto-advance handles progression; agent should only intervene (CULL) if failing.
    if stage == STAGE_TRAINING:
        if improvement > config.wait_patience_threshold:
            return config.wait_patience_bonus
        elif epochs_in_stage > config.wait_stagnant_epochs:
            return config.wait_stagnant_penalty
        # Neutral otherwise - WAIT is still correct, just no bonus

    # BLENDING: WAIT is correct. Agent cannot accelerate alpha ramping.
    # Mirror TRAINING logic but without stagnant penalty (progress is deterministic).
    if stage == STAGE_BLENDING:
        if improvement > config.wait_patience_threshold:
            return config.wait_patience_bonus
        return 0.0  # WAIT is correct, no penalty for slow blending

    # SHADOWING/PROBATIONARY: Auto-advance stages where WAIT is strongly encouraged.
    # These are "hands off" stages - agent should wait for mechanical completion.
    if stage in (STAGE_SHADOWING, STAGE_PROBATIONARY):
        if improvement > 0:
            return config.wait_patience_bonus  # Reward patience with good seeds
        return 0.0  # Still correct to WAIT even without improvement

    return 0.0


# Default config singleton (avoid repeated allocations)
_DEFAULT_CONFIG = RewardConfig()
_DEFAULT_CONTRIBUTION_CONFIG = ContributionRewardConfig()


# =============================================================================
# Contribution-Primary Reward (uses counterfactual validation)
# =============================================================================


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
    return_components: bool = False,
) -> float | tuple[float, RewardComponentsTelemetry]:
    """Compute reward using bounded attribution (ransomware-resistant).

    This function uses counterfactual validation but prevents "ransomware"
    rewards where seeds create structural dependencies that inflate their
    apparent contribution beyond actual value added.

    Key insight: A seed cannot create more value than the progress actually
    observed. The counterfactual measures "removal cost" not "value added".

    Solution: reward = min(progress, contribution)
    - If seed_contribution < 0: Toxic seed, penalize
    - If seed_contribution > progress: Dependency created, pay for progress only
    - If progress > seed_contribution: Host learning inflated progress, pay for contribution

    Args:
        action: Action taken (IntEnum member)
        seed_contribution: Counterfactual delta (real_acc - baseline_acc).
                          None if alpha=0 or no seed.
        val_acc: Current validation accuracy
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Extra params added (fossilized + active seed)
        host_params: Baseline host model params (for normalization)
        config: Reward configuration (uses default if None)
        acc_at_germination: Accuracy when seed was planted (for progress calc)
        return_components: If True, return (reward, components) tuple

    Returns:
        Shaped reward value, or (reward, components) if return_components=True
    """
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    # Track components if requested (no import needed - already at module level)
    components = RewardComponentsTelemetry() if return_components else None
    if components:
        components.action_name = action.name
        components.epoch = epoch
        components.seed_stage = seed_info.stage if seed_info else None
        # DRL Expert recommended diagnostic fields
        components.val_acc = val_acc
        components.acc_at_germination = acc_at_germination

    reward = 0.0

    # === 1. PRIMARY: Bounded Attribution (Ransomware-Resistant) ===
    # Pay for min(progress, contribution) to prevent dependency exploitation
    bounded_attribution = 0.0
    progress = None
    if seed_contribution is not None:
        if seed_contribution < 0:
            # Toxic seed - counterfactual shows it actively hurts
            # Pay the negative contribution as penalty
            bounded_attribution = config.contribution_weight * seed_contribution
        else:
            # Positive contribution - bound by actual progress
            if acc_at_germination is not None:
                progress = val_acc - acc_at_germination
                # Pay for whichever is SMALLER:
                # - Prevents ransomware (high contribution, low progress)
                # - Prevents free-riding (high progress, low contribution)
                attributed = min(max(0.0, progress), seed_contribution)
            else:
                # No baseline available - discount contribution
                attributed = seed_contribution * 0.5
            bounded_attribution = config.contribution_weight * attributed
        reward += bounded_attribution

    if components:
        components.seed_contribution = seed_contribution
        components.bounded_attribution = bounded_attribution
        components.progress_since_germination = progress

    # === 2. PBRS: Stage Progression ===
    # Potential-based shaping preserves optimal policy (Ng et al., 1999)
    pbrs_bonus = 0.0
    if seed_info is not None:
        pbrs_bonus = _contribution_pbrs_bonus(seed_info, config)
        reward += pbrs_bonus
    if components:
        components.pbrs_bonus = pbrs_bonus

    # === 3. RENT: Compute Cost ===
    # Logarithmic penalty on parameter bloat
    rent_penalty = 0.0
    growth_ratio = 0.0
    if host_params > 0 and total_params > 0:
        growth_ratio = total_params / host_params
        # Guard against negative ratios (defensive - shouldn't happen in practice)
        scaled_cost = math.log(1.0 + max(0.0, growth_ratio))
        rent_penalty = min(config.rent_weight * scaled_cost, config.max_rent)
        reward -= rent_penalty
    if components:
        components.compute_rent = -rent_penalty  # Negative because it's a penalty
        components.growth_ratio = growth_ratio  # DRL Expert diagnostic field

    # === 4. ACTION SHAPING ===
    # Minimal - just state machine enforcement and intervention costs
    action_shaping = 0.0
    action_name = action.name

    if is_germinate_action(action):
        if seed_info is not None:
            action_shaping += config.germinate_with_seed_penalty
        else:
            # PBRS bonus for successful germination (no existing seed)
            # Balances the PBRS penalty applied when culling seeds
            phi_germinated = STAGE_POTENTIALS.get(STAGE_GERMINATED, 0.0)
            phi_no_seed = 0.0
            pbrs_germinate = config.gamma * phi_germinated - phi_no_seed
            action_shaping += config.pbrs_weight * pbrs_germinate
        action_shaping += config.germinate_cost

    elif action_name == "FOSSILIZE":
        action_shaping += _contribution_fossilize_shaping(seed_info, seed_contribution, config)
        action_shaping += config.fossilize_cost

    elif action_name == "CULL":
        action_shaping += _contribution_cull_shaping(seed_info, seed_contribution, config)
        action_shaping += config.cull_cost

    # WAIT: No additional shaping (correct default action)

    reward += action_shaping
    if components:
        components.action_shaping = action_shaping

    # === 5. TERMINAL BONUS ===
    terminal_bonus = 0.0
    if epoch == max_epochs:
        terminal_bonus = val_acc * config.terminal_acc_weight
        reward += terminal_bonus
    if components:
        components.terminal_bonus = terminal_bonus

    if components:
        components.total_reward = reward
        return reward, components
    return reward


def _contribution_pbrs_bonus(
    seed_info: SeedInfo,
    config: ContributionRewardConfig,
) -> float:
    """Compute PBRS bonus for stage progression.

    Uses flattened potentials to prevent fossilization farming while
    still incentivizing lifecycle progression.

    PBRS telescoping property: F(s,a,s') = gamma * phi(s') - phi(s)
    Over a trajectory, intermediate potentials cancel, so we must ensure
    phi(s') at timestep t equals phi(s) at timestep t+1.
    """
    # Current potential
    phi_current = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    phi_current += min(
        seed_info.epochs_in_stage * config.epoch_progress_bonus,
        config.max_progress_bonus,
    )

    # Previous potential (reconstruct previous state)
    if seed_info.epochs_in_stage == 0:
        # Just transitioned - use actual previous epoch count for correct telescoping
        phi_prev = STAGE_POTENTIALS.get(seed_info.previous_stage, 0.0)
        phi_prev += min(
            seed_info.previous_epochs_in_stage * config.epoch_progress_bonus,
            config.max_progress_bonus,
        )
    else:
        # Same stage, one fewer epoch
        phi_prev = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
        phi_prev += min(
            (seed_info.epochs_in_stage - 1) * config.epoch_progress_bonus,
            config.max_progress_bonus,
        )

    return config.pbrs_weight * (config.gamma * phi_current - phi_prev)


def _contribution_fossilize_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for FOSSILIZE action - with legitimacy discount.

    Rapid fossilization (short PROBATIONARY period) is discounted to prevent
    dependency gaming where seeds create artificial dependencies during
    BLENDING/SHADOWING that inflate metrics.
    """
    if seed_info is None:
        return config.invalid_fossilize_penalty

    # FOSSILIZE only valid from PROBATIONARY
    if seed_info.stage != STAGE_PROBATIONARY:
        return config.invalid_fossilize_penalty

    # Legitimacy discount: must have spent time in PROBATIONARY to earn full bonus
    # This prevents rapid fossilization gaming
    legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_PROBATION_EPOCHS)

    # Use seed_contribution to determine if fossilization is earned
    if seed_contribution is not None and seed_contribution > 0:
        # Bonus scales with actual contribution AND legitimacy
        base_bonus = (
            config.fossilize_base_bonus
            + config.fossilize_contribution_scale * seed_contribution
        )
        return base_bonus * legitimacy_discount

    # Non-contributing or no counterfactual - penalty (no discount on penalties)
    return config.fossilize_noncontributing_penalty


def _contribution_cull_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for CULL action - simplified with counterfactual."""
    if seed_info is None:
        return -0.2  # Nothing to cull

    if seed_info.stage == STAGE_FOSSILIZED:
        return config.cull_fossilized_penalty

    # Age gate: penalize culling very young seeds
    if seed_info.seed_age_epochs < MIN_CULL_AGE:
        return -0.3 * (MIN_CULL_AGE - seed_info.seed_age_epochs)

    # Use seed_contribution if available (BLENDING+ stages)
    if seed_contribution is not None:
        if seed_contribution < config.cull_hurting_threshold:
            return config.cull_hurting_bonus  # Good: seed was hurting
        elif seed_contribution < 0:
            return config.cull_acceptable_bonus  # Acceptable: marginal harm
        else:
            # Penalize culling good seeds, scaled by how good
            return config.cull_good_seed_penalty - 0.05 * seed_contribution

    # No counterfactual (TRAINING stage) - neutral
    # We don't have information yet, so CULL is neither good nor bad
    return 0.0


# =============================================================================
# Potential-Based Shaping (for offline RL compatibility)
# =============================================================================


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
    return val_acc * time_factor * 0.1


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


def compute_seed_potential(obs: dict) -> float:
    """Compute potential value Phi(s) based on seed state.

    The potential captures the expected future value of having an active seed
    in various stages. This helps bridge the temporal gap where GERMINATE
    has negative immediate reward but high future value.

    Potential-based reward shaping: r' = r + gamma*Phi(s') - Phi(s)
    This preserves optimal policy (PBRS guarantee) while improving learning.

    Args:
        obs: Observation dictionary with has_active_seed, seed_stage, seed_epochs_in_stage

    Returns:
        Potential value for the current state

    Note:
        seed_stage values match SeedStage enum from leyline:
        - DORMANT=1, GERMINATED=2, TRAINING=3, BLENDING=4
        - SHADOWING=5, PROBATIONARY=6, FOSSILIZED=7
    """
    has_active = obs.get("has_active_seed", 0)
    seed_stage = obs.get("seed_stage", 0)
    epochs_in_stage = obs.get("seed_epochs_in_stage", 0)

    # No potential for inactive seeds or DORMANT (stage 1)
    if not has_active or seed_stage <= 1:
        return 0.0

    # Use unified STAGE_POTENTIALS for PBRS consistency across all reward functions
    base_potential = STAGE_POTENTIALS.get(seed_stage, 0.0)

    # Progress bonus matches ContributionRewardConfig defaults for PBRS consistency
    # epoch_progress_bonus=0.3, max_progress_bonus=2.0
    progress_bonus = min(epochs_in_stage * 0.3, 2.0)

    return base_potential + progress_bonus


# =============================================================================
# Loss-Primary Reward (Phase 2)
# =============================================================================


def compute_pbrs_stage_bonus(
    seed_info: SeedInfo,
    config: LossRewardConfig,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """PBRS-compatible stage bonus using potential function.

    Uses unified STAGE_POTENTIALS for consistency across all reward functions.
    """
    previous_stage = seed_info.previous_stage

    current_potential = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    previous_potential = STAGE_POTENTIALS.get(previous_stage, 0.0)

    return config.stage_potential_weight * (
        gamma * current_potential - previous_potential
    )


def compute_loss_reward(
    action: int,
    loss_delta: float,
    val_loss: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: LossRewardConfig | None = None,
) -> float:
    """Compute loss-primary reward for seed lifecycle control."""
    if config is None:
        config = LossRewardConfig.default()

    reward = 0.0

    # Primary: loss improvement (negative delta = improvement)
    normalized_delta = loss_delta / config.typical_loss_delta_std
    clipped = max(-config.max_loss_delta, min(normalized_delta, config.max_loss_delta))
    if clipped > 0:
        clipped *= config.regression_penalty_scale
    reward += (-clipped) * config.loss_delta_weight

    # Compute rent with grace period (logarithmic scaling)
    if host_params > 0 and total_params > 0:
        in_grace = False
        if seed_info is not None:
            in_grace = seed_info.seed_age_epochs < config.grace_epochs
        if not in_grace:
            growth_ratio = total_params / host_params
            # Guard against negative ratios (defensive - shouldn't happen in practice)
            scaled_cost = math.log(1.0 + max(0.0, growth_ratio))
            rent_penalty = config.compute_rent_weight * scaled_cost
            rent_penalty = min(rent_penalty, config.max_rent_penalty)
            reward -= rent_penalty

    # Stage bonuses (PBRS)
    if seed_info is not None:
        reward += compute_pbrs_stage_bonus(seed_info, config)

    # Terminal bonus based on normalized improvement
    if epoch == max_epochs:
        improvement = config.baseline_loss - val_loss
        achievable_range = config.achievable_range or 1.0
        normalized = max(0.0, min(improvement / achievable_range, 1.0))
        reward += normalized * config.terminal_loss_weight

    return reward


# =============================================================================
# Intervention Costs
# =============================================================================

GERMINATE_INTERVENTION_COST = -0.02

INTERVENTION_COSTS_BY_NAME = {
    "WAIT": 0.0,
    "FOSSILIZE": -0.01,
    "CULL": -0.005,
}


def get_intervention_cost(action: IntEnum) -> float:
    """Get intervention cost for an action.

    Small negative costs discourage unnecessary interventions,
    encouraging the agent to only act when beneficial.
    """
    if is_germinate_action(action):
        return GERMINATE_INTERVENTION_COST
    return INTERVENTION_COSTS_BY_NAME.get(action.name, 0.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    "RewardConfig",
    "LossRewardConfig",
    "ContributionRewardConfig",
    # Seed info
    "SeedInfo",
    # Reward functions
    "compute_shaped_reward",  # Legacy: uses acc_delta (conflated signal)
    "compute_contribution_reward",  # NEW: uses seed_contribution (causal)
    "compute_loss_reward",
    # PBRS utilities
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_seed_potential",
    # Intervention costs
    "get_intervention_cost",
    "INTERVENTION_COSTS_BY_NAME",
    # Stage constants and PBRS configuration
    "DEFAULT_GAMMA",
    "STAGE_POTENTIALS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_SHADOWING",
    "STAGE_PROBATIONARY",
]
