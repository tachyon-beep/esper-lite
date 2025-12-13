"""Reward Computation for Tamiyo Seed Lifecycle Controller.

This module consolidates reward functions used across:
- Online PPO training (simic/ppo.py)
- Offline data generation (datagen/generate.py)
- Offline RL (simic/iql.py)

The reward design follows these principles:
1. Counterfactual validation is the primary signal (seed_contribution)
2. Lifecycle progression bonuses encourage exploration (PBRS)
3. Compute rent penalizes parameter bloat
4. Potential-based shaping maintains optimal policy invariance

Usage:
    from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig

    reward = compute_contribution_reward(
        action=ActionEnum.GERMINATE_CONV,
        seed_contribution=0.05,
        val_acc=65.0,
        seed_info=SeedInfo(...),
        epoch=10,
        max_epochs=25,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

_logger = logging.getLogger(__name__)

from esper.leyline import SeedStage, MIN_CULL_AGE, MIN_PROBATION_EPOCHS
from esper.kasmina.slot import MIN_FOSSILIZE_CONTRIBUTION
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.reward_telemetry import RewardComponentsTelemetry


def _is_germinate_action(action) -> bool:
    """Check if action is a germinate action.

    Handles both LifecycleOp (factored) and flat action enums (heuristic baseline).
    """
    # Factored action: LifecycleOp.GERMINATE
    if isinstance(action, LifecycleOp):
        return action == LifecycleOp.GERMINATE
    # Flat action: action.name starts with "GERMINATE_"
    return action.name.startswith("GERMINATE_")


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
# - (5 was SHADOWING, now unused - kept for serialization)
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
    5: 4.5,   # (was SHADOWING - kept for serialization, unused in new lifecycle)
    6: 5.5,   # PROBATIONARY
    7: 6.0,   # FOSSILIZED (smallest increment - not a farming target)
}

# Default discount factor for PBRS. MUST match SimicConfig.gamma (0.995)!
# PBRS theory requires gamma_pbrs == gamma_ppo for policy invariance.
# If they differ, reward shaping can change the optimal policy.
# This value is optimized for 25-epoch episodes: gamma^25 ~ 0.88
DEFAULT_GAMMA = 0.995


# =============================================================================
# Reward Configuration
# =============================================================================


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
    # Reduced from 3.0 to 1.0 for stable PPO (per-step rewards should be in [-10, +10])
    contribution_weight: float = 1.0

    # Proxy signal for pre-blending stages (when counterfactual unavailable)
    # Proportionally reduced from 1.0 to 0.3 (maintains 3:1 ratio with contribution_weight)
    proxy_contribution_weight: float = 0.3

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
    # Terminal bonus per fossilized seed (incentivizes completion over farming)
    # DRL Expert review 2025-12-10: set to 3.0 to compete with post-scale attribution
    fossilize_terminal_scale: float = 3.0

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

    # Compute rent (logarithmic scaling)
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
        4=BLENDING, 5=(deprecated SHADOWING), 6=PROBATIONARY, 7=FOSSILIZED, etc.
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
STAGE_PROBATIONARY = SeedStage.PROBATIONARY.value


# =============================================================================
# Core Reward Functions
# =============================================================================


# Default config singleton (avoid repeated allocations)
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
    acc_delta: float | None = None,
    return_components: bool = False,
    num_fossilized_seeds: int = 0,
    num_contributing_fossilized: int = 0,
) -> float | tuple[float, RewardComponentsTelemetry]:
    """Compute reward using bounded attribution (ransomware-resistant).

    This function uses counterfactual validation but prevents "ransomware"
    rewards where seeds create structural dependencies that inflate their
    apparent contribution beyond actual value added.

    For pre-blending stages where counterfactual is unavailable, uses acc_delta
    as a proxy signal with lower weight to maintain reward continuity.

    Key insight: A seed cannot create more value than the progress actually
    observed. The counterfactual measures "removal cost" not "value added".

    Attribution logic:
    - If seed_contribution < 0: Toxic seed, penalize
    - If seed_contribution >= progress: High causal, use geometric mean
    - If seed_contribution < progress: Low causal, cap at contribution
    - If seed_contribution is None: Use acc_delta as proxy (pre-blending)

    Args:
        action: Action taken (IntEnum member)
        seed_contribution: Counterfactual delta (real_acc - baseline_acc).
                          None if alpha=0 or no seed (pre-blending stages).
        val_acc: Current validation accuracy
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Extra params added (fossilized + active seed)
        host_params: Baseline host model params (for normalization)
        config: Reward configuration (uses default if None)
        acc_at_germination: Accuracy when seed was planted (for progress calc)
        acc_delta: Per-epoch accuracy change (proxy signal for pre-blending)
        return_components: If True, return (reward, components) tuple
        num_fossilized_seeds: Count of all fossilized seeds (for telemetry)
        num_contributing_fossilized: Count of fossilized seeds with total_improvement >= MIN_FOSSILIZE_CONTRIBUTION.
            Only these seeds receive terminal bonus. This prevents bad fossilizations from being NPV-positive.

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

    # === 1. PRIMARY: Contribution or Proxy Signal ===
    # Uses counterfactual when available, falls back to acc_delta proxy for pre-blending
    bounded_attribution = 0.0
    progress = None

    # Skip attribution for FOSSILIZED seeds - no decision to be made, seed is permanent
    # Without this check, fossilized seeds continue generating high rewards indefinitely
    seed_is_fossilized = seed_info is not None and seed_info.stage == STAGE_FOSSILIZED

    if seed_contribution is not None and not seed_is_fossilized:
        # Counterfactual available (BLENDING+ stages)
        if seed_contribution < 0:
            # Toxic seed - counterfactual shows it actively hurts
            # Pay the negative contribution as penalty
            bounded_attribution = config.contribution_weight * seed_contribution
        else:
            # Positive contribution - asymmetric handling preserves causal signal
            if acc_at_germination is not None:
                progress = val_acc - acc_at_germination
                if progress <= 0:
                    # Anti-ransomware: no reward without actual progress
                    attributed = 0.0
                elif seed_contribution >= progress:
                    # High causal, low progress: timing mismatch, seed is valuable
                    # Geometric mean recovers signal: sqrt(5% * 47%) = 15.3% vs min = 5%
                    attributed = math.sqrt(progress * seed_contribution)
                else:
                    # Low causal, high progress: host did the work
                    # Cap at actual contribution to prevent free-riding
                    attributed = seed_contribution
            else:
                # No baseline available - discount contribution
                attributed = seed_contribution * 0.5

            # === ATTRIBUTION DISCOUNT: Reduce rewards when total trajectory is negative ===
            # Prevents rewarding seeds that show good per-step counterfactual but
            # have negative total_improvement (the ransomware buildup pattern).
            # Sigmoid smoothly transitions: ~0.5 at total_imp=0, →0 as total_imp→-∞
            attribution_discount = 1.0
            if seed_info is not None:
                total_imp = seed_info.total_improvement
                if total_imp < 0:
                    # Discount factor: 1/(1 + e^(-10*x)) gives smooth 0→1 transition
                    # Steepened from -5 to -10 per DRL Expert review (2025-12-10)
                    # to reduce reward leakage for ransomware seeds
                    # At total_imp=-0.2%, discount ≈ 0.12 (was 0.27 at -5)
                    # At total_imp=-0.5%, discount ≈ 0.007 (was 0.076 at -5)
                    # At total_imp=-1.0%, discount ≈ 0.00005 (essentially zero)
                    attribution_discount = 1.0 / (1.0 + math.exp(-10 * total_imp))
                    attributed *= attribution_discount
            if components:
                components.attribution_discount = attribution_discount

            # === RANSOMWARE RATIO PENALTY ===
            # High causal contribution with low/negative improvement = structural entanglement
            # This directly penalizes the ransomware signature where seeds create
            # dependencies that inflate their apparent value beyond actual contribution.
            # DRL Expert review 2025-12-10: targets conv_heavy pattern specifically.
            #
            # IMPORTANT: Skip when attribution_discount < 0.5 to avoid penalty stacking.
            # The attribution discount already zeros rewards for ransomware seeds;
            # ratio_penalty is only for edge cases where discount alone is insufficient
            # (e.g., high contribution with small positive improvement near threshold).
            ratio_penalty = 0.0
            if seed_contribution > 1.0 and attribution_discount >= 0.5:
                total_imp = seed_info.total_improvement if seed_info else 0.0
                if total_imp > 0.1:
                    # Safe zone: actual improvement exists
                    # Check if contribution vastly exceeds improvement (suspicious)
                    ratio = seed_contribution / total_imp
                    if ratio > 5.0:
                        # Contribution > 5x improvement - possible dependency gaming
                        # Escalating penalty: 0 at ratio 5, -0.1 at ratio 10, -0.3 at ratio 20+
                        ratio_penalty = -min(0.3, 0.1 * (ratio - 5) / 5)
                elif total_imp <= 0.1:
                    # Dangerous: high contribution but no real improvement
                    # Scale penalty by contribution magnitude (cap at 10%)
                    ratio_penalty = -0.3 * min(1.0, seed_contribution / 10.0)
                attributed += ratio_penalty / config.contribution_weight  # Apply before weight

            if components:
                components.ratio_penalty = ratio_penalty

            bounded_attribution = config.contribution_weight * attributed
    elif seed_info is not None and not seed_is_fossilized:
        # Pre-blending: use accuracy delta as proxy signal (lower weight)
        # This maintains reward continuity without imputing fake counterfactual.
        # No penalty for negative delta - we don't have causal data yet.
        # NOTE: Only applies when seed exists - seedless states get zero attribution
        if acc_delta is not None and acc_delta > 0:
            bounded_attribution = config.proxy_contribution_weight * acc_delta
    # else: No seed exists - zero attribution (host-only learning is not credited)
    # === FOSSILIZE-SPECIFIC ATTRIBUTION OVERRIDE ===
    # Zero attribution for fossilizing negative-improvement seeds.
    # The fossilize shaping penalty handles the negative signal - we shouldn't
    # ALSO give attribution credit for counterfactual "contribution" that
    # represents entanglement rather than value creation.
    action_name = action.name
    if action_name == "FOSSILIZE" and seed_info is not None:
        if seed_info.total_improvement < 0:
            bounded_attribution = 0.0

    # CULL: Invert attribution signal to align incentives correctly
    # - Culling a GOOD seed (positive contribution) = BAD decision → negative reward
    # - Culling a BAD seed (negative contribution) = GOOD decision → positive reward
    # Without this, the policy learns "CULL everything for +attribution rewards"
    if action_name == "CULL":
        bounded_attribution = -bounded_attribution

    reward += bounded_attribution

    if components:
        components.seed_contribution = seed_contribution
        components.bounded_attribution = bounded_attribution
        components.progress_since_germination = progress

    # === 1b. BLENDING WARNING: Escalating penalty for negative trajectory ===
    # Provides early signal to CULL seeds that are hurting performance
    # Credit assignment: penalize during BLENDING so policy learns to cull early
    blending_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_BLENDING:
        total_imp = seed_info.total_improvement
        if total_imp < 0:
            # Escalating penalty: longer negative trajectory = stronger signal to cull
            # Epoch 1: -0.15, Epoch 3: -0.25, Epoch 6+: -0.40
            escalation = min(seed_info.epochs_in_stage * 0.05, 0.3)
            blending_warning = -0.1 - escalation
            reward += blending_warning
    if components:
        components.blending_warning = blending_warning

    # === 1c. PROBATIONARY INDECISION PENALTY ===
    # Exponential escalation for WAITing too long in PROBATIONARY
    # Creates urgency to make FOSSILIZE/CULL decision before timeout
    # DRL Expert review 2025-12-10: steepened to overcome +7.5 attribution
    # Note: Orthogonal to blending_warning (different stage, different pathology)
    #
    # IMPORTANT: Only apply when bounded_attribution > 0 (legitimate seed being farmed).
    # For ransomware seeds (attr ~= 0 due to discount), the agent's correct action is
    # CULL, not FOSSILIZE. Penalizing WAIT in this case provides no useful gradient -
    # the attribution discount already zeroed rewards. Penalty stacking creates an
    # unlearnable reward landscape where every action is punished.
    probation_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_PROBATIONARY:
        if action_name == "WAIT":
            # Only penalize when:
            # 1. Counterfactual data is available (agent has info to act)
            # 2. Attribution is positive (legitimate seed being farmed)
            # 3. Past grace period (epoch 1 is free for information gathering)
            if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
                has_counterfactual = (
                    seed_info.total_improvement is not None
                    or seed_info.improvement_since_stage_start is not None
                )
                if has_counterfactual:
                    # Exponential: epoch 2 -> -1.0, epoch 3 -> -3.0, epoch 4 -> -9.0
                    # Formula: -1.0 * (3 ** (epochs_waiting - 1))
                    epochs_waiting = seed_info.epochs_in_stage - 1
                    probation_warning = -1.0 * (3 ** (epochs_waiting - 1))
                    # Cap at -10.0 (clip boundary) to avoid extreme penalties
                    probation_warning = max(probation_warning, -10.0)
                    reward += probation_warning
    if components:
        components.probation_warning = probation_warning

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
    # action_name already computed above for FOSSILIZE attribution override

    if _is_germinate_action(action):
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
    fossilize_terminal_bonus = 0.0
    if epoch == max_epochs:
        # Base accuracy bonus
        terminal_bonus = val_acc * config.terminal_acc_weight
        # ASYMMETRIC TERMINAL BONUS: Only reward CONTRIBUTING fossilized seeds
        # Seeds with total_improvement < MIN_FOSSILIZE_CONTRIBUTION get no terminal bonus.
        # This makes bad fossilizations NPV-negative (immediate penalty not offset by terminal).
        # DRL Expert review 2025-12-11: prevents ransomware seeds from being NPV-positive
        fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
        terminal_bonus += fossilize_terminal_bonus
        reward += terminal_bonus
    if components:
        components.terminal_bonus = terminal_bonus
        components.fossilize_terminal_bonus = fossilize_terminal_bonus
        components.num_fossilized_seeds = num_fossilized_seeds
        components.num_contributing_fossilized = num_contributing_fossilized

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
        if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
            _logger.warning(
                "PBRS telescoping risk: transition from stage %d with previous_epochs_in_stage=0. "
                "phi_prev will be underestimated. This indicates SeedInfo was constructed incorrectly.",
                seed_info.previous_stage,
            )
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
    """Shaping for FOSSILIZE action - with legitimacy and ransomware checks.

    Rapid fossilization (short PROBATIONARY period) is discounted to prevent
    dependency gaming where seeds create artificial dependencies during
    BLENDING that inflate metrics.

    Ransomware check: Seeds with negative total_improvement are penalized
    regardless of counterfactual contribution. High causal contribution with
    negative total delta indicates the seed created dependencies without
    adding value - the "ransomware" pattern.
    """
    if seed_info is None:
        return config.invalid_fossilize_penalty

    # FOSSILIZE only valid from PROBATIONARY
    if seed_info.stage != STAGE_PROBATIONARY:
        return config.invalid_fossilize_penalty

    # === RANSOMWARE CHECK: total_improvement must be non-negative ===
    # A seed that hurt total performance should be penalized for fossilizing
    total_delta = seed_info.total_improvement
    if total_delta < 0:
        # Base penalty + scaled by damage done (capped at 1.0 extra)
        base_penalty = -0.5
        damage_scale = min(abs(total_delta) * 0.2, 1.0)

        # Extra penalty for ransomware signature: high contribution + negative total
        # This seed created dependencies without adding value
        ransomware_signature = (
            seed_contribution is not None
            and seed_contribution > 0.1
            and total_delta < -0.2
        )
        ransomware_penalty = -0.3 if ransomware_signature else 0.0

        return base_penalty - damage_scale + ransomware_penalty

    # Legitimacy discount: must have spent time in PROBATIONARY to earn full bonus
    # This prevents rapid fossilization gaming
    legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_PROBATION_EPOCHS)

    # Use seed_contribution to determine if fossilization is earned
    # Aligned with G5 gate: require MIN_FOSSILIZE_CONTRIBUTION to get bonus
    # This prevents reward leak from low-contribution FOSSILIZE attempts
    if seed_contribution is not None and seed_contribution >= MIN_FOSSILIZE_CONTRIBUTION:
        # Bonus scales with actual contribution AND legitimacy
        base_bonus = (
            config.fossilize_base_bonus
            + config.fossilize_contribution_scale * seed_contribution
        )
        return base_bonus * legitimacy_discount

    # Below threshold or no counterfactual - penalty (no discount on penalties)
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
        - PROBATIONARY=6, FOSSILIZED=7
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
    if _is_germinate_action(action):
        return GERMINATE_INTERVENTION_COST
    return INTERVENTION_COSTS_BY_NAME.get(action.name, 0.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    "LossRewardConfig",
    "ContributionRewardConfig",
    # Seed info
    "SeedInfo",
    # Reward functions
    "compute_contribution_reward",
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
    "STAGE_PROBATIONARY",
]
