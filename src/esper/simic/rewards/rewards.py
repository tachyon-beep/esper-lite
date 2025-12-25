"""Reward Computation for Tamiyo Seed Lifecycle Controller.

This module consolidates reward functions used across:
- Vectorized PPO training (simic/vectorized.py, simic/ppo.py)
- Heuristic baseline comparisons (simic/training.py)

The reward design follows these principles:
1. Counterfactual validation is the primary signal (seed_contribution)
2. Lifecycle progression bonuses encourage exploration (PBRS)
3. Compute rent penalizes parameter bloat
4. Potential-based shaping maintains optimal policy invariance

Usage:
    from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig
    from esper.leyline import LifecycleOp

    reward = compute_contribution_reward(
        action=LifecycleOp.GERMINATE,
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
from enum import Enum
from typing import Any, NamedTuple, cast

from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    LifecycleOp,
    MIN_HOLDING_EPOCHS,
    MIN_PRUNE_AGE,
    SeedStage,
)
from esper.nissa import get_hub
from .reward_telemetry import RewardComponentsTelemetry

_logger = logging.getLogger(__name__)


class RewardMode(Enum):
    """Reward function variant for experimentation.

    SHAPED: Current dense shaping with PBRS, attribution, warnings (default)
    SPARSE: Terminal-only ground truth (accuracy - param_cost)
    MINIMAL: Sparse + early-prune penalty only
    SIMPLIFIED: DRL Expert recommended - PBRS + intervention cost + terminal only
    """
    SHAPED = "shaped"
    SPARSE = "sparse"
    MINIMAL = "minimal"
    SIMPLIFIED = "simplified"


class RewardFamily(Enum):
    """Top-level reward family selection."""

    CONTRIBUTION = "contribution"
    LOSS = "loss"


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
    # Value 5 intentionally skipped (was SHADOWING, removed)
    6: 5.5,   # HOLDING
    7: 6.0,   # FOSSILIZED (smallest increment - not a farming target)
}

# DEFAULT_GAMMA imported from leyline - single source of truth for PPO/PBRS gamma.
# PBRS theory requires gamma_pbrs == gamma_ppo for policy invariance (Ng et al., 1999).
# See leyline/__init__.py for value rationale.


# =============================================================================
# Reward Configuration
# =============================================================================


# =============================================================================
# Loss-Primary Reward Configuration (Phase 2)
# =============================================================================


# =============================================================================
# Contribution-Primary Reward Configuration (uses counterfactual validation)
# =============================================================================


@dataclass
class ContributionRewardConfig:
    """Configuration for contribution-primary reward computation.

    Uses counterfactual validation (seed_contribution) as the primary signal,
    eliminating heuristics that conflate host drift with seed impact.

    This is the recommended reward function when counterfactual validation
    is enabled in vectorized training.

    Note:
        `proxy_contribution_weight` is derived from `contribution_weight * proxy_confidence_factor`.
        This ensures the proxy signal automatically scales with contribution_weight changes.
    """

    # Primary signal: seed contribution weight
    # Reduced from 3.0 to 1.0 for stable PPO (per-step rewards should be in [-10, +10])
    contribution_weight: float = 1.0

    # M2: Proxy confidence factor - how trustworthy is proxy vs counterfactual signal?
    # 0.3 means "proxy is 30% as reliable as true counterfactual"
    # proxy_contribution_weight is derived as: contribution_weight * proxy_confidence_factor
    proxy_confidence_factor: float = 0.3

    @property
    def proxy_contribution_weight(self) -> float:
        """Derived proxy weight: contribution_weight * proxy_confidence_factor."""
        return self.contribution_weight * self.proxy_confidence_factor

    # PBRS stage progression
    pbrs_weight: float = 0.3
    epoch_progress_bonus: float = 0.3
    max_progress_bonus: float = 2.0

    # Compute rent (logarithmic scaling)
    rent_weight: float = 0.5
    max_rent: float = 8.0
    # Alpha-weighted rent floor (BaseSlotRent): ratio of host params per occupied slot.
    # Calibrated from telemetry_2025-12-20_044944: ~0.0039.
    base_slot_rent_ratio: float = 0.0039
    # Convex alpha shock coefficient (penalizes fast alpha changes).
    # Calibrated from telemetry_2025-12-20_044944: ~0.1958.
    alpha_shock_coef: float = 0.1958

    # Enforcement penalties (state machine compliance)
    invalid_fossilize_penalty: float = -1.0
    prune_fossilized_penalty: float = -1.0
    germinate_with_seed_penalty: float = -0.3

    # Intervention costs (action friction)
    germinate_cost: float = -0.02
    fossilize_cost: float = -0.01
    prune_cost: float = -0.005
    set_alpha_target_cost: float = -0.005

    # Fossilize shaping
    fossilize_base_bonus: float = 0.5
    fossilize_contribution_scale: float = 0.1
    fossilize_noncontributing_penalty: float = -0.2

    # Prune shaping
    prune_hurting_bonus: float = 0.3
    prune_acceptable_bonus: float = 0.1
    prune_good_seed_penalty: float = -0.3
    prune_hurting_threshold: float = -0.5

    # Anti-gaming: attribution discount and ratio penalty thresholds
    # Prevents seeds from gaming counterfactual by creating dependencies
    # - improvement_safe_threshold: below this, high contribution is suspicious
    # - hacking_ratio_threshold: contribution/improvement ratio triggering penalty
    # - attribution_sigmoid_steepness: controls discount curve for regressing seeds
    #   Lower values are more forgiving of normal training variance (±0.1-0.3%)
    #   steepness=10: -0.1% regression → 27% credit (too aggressive)
    #   steepness=3:  -0.1% regression → 43% credit, -0.5% → 18% (balanced)
    improvement_safe_threshold: float = 0.1
    hacking_ratio_threshold: float = 5.0
    attribution_sigmoid_steepness: float = 3.0

    # Terminal bonus
    terminal_acc_weight: float = 0.05
    # Terminal bonus per fossilized seed (incentivizes completion over farming)
    # DRL Expert review 2025-12-10: set to 3.0 to compete with post-scale attribution
    fossilize_terminal_scale: float = 3.0

    # Gamma for PBRS (uses module constant for consistency)
    gamma: float = DEFAULT_GAMMA

    # === Experiment Mode ===
    reward_mode: RewardMode = RewardMode.SHAPED

    # === Sparse Reward Parameters ===
    # Parameter budget for efficiency calculation (sparse/minimal modes)
    param_budget: int = 500_000
    # Weight for parameter penalty in sparse reward
    param_penalty_weight: float = 0.1
    # Reward scaling factor for sparse mode (DRL Expert: try 2.0-3.0 if learning fails)
    # Higher scale helps with credit assignment over 25 timesteps
    sparse_reward_scale: float = 1.0

    # === Minimal Mode Parameters ===
    # Minimum seed age before prune (epochs)
    early_prune_threshold: int = 5
    # Penalty for pruning young seeds
    early_prune_penalty: float = -0.1

    # === Auto-prune Penalty (degenerate policy prevention) ===
    # Penalty applied when environment auto-prunes a seed (safety or timeout)
    # instead of the policy explicitly choosing to prune.
    # (DRL Expert review 2025-12-17: prevents WAIT-spam policies that rely on
    # environment cleanup rather than learning proactive lifecycle management)
    auto_prune_penalty: float = -0.2

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
        4=BLENDING, 6=HOLDING, 7=FOSSILIZED, etc. (5 skipped)
    """

    stage: int  # SeedStage.value
    improvement_since_stage_start: float
    total_improvement: float  # Since germination (for G5 gate alignment)
    epochs_in_stage: int
    seed_params: int = 0  # Trainable params of active seed
    previous_stage: int = 0  # For PBRS stage bonus calculation
    previous_epochs_in_stage: int = 0  # Epochs in previous stage at transition (for PBRS telescoping)
    seed_age_epochs: int = 0  # Total epochs since germination (for rent grace)
    # Scaffolding support (Phase 3.1)
    interaction_sum: float = 0.0  # Total synergy with other seeds
    boost_received: float = 0.0  # Strongest single interaction

    @staticmethod
    def from_seed_state(seed_state: Any, seed_params: int = 0) -> "SeedInfo | None":
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
        interaction_sum = 0.0
        boost_received = 0.0
        if metrics:
            improvement = metrics.current_val_accuracy - metrics.accuracy_at_stage_start
            total_improvement = metrics.total_improvement
            seed_age = metrics.epochs_total
            interaction_sum = metrics.interaction_sum
            boost_received = metrics.boost_received
        return SeedInfo(
            stage=seed_state.stage.value,
            improvement_since_stage_start=improvement,
            total_improvement=total_improvement,
            epochs_in_stage=seed_state.epochs_in_stage,
            seed_params=seed_params,
            previous_stage=seed_state.previous_stage.value,
            previous_epochs_in_stage=seed_state.previous_epochs_in_stage,
            seed_age_epochs=seed_age,
            interaction_sum=interaction_sum,
            boost_received=boost_received,
        )


# Stage constants from leyline contract
STAGE_GERMINATED = SeedStage.GERMINATED.value
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
STAGE_HOLDING = SeedStage.HOLDING.value


# =============================================================================
# Core Reward Functions
# =============================================================================


# Default config singleton (avoid repeated allocations)
_DEFAULT_CONTRIBUTION_CONFIG = ContributionRewardConfig()


# =============================================================================
# Contribution-Primary Reward (uses counterfactual validation)
# =============================================================================
#
# P2-2 Design Decision: This function is intentionally large (~300 lines, 7 components)
# because it implements sophisticated reward engineering where each component addresses
# a specific failure mode:
#   1. Bounded attribution - prevents ransomware signatures (gaming via model corruption)
#   2. Blending warning - early signal for problematic seeds before they cause damage
#   3. Holding indecision - prevents farming WAIT actions for positive rewards
#   4. PBRS stage progression - potential-based shaping preserves optimal policy
#   5. Compute rent - logarithmic parameter cost prevents model bloat
#   6. Action shaping - state machine compliance penalties
#   7. Terminal bonus - episode completion incentive
#
# Splitting into smaller functions would obscure the reward design and make it
# harder to reason about component interactions. Property tests in
# tests/simic/properties/test_pbrs_properties.py verify PBRS guarantees hold.
# =============================================================================


def compute_contribution_reward(
    action: LifecycleOp,
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
    slot_id: str | None = None,
    seed_id: str | None = None,
    effective_seed_params: float | None = None,
    alpha_delta_sq_sum: float = 0.0,
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
        action: Action taken (LifecycleOp enum member)
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
        num_contributing_fossilized: Count of fossilized seeds with total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION.
            Only these seeds receive terminal bonus. This prevents bad fossilizations from being NPV-positive.
        slot_id: Optional slot identifier for telemetry events (enables reward hacking detection)
        seed_id: Optional seed identifier for telemetry events (enables reward hacking detection)
        effective_seed_params: Optional alpha-weighted + BaseSlotRent param count.
            If None, uses total_params - host_params.
        alpha_delta_sq_sum: Sum of per-slot (Δalpha^2 * scale) for convex shock penalty.

    Returns:
        Shaped reward value, or (reward, components) if return_components=True
    """
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    # PBRS requires gamma_pbrs == gamma_ppo for policy invariance (Ng et al., 1999)
    # Runtime validation catches misconfiguration that would invalidate shaping guarantees
    # NOTE: Using ValueError instead of assert ensures check runs even with python -O
    if config.gamma != DEFAULT_GAMMA:
        raise ValueError(
            f"PBRS gamma mismatch: config.gamma={config.gamma} != DEFAULT_GAMMA={DEFAULT_GAMMA}. "
            "This breaks policy invariance guarantees (Ng et al., 1999). Use DEFAULT_GAMMA from leyline."
        )

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

    # Pre-compute attribution_discount and ratio_penalty for ALL seeds (including fossilized)
    # These are needed for telemetry and property tests even if we skip attribution rewards
    attribution_discount = 1.0
    ratio_penalty = 0.0
    if seed_contribution is not None and seed_info is not None:
        total_imp = seed_info.total_improvement

        # Attribution discount applies to all seeds with negative total_improvement
        # Sigmoid steepness controls how quickly discount kicks in for regressing seeds
        if total_imp < 0:
            # Clamp exponent to prevent overflow: exp(709) is the float64 limit
            exp_arg = min(-config.attribution_sigmoid_steepness * total_imp, 700.0)
            attribution_discount = 1.0 / (1.0 + math.exp(exp_arg))

        # Ratio penalty only for high contribution (> 1.0) to avoid noise
        # Only calculate when attribution_discount >= 0.5 (avoid penalty stacking)
        if seed_contribution > 1.0 and attribution_discount >= 0.5:
            # Guard against division by very small values even if threshold is misconfigured
            safe_threshold = max(config.improvement_safe_threshold, 1e-8)
            if total_imp > safe_threshold:
                # Safe zone: actual improvement exists
                # Check if contribution vastly exceeds improvement (suspicious)
                ratio = seed_contribution / total_imp
                if ratio > config.hacking_ratio_threshold:
                    # Contribution > threshold × improvement - possible dependency gaming
                    # Escalating penalty: 0 at threshold, -0.1 at 2x threshold, capped at prune_good_seed_penalty
                    ratio_penalty = -min(
                        -config.prune_good_seed_penalty,
                        0.1 * (ratio - config.hacking_ratio_threshold) / config.hacking_ratio_threshold
                    )
            elif total_imp <= config.improvement_safe_threshold:
                # Dangerous: high contribution but no real improvement
                # Scale penalty by contribution magnitude (cap at prune_good_seed_penalty)
                ratio_penalty = config.prune_good_seed_penalty * min(1.0, seed_contribution / 10.0)

        # === H9: Wire telemetry for reward hacking detection ===
        # Emit telemetry events when attribution anomalies are detected
        if slot_id is not None and seed_id is not None:
            hub = get_hub()
            # Check for reward hacking (contribution >> improvement)
            if total_imp > 0 and ratio_penalty != 0:
                _check_reward_hacking(
                    hub,
                    seed_contribution=seed_contribution,
                    total_improvement=total_imp,
                    hacking_ratio_threshold=config.hacking_ratio_threshold,
                    slot_id=slot_id,
                    seed_id=seed_id,
                )
            # Check for ransomware signature (high contribution + negative total)
            _check_ransomware_signature(
                hub,
                seed_contribution=seed_contribution,
                total_improvement=total_imp,
                slot_id=slot_id,
                seed_id=seed_id,
            )

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

            # Apply attribution discount (pre-computed above)
            # Prevents rewarding seeds that show good per-step counterfactual but
            # have negative total_improvement (the ransomware buildup pattern).
            attributed *= attribution_discount

            # Apply ratio penalty (pre-computed above)
            # High causal contribution with low improvement = structural entanglement
            attributed += ratio_penalty / config.contribution_weight  # Apply before weight

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
    if action == LifecycleOp.FOSSILIZE and seed_info is not None:
        if seed_info.total_improvement < 0:
            bounded_attribution = 0.0

    # PRUNE: Invert attribution signal to align incentives correctly
    # - Pruning a GOOD seed (positive contribution) = BAD decision → negative reward
    # - Pruning a BAD seed (negative contribution) = GOOD decision → positive reward
    # Without this, the policy learns "PRUNE everything for +attribution rewards"
    if action == LifecycleOp.PRUNE:
        bounded_attribution = -bounded_attribution

    reward += bounded_attribution

    if components:
        components.seed_contribution = seed_contribution
        components.bounded_attribution = bounded_attribution
        components.progress_since_germination = progress
        components.attribution_discount = attribution_discount
        components.ratio_penalty = ratio_penalty

    # === 1b. BLENDING WARNING: Escalating penalty for negative trajectory ===
    # Provides early signal to PRUNE seeds that are hurting performance
    # Credit assignment: penalize during BLENDING so policy learns to prune early
    blending_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_BLENDING:
        total_imp = seed_info.total_improvement
        if total_imp < 0:
            # Escalating penalty: longer negative trajectory = stronger signal to prune
            # Epoch 1: -0.15, Epoch 3: -0.25, Epoch 6+: -0.40
            escalation = min(seed_info.epochs_in_stage * 0.05, 0.3)
            blending_warning = -0.1 - escalation
            reward += blending_warning
    if components:
        components.blending_warning = blending_warning

    # === 1c. HOLDING INDECISION PENALTY ===
    # Exponential escalation for WAITing too long in HOLDING
    # Creates urgency to make FOSSILIZE/PRUNE decision before timeout
    # DRL Expert review 2025-12-10: steepened to overcome +7.5 attribution
    # Note: Orthogonal to blending_warning (different stage, different pathology)
    #
    # IMPORTANT: Only apply when bounded_attribution > 0 (legitimate seed being farmed).
    # For ransomware seeds (attr ~= 0 due to discount), the agent's correct action is
    # PRUNE, not FOSSILIZE. Penalizing WAIT in this case provides no useful gradient -
    # the attribution discount already zeroed rewards. Penalty stacking creates an
    # unlearnable reward landscape where every action is punished.
    holding_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_HOLDING:
        if action == LifecycleOp.WAIT:
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
                    holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
                    # Cap at -10.0 (clip boundary) to avoid extreme penalties
                    holding_warning = max(holding_warning, -10.0)
                    reward += holding_warning
    if components:
        components.holding_warning = holding_warning

    # === 2. PBRS: Stage Progression ===
    # Potential-based shaping preserves optimal policy (Ng et al., 1999)
    pbrs_bonus = 0.0
    if seed_info is not None:
        pbrs_bonus = _contribution_pbrs_bonus(seed_info, config)
        reward += pbrs_bonus
    if components:
        components.pbrs_bonus = pbrs_bonus

    # === 2b. SCAFFOLDING: Synergy Bonus ===
    # Reward seeds that have positive interactions with others
    synergy_bonus = 0.0
    if seed_info is not None:
        synergy_bonus = _compute_synergy_bonus(
            seed_info.interaction_sum,
            seed_info.boost_received,
        )
        reward += synergy_bonus
    if components:
        components.synergy_bonus = synergy_bonus

    # === 3. RENT: Compute Cost ===
    # Logarithmic penalty on parameter bloat from seeds (alpha-weighted + BaseSlotRent)
    rent_penalty = 0.0
    growth_ratio = 0.0
    if host_params > 0:
        effective_overhead = (
            effective_seed_params
            if effective_seed_params is not None
            else max(total_params - host_params, 0)
        )
        if effective_overhead > 0:
            # Measure EXCESS params from seeds, not total ratio
            # growth_ratio = 0 when no seeds, so no rent penalty
            growth_ratio = effective_overhead / host_params
            scaled_cost = math.log(1.0 + growth_ratio)
            rent_penalty = min(config.rent_weight * scaled_cost, config.max_rent)
            reward -= rent_penalty
    if components:
        components.compute_rent = -rent_penalty  # Negative because it's a penalty
        components.growth_ratio = growth_ratio  # DRL Expert diagnostic field

    # === 3b. SHOCK: Convex penalty on alpha changes ===
    alpha_shock = 0.0
    if alpha_delta_sq_sum > 0 and config.alpha_shock_coef != 0.0:
        alpha_shock = -config.alpha_shock_coef * alpha_delta_sq_sum
        reward += alpha_shock
    if components:
        components.alpha_shock = alpha_shock

    # === 4. ACTION SHAPING ===
    # Minimal - just state machine enforcement and intervention costs
    action_shaping = 0.0

    if action == LifecycleOp.GERMINATE:
        if seed_info is not None:
            action_shaping += config.germinate_with_seed_penalty
        else:
            # PBRS bonus for successful germination (no existing seed)
            # Balances the PBRS penalty applied when pruning seeds
            phi_germinated = STAGE_POTENTIALS.get(STAGE_GERMINATED, 0.0)
            phi_no_seed = 0.0
            pbrs_germinate = config.gamma * phi_germinated - phi_no_seed
            action_shaping += config.pbrs_weight * pbrs_germinate
        action_shaping += config.germinate_cost

    elif action == LifecycleOp.FOSSILIZE:
        action_shaping += _contribution_fossilize_shaping(seed_info, seed_contribution, config)
        action_shaping += config.fossilize_cost

    elif action == LifecycleOp.PRUNE:
        action_shaping += _contribution_prune_shaping(seed_info, seed_contribution, config)
        action_shaping += config.prune_cost
    elif action == LifecycleOp.SET_ALPHA_TARGET:
        action_shaping += config.set_alpha_target_cost

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
        # Seeds with total_improvement < DEFAULT_MIN_FOSSILIZE_CONTRIBUTION get no terminal bonus.
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


def compute_sparse_reward(
    host_max_acc: float,
    total_params: int,
    epoch: int,
    max_epochs: int,
    config: ContributionRewardConfig,
) -> float:
    """Compute sparse (terminal-only) reward.

    This reward function returns 0.0 for all non-terminal timesteps,
    forcing the LSTM policy to perform genuine temporal credit assignment
    over the full episode. At terminal, it rewards accuracy and penalizes
    parameter count.

    Design rationale:
    - Terminal-only: Forces credit assignment, tests if shaping is necessary
    - Accuracy-primary: The true objective is host performance
    - Param penalty: Efficiency matters, but less than accuracy
    - Scale factor: DRL Expert recommends 2.0-3.0 if learning fails

    Args:
        host_max_acc: Maximum accuracy achieved during episode (0-100)
        total_params: Extra seed parameters (active + fossilized) at episode end
        epoch: Current epoch (1-indexed)
        max_epochs: Maximum epochs in episode
        config: Reward configuration with param_budget, param_penalty_weight, sparse_reward_scale

    Returns:
        0.0 for non-terminal epochs, scaled reward at terminal in [-scale, scale]
    """
    # Non-terminal: return 0.0 (the defining property of sparse rewards)
    if epoch != max_epochs:
        return 0.0

    # Terminal reward: accuracy minus parameter cost
    accuracy_reward = host_max_acc / 100.0
    param_cost = config.param_penalty_weight * (total_params / config.param_budget)

    # H10 FIX: Clamp base reward to [-1, 1] BEFORE scaling, not after.
    # This ensures sparse_reward_scale actually affects magnitude.
    # Without this fix, scale=2.5 with base=0.78 → scaled=1.95 → clamped to 1.0 (scale is defeated).
    # With fix: base=0.78 → clamped to 0.78 → scaled to 1.95 (scale is effective).
    base_reward = accuracy_reward - param_cost
    clamped_base = max(-1.0, min(1.0, base_reward))

    # Apply scale for better gradient signal (DRL Expert recommendation)
    # Final reward is in [-scale, scale] range
    return config.sparse_reward_scale * clamped_base


def compute_minimal_reward(
    host_max_acc: float,
    total_params: int,
    epoch: int,
    max_epochs: int,
    action: LifecycleOp,
    seed_age: int | None,
    config: ContributionRewardConfig,
) -> float:
    """Compute minimal reward (sparse + early-prune penalty).

    This is a fallback if pure sparse rewards fail to learn. It adds
    a single shaping signal: penalize pruning seeds before they've had
    a chance to prove themselves.

    Design rationale:
    - Sparse base: Preserves most of the credit assignment challenge
    - Early-prune penalty: Prevents degenerate "prune everything" policy
    - No other shaping: Tests if minimal guidance is sufficient

    Args:
        host_max_acc: Maximum accuracy achieved during episode
        total_params: Extra seed parameters (active + fossilized) at episode end
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        action: Action taken this timestep
        seed_age: Age of the seed in epochs (None if no seed)
        config: Reward configuration

    Returns:
        Sparse reward + early-prune penalty if applicable
    """
    # Start with sparse reward
    reward = compute_sparse_reward(
        host_max_acc=host_max_acc,
        total_params=total_params,
        epoch=epoch,
        max_epochs=max_epochs,
        config=config,
    )

    # Add early-prune penalty if applicable
    if action == LifecycleOp.PRUNE and seed_age is not None:
        if seed_age < config.early_prune_threshold:
            reward += config.early_prune_penalty

    return reward


def compute_simplified_reward(
    action: LifecycleOp,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    val_acc: float,
    num_contributing_fossilized: int,
    config: ContributionRewardConfig | None = None,
) -> float:
    """Compute simplified 3-component reward (DRL Expert recommended).

    This reward function addresses the "unlearnable landscape" problem by
    removing conflicting components. Only three signals remain:

    1. PBRS stage progression (preserves optimal policy per Ng et al., 1999)
    2. Uniform intervention cost (small friction on non-WAIT actions)
    3. Terminal bonus (accuracy + fossilize count, scaled for 25-step credit)

    Removed vs SHAPED:
    - bounded_attribution (replace with terminal accuracy)
    - blending_warning (let terminal handle bad seeds)
    - holding_warning (let PBRS + terminal handle pacing)
    - ratio_penalty / attribution_discount (address via environment, not reward)
    - compute_rent (simplify - not critical for learning)

    Args:
        action: Action taken (LifecycleOp enum member)
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        val_acc: Current validation accuracy
        num_contributing_fossilized: Count of fossilized seeds with meaningful contribution
        config: Reward configuration (uses default if None)

    Returns:
        Simplified reward value
    """
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    reward = 0.0

    # === 1. PBRS: Stage Progression ===
    # This is the ONLY shaping that preserves optimal policy guarantees
    if seed_info is not None:
        reward += _contribution_pbrs_bonus(seed_info, config)

    # === 2. Intervention Cost ===
    # Uniform small negative cost for any non-WAIT action
    # Prevents "action spam" without creating complex penalty landscape
    if action != LifecycleOp.WAIT:
        reward -= 0.01

    # === 3. Terminal Bonus ===
    # Scaled for 25-step credit assignment (DRL Expert recommendation)
    if epoch == max_epochs:
        # Accuracy component: [0, 3] range
        accuracy_bonus = (val_acc / 100.0) * 3.0
        # Fossilize component: [0, 6] for 3 slots max
        fossilize_bonus = num_contributing_fossilized * 2.0
        reward += accuracy_bonus + fossilize_bonus

    return reward


def compute_reward(
    action: LifecycleOp,
    seed_contribution: float | None,
    val_acc: float,
    host_max_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int,
    host_params: int,
    acc_at_germination: float | None,
    acc_delta: float,
    num_fossilized_seeds: int = 0,
    num_contributing_fossilized: int = 0,
    config: ContributionRewardConfig | None = None,
    return_components: bool = False,
    effective_seed_params: float | None = None,
    alpha_delta_sq_sum: float = 0.0,
) -> float | tuple[float, "RewardComponentsTelemetry"]:
    """Unified reward computation dispatcher.

    Routes to the appropriate reward function based on config.reward_mode:
    - SHAPED: Dense shaping with PBRS, attribution, warnings (default)
    - SPARSE: Terminal-only ground truth reward
    - MINIMAL: Sparse + early-prune penalty
    - SIMPLIFIED: PBRS + intervention cost + terminal (DRL Expert recommended)

    Args:
        action: Action taken (LifecycleOp or similar IntEnum)
        seed_contribution: Counterfactual contribution (None if unavailable)
        val_acc: Current validation accuracy
        host_max_acc: Maximum accuracy achieved during episode
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Extra seed parameters (active + fossilized)
        host_params: Host model parameters
        acc_at_germination: Accuracy when seed was planted
        acc_delta: Per-epoch accuracy change
        num_fossilized_seeds: Count of fossilized seeds
        num_contributing_fossilized: Count of contributing fossilized seeds
        config: Reward configuration (uses default if None)
        return_components: If True, return (reward, components) tuple
        effective_seed_params: Optional alpha-weighted + BaseSlotRent param count.
        alpha_delta_sq_sum: Sum of per-slot (Δalpha^2 * scale) for convex shock penalty.

    Returns:
        Reward value, or (reward, components) if return_components=True
    """
    if config is None:
        config = ContributionRewardConfig()

    # Dispatch based on reward mode
    if config.reward_mode == RewardMode.SHAPED:
        return compute_contribution_reward(
            action=action,
            seed_contribution=seed_contribution,
            val_acc=val_acc,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            config=config,
            acc_at_germination=acc_at_germination,
            acc_delta=acc_delta,
            return_components=return_components,
            num_fossilized_seeds=num_fossilized_seeds,
            num_contributing_fossilized=num_contributing_fossilized,
            effective_seed_params=effective_seed_params,
            alpha_delta_sq_sum=alpha_delta_sq_sum,
        )

    elif config.reward_mode == RewardMode.SPARSE:
        reward = compute_sparse_reward(
            host_max_acc=host_max_acc,
            total_params=total_params,
            epoch=epoch,
            max_epochs=max_epochs,
            config=config,
        )

    elif config.reward_mode == RewardMode.MINIMAL:
        seed_age = seed_info.seed_age_epochs if seed_info else None
        reward = compute_minimal_reward(
            host_max_acc=host_max_acc,
            total_params=total_params,
            epoch=epoch,
            max_epochs=max_epochs,
            action=action,
            seed_age=seed_age,
            config=config,
        )

    elif config.reward_mode == RewardMode.SIMPLIFIED:
        reward = compute_simplified_reward(
            action=action,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            val_acc=val_acc,
            num_contributing_fossilized=num_contributing_fossilized,
            config=config,
        )

    else:
        raise ValueError(f"Unknown reward mode: {config.reward_mode}")

    # Handle return_components for sparse/minimal/simplified modes
    if return_components:
        components = RewardComponentsTelemetry()
        components.total_reward = reward
        components.action_name = action.name
        components.epoch = epoch
        components.seed_stage = seed_info.stage if seed_info else None
        components.val_acc = val_acc
        return reward, components

    return reward


def compute_reward_for_family(
    reward_family: RewardFamily,
    *,
    action: LifecycleOp,
    seed_contribution: float | None,
    val_acc: float,
    host_max_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int,
    host_params: int,
    acc_at_germination: float | None,
    acc_delta: float,
    num_fossilized_seeds: int = 0,
    num_contributing_fossilized: int = 0,
    contribution_config: ContributionRewardConfig | None = None,
    loss_config: LossRewardConfig | None = None,
    loss_delta: float = 0.0,
    val_loss: float = 0.0,
    effective_seed_params: float | None = None,
    alpha_delta_sq_sum: float = 0.0,
) -> float:
    """Dispatch reward based on family (contribution vs loss-primary)."""
    if contribution_config is None:
        contribution_config = ContributionRewardConfig()
    if loss_config is None:
        loss_config = LossRewardConfig.default()

    if reward_family == RewardFamily.CONTRIBUTION:
        # cast() needed because compute_reward's return type depends on return_components
        # When return_components=False, it returns float (not tuple)
        return cast(float, compute_reward(
            action=action,
            seed_contribution=seed_contribution,
            val_acc=val_acc,
            host_max_acc=host_max_acc,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            acc_at_germination=acc_at_germination,
            acc_delta=acc_delta,
            num_fossilized_seeds=num_fossilized_seeds,
            num_contributing_fossilized=num_contributing_fossilized,
            config=contribution_config,
            return_components=False,
            effective_seed_params=effective_seed_params,
            alpha_delta_sq_sum=alpha_delta_sq_sum,
        ))
    if reward_family == RewardFamily.LOSS:
        return compute_loss_reward(
            action=action,
            loss_delta=loss_delta,
            val_loss=val_loss,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            config=loss_config,
        )
    raise ValueError(f"Unknown reward family: {reward_family}")


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


def _compute_synergy_bonus(
    interaction_sum: float,
    boost_received: float,
    synergy_weight: float = 0.1,
) -> float:
    """Compute synergy bonus for scaffolding behavior.

    Rewards seeds that have positive interactions with others.
    Uses tanh to bound the bonus and prevent reward hacking.

    Args:
        interaction_sum: Total interaction I_ij with all other seeds
        boost_received: Maximum single interaction
        synergy_weight: Scaling factor for bonus (default 0.1)

    Returns:
        Bounded synergy bonus in [0, synergy_weight]
    """
    if interaction_sum <= 0:
        return 0.0

    # Tanh bounds the bonus, preventing runaway positive feedback
    raw_bonus = math.tanh(interaction_sum * 0.5)
    return raw_bonus * synergy_weight


def compute_scaffold_hindsight_credit(
    boost_given: float,
    beneficiary_improvement: float,
    credit_weight: float = 0.2,
) -> float:
    """Compute retroactive credit for scaffold seeds.

    When a beneficiary seed fossilizes successfully, the scaffold seed
    that boosted it receives credit proportional to its contribution.

    This implements Hindsight Credit Assignment for scaffolding:
    the scaffold's value is only known after the beneficiary succeeds.

    Args:
        boost_given: The interaction term I_ij that scaffold provided
        beneficiary_improvement: The improvement the beneficiary achieved
        credit_weight: Maximum credit amount (default 0.2)

    Returns:
        Credit in [0, credit_weight]
    """
    if boost_given <= 0 or beneficiary_improvement <= 0:
        return 0.0

    # Credit is proportional to boost given and beneficiary success
    raw_credit = math.tanh(boost_given * beneficiary_improvement * 0.1)
    return raw_credit * credit_weight


def _contribution_fossilize_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for FOSSILIZE action - with legitimacy and ransomware checks.

    Rapid fossilization (short HOLDING period) is discounted to prevent
    dependency gaming where seeds create artificial dependencies during
    BLENDING that inflate metrics.

    Ransomware check: Seeds with negative total_improvement are penalized
    regardless of counterfactual contribution. High causal contribution with
    negative total delta indicates the seed created dependencies without
    adding value - the "ransomware" pattern.
    """
    if seed_info is None:
        return config.invalid_fossilize_penalty

    # FOSSILIZE only valid from HOLDING
    if seed_info.stage != STAGE_HOLDING:
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

    # Legitimacy discount: must have spent time in HOLDING to earn full bonus
    # This prevents rapid fossilization gaming
    legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_HOLDING_EPOCHS)

    # Use seed_contribution to determine if fossilization is earned
    # Aligned with G5 gate: require DEFAULT_MIN_FOSSILIZE_CONTRIBUTION to get bonus
    # This prevents reward leak from low-contribution FOSSILIZE attempts
    if seed_contribution is not None and seed_contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION:
        # Bonus scales with actual contribution AND legitimacy
        base_bonus = (
            config.fossilize_base_bonus
            + config.fossilize_contribution_scale * seed_contribution
        )
        return base_bonus * legitimacy_discount

    # Below threshold or no counterfactual - penalty (no discount on penalties)
    return config.fossilize_noncontributing_penalty


def _contribution_prune_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for PRUNE action - simplified with counterfactual."""
    if seed_info is None:
        return -0.2  # Nothing to prune

    if seed_info.stage == STAGE_FOSSILIZED:
        return config.prune_fossilized_penalty

    # Age gate: penalize pruning very young seeds
    if seed_info.seed_age_epochs < MIN_PRUNE_AGE:
        return -0.3 * (MIN_PRUNE_AGE - seed_info.seed_age_epochs)

    # Use seed_contribution if available (BLENDING+ stages)
    if seed_contribution is not None:
        if seed_contribution < config.prune_hurting_threshold:
            return config.prune_hurting_bonus  # Good: seed was hurting
        elif seed_contribution < 0:
            return config.prune_acceptable_bonus  # Acceptable: marginal harm
        else:
            # M3 FIX: Penalize pruning good seeds, scaled by how good, but BOUNDED.
            # Without cap: contribution=20 → penalty=-1.3, breaking reward scale.
            # Cap at 3x base penalty to keep in reasonable [-1, 0] range.
            base_penalty = config.prune_good_seed_penalty  # -0.3 default
            scaled_penalty = base_penalty - 0.05 * seed_contribution
            # Cap at 3x base penalty magnitude (e.g., -0.9 with default -0.3)
            max_penalty = 3.0 * base_penalty  # More negative than base
            return max(scaled_penalty, max_penalty)

    # No counterfactual (TRAINING stage) - neutral
    # We don't have information yet, so PRUNE is neither good nor bad
    return 0.0


def _check_reward_hacking(
    hub: Any,
    *,
    seed_contribution: float,
    total_improvement: float,
    hacking_ratio_threshold: float = 5.0,
    slot_id: str,
    seed_id: str,
) -> bool:
    """Emit REWARD_HACKING_SUSPECTED if attribution ratio is anomalous.

    A seed claiming more than 500% of total improvement is suspicious
    and may indicate reward hacking or measurement error. The 5x threshold
    aligns with the penalty threshold in compute_contribution_reward().

    Note: For ransomware detection (high contribution + negative total),
    use _check_ransomware_signature() instead.

    Returns True if event was emitted.
    """
    from esper.leyline import TelemetryEvent, TelemetryEventType

    if total_improvement <= 0 or seed_contribution <= 0:
        return False

    ratio = seed_contribution / total_improvement

    if ratio < hacking_ratio_threshold:
        return False

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.REWARD_HACKING_SUSPECTED,
        severity="warning",
        data={  # type: ignore[arg-type]
            "ratio": ratio,
            "seed_contribution": seed_contribution,
            "total_improvement": total_improvement,
            "threshold": hacking_ratio_threshold,
            "slot_id": slot_id,
            "seed_id": seed_id,
        },
    ))
    return True


def _check_ransomware_signature(
    hub: Any,
    *,
    seed_contribution: float,
    total_improvement: float,
    contribution_threshold: float = 1.0,
    degradation_threshold: float = -0.2,
    slot_id: str,
    seed_id: str,
) -> bool:
    """Emit REWARD_HACKING_SUSPECTED if seed shows ransomware signature.

    A "ransomware seed" is one that claims high contribution while the
    system is actually getting worse. This is the most dangerous Goodhart
    pattern: the seed has learned to maximize its counterfactual signal
    at the expense of actual performance.

    Signature: seed_contribution > 1.0 AND total_improvement < -0.2

    Returns True if event was emitted.
    """
    from esper.leyline import TelemetryEvent, TelemetryEventType

    if seed_contribution <= contribution_threshold:
        return False

    if total_improvement >= degradation_threshold:
        return False

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.REWARD_HACKING_SUSPECTED,
        severity="critical",
        data={  # type: ignore[arg-type]
            "pattern": "ransomware_signature",
            "seed_contribution": seed_contribution,
            "total_improvement": total_improvement,
            "contribution_threshold": contribution_threshold,
            "degradation_threshold": degradation_threshold,
            "slot_id": slot_id,
            "seed_id": seed_id,
        },
    ))
    return True


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
    base_potential = STAGE_POTENTIALS.get(seed_stage, 0.0)

    # Progress bonus matches ContributionRewardConfig defaults for PBRS consistency
    # epoch_progress_bonus=0.3, max_progress_bonus=2.0
    progress_bonus = min(epochs_in_stage * 0.3, 2.0)

    return float(base_potential + progress_bonus)


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

    # Compute rent with grace period (logarithmic scaling on seed overhead)
    if host_params > 0 and total_params > host_params:
        in_grace = False
        if seed_info is not None:
            in_grace = seed_info.seed_age_epochs < config.grace_epochs
        if not in_grace:
            # Measure EXCESS params from seeds, not total ratio
            # growth_ratio = 0 when no seeds, so no rent penalty
            growth_ratio = (total_params - host_params) / host_params
            scaled_cost = math.log(1.0 + growth_ratio)
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

# M5: Derive from ContributionRewardConfig defaults to avoid value duplication.
# This ensures INTERVENTION_COSTS stays in sync with config defaults.
_default_config = ContributionRewardConfig()
INTERVENTION_COSTS: dict[LifecycleOp, float] = {
    LifecycleOp.WAIT: 0.0,
    LifecycleOp.GERMINATE: _default_config.germinate_cost,
    LifecycleOp.FOSSILIZE: _default_config.fossilize_cost,
    LifecycleOp.PRUNE: _default_config.prune_cost,
    LifecycleOp.SET_ALPHA_TARGET: _default_config.set_alpha_target_cost,
    LifecycleOp.ADVANCE: 0.0,
}
del _default_config  # Don't pollute module namespace


def get_intervention_cost(action: LifecycleOp) -> float:
    """Get intervention cost for an action using default config values.

    Small negative costs discourage unnecessary interventions,
    encouraging the agent to only act when beneficial.

    Note: For custom costs, use ContributionRewardConfig fields directly.
    """
    return INTERVENTION_COSTS.get(action, 0.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    "LossRewardConfig",
    "ContributionRewardConfig",
    "RewardMode",
    "RewardFamily",
    # Seed info
    "SeedInfo",
    # Reward functions
    "compute_reward",
    "compute_contribution_reward",
    "compute_sparse_reward",
    "compute_minimal_reward",
    "compute_simplified_reward",
    "compute_loss_reward",
    "compute_scaffold_hindsight_credit",
    # PBRS utilities
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_seed_potential",
    # Intervention costs
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    # Stage constants and PBRS configuration
    "DEFAULT_GAMMA",
    "STAGE_POTENTIALS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_HOLDING",
]
