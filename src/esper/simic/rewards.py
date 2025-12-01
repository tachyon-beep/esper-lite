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

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from esper.leyline import SeedStage
from esper.leyline.actions import is_germinate_action


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
    acc_delta_weight: float = 0.5

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

    cull_failing_bonus: float = 0.0  # Was 0.3
    cull_acceptable_bonus: float = 0.0  # Was 0.1
    cull_promising_penalty: float = -0.3
    cull_no_seed_penalty: float = -0.2

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
    compute_rent_weight: float = 0.05
    compute_rent_exponent: float = 1.0  # Progressive tax exponent (1.0 = linear)
    max_rent_penalty: float = 5.0  # Cap to prevent runaway negatives

    # PBRS scaling for seed-based potential shaping
    seed_potential_weight: float = 0.3

    @staticmethod
    def default() -> "RewardConfig":
        """Return default configuration."""
        return RewardConfig()


# =============================================================================
# Loss-Primary Reward Configuration (Phase 2)
# =============================================================================


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

    # Compute rent
    compute_rent_weight: float = 0.05
    compute_rent_exponent: float = 1.0
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
            compute_rent_exponent=1.1,
        )

    @staticmethod
    def for_tinystories() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,
            typical_loss_delta_std=0.15,
            compute_rent_weight=0.01,
            compute_rent_exponent=1.5,
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
    epochs_in_stage: int
    seed_params: int = 0  # Trainable params of active seed
    previous_stage: int = 0  # For PBRS stage bonus calculation
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
        seed_age = 0
        if metrics:
            improvement = metrics.current_val_accuracy - metrics.accuracy_at_stage_start
            seed_age = metrics.epochs_total
        return SeedInfo(
            stage=seed_state.stage.value,
            improvement_since_stage_start=improvement,
            epochs_in_stage=seed_state.epochs_in_stage,
            seed_params=seed_params,
            previous_stage=seed_state.previous_stage.value,
            seed_age_epochs=seed_age,
        )


# Stage constants from leyline contract
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
) -> float:
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

    Returns:
        Shaped reward value
    """
    if config is None:
        config = _DEFAULT_CONFIG

    reward = 0.0

    # Base: accuracy improvement
    reward += acc_delta * config.acc_delta_weight

    # Compute rent: penalize excess params progressively
    if host_params > 0 and total_params > 0:
        # total_params currently passed as added params; convert to growth ratio
        growth_ratio = 1.0 + (total_params / host_params)
        scaled_cost = (growth_ratio**config.compute_rent_exponent) - 1.0
        rent_penalty = config.compute_rent_weight * scaled_cost
        rent_penalty = min(rent_penalty, config.max_rent_penalty)
        reward -= rent_penalty

    # Lifecycle stage rewards
    if seed_info is not None:
        stage = seed_info.stage
        improvement = seed_info.improvement_since_stage_start

        if stage == STAGE_TRAINING:
            reward += config.training_bonus
            if improvement > 0:
                reward += improvement * config.stage_improvement_weight

        elif stage == STAGE_BLENDING:
            reward += config.blending_bonus
            if acc_delta > 0:
                reward += config.blending_improvement_bonus

        elif stage == STAGE_FOSSILIZED:
            reward += config.fossilized_bonus

    # Potential-based reward shaping for lifecycle progression
    if seed_info is not None:
        current_stage = seed_info.stage
        previous_stage = seed_info.previous_stage
        current_obs = {
            "has_active_seed": 1,
            "seed_stage": current_stage,
            "seed_epochs_in_stage": seed_info.epochs_in_stage,
        }
        prev_obs = {
            "has_active_seed": 1,
            "seed_stage": previous_stage,
            "seed_epochs_in_stage": max(0, seed_info.epochs_in_stage - 1),
        }
        phi_t = compute_seed_potential(current_obs)
        phi_t_prev = compute_seed_potential(prev_obs)
        pb_bonus = compute_pbrs_bonus(phi_t_prev, phi_t, gamma=0.99)
        reward += config.seed_potential_weight * pb_bonus

    # Action-specific shaping (semantic, enum-based)
    # We rely on action.name (IntEnum member)
    action_name = action.name
    if is_germinate_action(action):
        reward += _germinate_shaping(seed_info, epoch, max_epochs, config)
    elif action_name == "FOSSILIZE":
        reward += _advance_shaping(seed_info, config)
    elif action_name == "CULL":
        reward += _cull_shaping(seed_info, config)
    elif action_name == "WAIT":
        reward += _wait_shaping(seed_info, acc_delta, config)

    # Terminal bonus
    if epoch == max_epochs:
        reward += val_acc * config.terminal_acc_weight

    return reward


def _germinate_shaping(
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    config: RewardConfig,
) -> float:
    """Compute shaping for GERMINATE action."""
    if seed_info is None:
        # Bonus for germinating when no active seed
        bonus = config.germinate_no_seed_bonus
        # Extra bonus for germinating early
        if epoch < max_epochs * config.early_epoch_fraction:
            bonus += config.germinate_early_bonus
        return bonus
    else:
        # Penalty for trying to germinate with existing seed
        return config.germinate_with_seed_penalty


def _advance_shaping(seed_info: SeedInfo | None, config: RewardConfig) -> float:
    """Compute shaping for FOSSILIZE action."""
    if seed_info is None:
        return config.advance_no_seed_penalty

    stage = seed_info.stage
    improvement = seed_info.improvement_since_stage_start

    # Only reward FOSSILIZE where it can actually finalize the seed.
    # Leyline VALID_TRANSITIONS only allow PROBATIONARY → FOSSILIZED; calls
    # from SHADOWING are rejected by SeedState.transition and treated as a
    # failed/no-op fossilize in the environment.
    if stage == STAGE_PROBATIONARY:
        if improvement > 0:
            return config.advance_good_bonus
        return config.advance_premature_penalty

    # Earlier lifecycle stages: FOSSILIZE is a no-op and should be discouraged.
    if stage in (STAGE_TRAINING, STAGE_BLENDING, STAGE_SHADOWING):
        return config.advance_premature_penalty

    # FOSSILIZED / others: no shaping for FOSSILIZE.
    return 0.0


def _cull_shaping(seed_info: SeedInfo | None, config: RewardConfig) -> float:
    """Compute shaping for CULL action."""
    if seed_info is None:
        return config.cull_no_seed_penalty

    improvement = seed_info.improvement_since_stage_start

    if improvement < config.cull_failing_threshold:
        return config.cull_failing_bonus
    elif improvement < 0:
        return config.cull_acceptable_bonus
    else:
        return config.cull_promising_penalty


def _wait_shaping(
    seed_info: SeedInfo | None,
    acc_delta: float,
    config: RewardConfig,
) -> float:
    """Compute shaping for WAIT action."""
    if seed_info is None:
        # Penalize waiting when plateauing with no seed
        if acc_delta < 0.5:
            return config.wait_plateau_penalty
        return 0.0

    stage = seed_info.stage
    improvement = seed_info.improvement_since_stage_start
    epochs_in_stage = seed_info.epochs_in_stage

    if stage == STAGE_TRAINING:
        if improvement > config.wait_patience_threshold:
            return config.wait_patience_bonus
        elif epochs_in_stage > config.wait_stagnant_epochs:
            return config.wait_stagnant_penalty

    return 0.0


# Default config singleton (avoid repeated allocations)
_DEFAULT_CONFIG = RewardConfig()


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
    gamma: float = 0.99,
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

    # Stage-based potential values (matching SeedStage enum values)
    # Monotonically increasing toward FOSSILIZED (terminal success = highest)
    # This avoids the shaping cliff that punished fossilization
    stage_potentials = {
        2: 5.0,  # GERMINATED - just started
        3: 10.0,  # TRAINING - actively learning
        4: 15.0,  # BLENDING - about to integrate
        5: 20.0,  # SHADOWING - monitoring integration
        6: 25.0,  # PROBATIONARY - almost done
        7: 35.0,  # FOSSILIZED - closing bonus for banked value
    }

    base_potential = stage_potentials.get(seed_stage, 0.0)
    progress_bonus = min(epochs_in_stage * 0.5, 3.0)

    return base_potential + progress_bonus


# =============================================================================
# Loss-Primary Reward (Phase 2)
# =============================================================================

# Stage potentials for PBRS (monotonically increasing toward FOSSILIZED)
_STAGE_POTENTIALS = {
    0: 0.0,  # UNKNOWN
    1: 0.0,  # DORMANT
    2: 1.0,  # GERMINATED
    3: 2.0,  # TRAINING
    4: 3.0,  # BLENDING
    5: 4.0,  # SHADOWING
    6: 5.0,  # PROBATIONARY
    7: 7.0,  # FOSSILIZED (highest)
}


def compute_pbrs_stage_bonus(
    seed_info: SeedInfo,
    config: LossRewardConfig,
    gamma: float = 0.99,
) -> float:
    """PBRS-compatible stage bonus using potential function."""
    previous_stage = seed_info.previous_stage

    current_potential = _STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    previous_potential = _STAGE_POTENTIALS.get(previous_stage, 0.0)

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

    # Compute rent with grace period
    if host_params > 0 and total_params > 0:
        in_grace = False
        if seed_info is not None:
            in_grace = seed_info.seed_age_epochs < config.grace_epochs
        if not in_grace:
            growth_ratio = 1.0 + (total_params / host_params)
            scaled_cost = (growth_ratio**config.compute_rent_exponent) - 1.0
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
    "RewardConfig",
    "LossRewardConfig",
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_loss_reward",
    "compute_seed_potential",
    "get_intervention_cost",
    "INTERVENTION_COSTS_BY_NAME",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_SHADOWING",
    "STAGE_PROBATIONARY",
]
