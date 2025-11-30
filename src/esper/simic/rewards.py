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
        action=SimicAction.GERMINATE_CONV.value,  # Action as int
        acc_delta=0.5,
        val_acc=65.0,
        seed_info=SeedInfo(...),  # or None if no active seed
        epoch=10,
        max_epochs=25,
    )

Action values (SimicAction enum):
    0: WAIT
    1: GERMINATE_CONV
    2: GERMINATE_ATTENTION
    3: GERMINATE_NORM
    4: GERMINATE_DEPTHWISE
    5: ADVANCE
    6: CULL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from esper.leyline import SeedStage


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
    germinate_early_bonus: float = 0.0    # Was 0.2
    germinate_with_seed_penalty: float = -0.3

    advance_good_bonus: float = 0.5
    advance_premature_penalty: float = -0.2
    advance_blending_bonus: float = 0.4
    advance_no_seed_penalty: float = -0.2

    cull_failing_bonus: float = 0.0       # Was 0.3
    cull_acceptable_bonus: float = 0.0    # Was 0.1
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


# =============================================================================
# Core Reward Functions
# =============================================================================

def compute_shaped_reward(
    action: int,  # SimicAction.value
    acc_delta: float,
    val_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: RewardConfig | None = None,
) -> float:
    """Compute shaped reward for seed lifecycle control.

    This function is designed for high-throughput use:
    - Uses primitive types and NamedTuples (no heavy objects)
    - All configuration is explicit (no global state)
    - Zero allocations in hot path

    Args:
        action: Action taken (0=WAIT, 1-4=GERMINATE_*, 5=ADVANCE, 6=CULL)
        acc_delta: Accuracy improvement this step
        val_acc: Current validation accuracy
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Extra params added (fossilized + active seed)
        host_params: Baseline host model params (for normalization)
        config: Reward configuration (uses default if None)

    Returns:
        Shaped reward value
    """
    if config is None:
        config = _DEFAULT_CONFIG

    reward = 0.0

    # Base: accuracy improvement
    reward += acc_delta * config.acc_delta_weight

    # Compute rent: penalize excess params proportionally
    # This is blueprint-agnostic: cost is determined by actual params added
    if host_params > 0 and total_params > 0:
        excess_params_ratio = total_params / host_params
        reward -= config.compute_rent_weight * excess_params_ratio

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

    # Action-specific shaping
    # Action values: 0=WAIT, 1-4=GERMINATE_*, 5=ADVANCE, 6=CULL
    if 1 <= action <= 4:  # Any GERMINATE variant
        reward += _germinate_shaping(seed_info, epoch, max_epochs, config)
    elif action == 5:  # ADVANCE
        reward += _advance_shaping(seed_info, config)
    elif action == 6:  # CULL
        reward += _cull_shaping(seed_info, config)
    elif action == 0:  # WAIT
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
    """Compute shaping for ADVANCE action."""
    if seed_info is None:
        return config.advance_no_seed_penalty

    stage = seed_info.stage
    improvement = seed_info.improvement_since_stage_start

    if stage == STAGE_TRAINING:
        if improvement > 0:
            return config.advance_good_bonus
        else:
            return config.advance_premature_penalty
    elif stage == STAGE_BLENDING:
        return config.advance_blending_bonus

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
    has_active = obs.get('has_active_seed', 0)
    seed_stage = obs.get('seed_stage', 0)
    epochs_in_stage = obs.get('seed_epochs_in_stage', 0)

    # No potential for inactive seeds or DORMANT (stage 1)
    if not has_active or seed_stage <= 1:
        return 0.0

    # Stage-based potential values (matching SeedStage enum values)
    # Monotonically increasing toward FOSSILIZED (terminal success = highest)
    # This avoids the shaping cliff that punished fossilization
    stage_potentials = {
        2: 5.0,   # GERMINATED - just started
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
# Intervention Costs
# =============================================================================

INTERVENTION_COSTS = {
    0: 0.0,    # WAIT
    1: -0.02,  # GERMINATE_CONV
    2: -0.02,  # GERMINATE_ATTENTION
    3: -0.02,  # GERMINATE_NORM
    4: -0.02,  # GERMINATE_DEPTHWISE
    5: -0.01,  # ADVANCE
    6: -0.005, # CULL
}


def get_intervention_cost(action: int) -> float:
    """Get intervention cost for an action.

    Small negative costs discourage unnecessary interventions,
    encouraging the agent to only act when beneficial.
    """
    return INTERVENTION_COSTS.get(action, 0.0)


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
    "compute_seed_potential",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
]
