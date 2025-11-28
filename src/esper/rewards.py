"""Reward Computation for Tamiyo Seed Lifecycle Controller.

This module consolidates reward functions used across:
- Online PPO training (simic_ppo.py)
- Offline data generation (datagen/generate.py)
- Offline RL (simic_iql.py)

The reward design follows these principles:
1. Accuracy improvement is the primary signal
2. Lifecycle progression bonuses encourage exploration
3. Action-specific shaping guides decision making
4. Potential-based shaping maintains optimal policy invariance

Usage:
    from esper.rewards import compute_shaped_reward, RewardConfig

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

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple


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

    # Action-specific weights
    germinate_no_seed_bonus: float = 0.3
    germinate_early_bonus: float = 0.2
    germinate_with_seed_penalty: float = -0.3

    advance_good_bonus: float = 0.5
    advance_premature_penalty: float = -0.2
    advance_blending_bonus: float = 0.4
    advance_no_seed_penalty: float = -0.2

    cull_failing_bonus: float = 0.3
    cull_acceptable_bonus: float = 0.1
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

    @staticmethod
    def default() -> "RewardConfig":
        """Return default configuration."""
        return RewardConfig()


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

    @staticmethod
    def from_seed_state(seed_state) -> "SeedInfo":
        """Convert from kasmina.SeedState to SeedInfo."""
        if seed_state is None:
            return None
        metrics = seed_state.metrics
        improvement = 0.0
        if metrics:
            improvement = metrics.current_val_accuracy - metrics.accuracy_at_stage_start
        return SeedInfo(
            stage=seed_state.stage.value,
            improvement_since_stage_start=improvement,
            epochs_in_stage=seed_state.epochs_in_stage,
        )


# Stage constants (match SeedStage IntEnum values)
STAGE_TRAINING = 3
STAGE_BLENDING = 4
STAGE_FOSSILIZED = 7


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
        config: Reward configuration (uses default if None)

    Returns:
        Shaped reward value
    """
    if config is None:
        config = _DEFAULT_CONFIG

    reward = 0.0

    # Base: accuracy improvement
    reward += acc_delta * config.acc_delta_weight

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
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
]
