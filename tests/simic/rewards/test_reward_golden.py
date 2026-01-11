"""Golden-value tests for reward outputs."""

from __future__ import annotations

import pytest

from esper.leyline import LifecycleOp, LossRewardConfig, SeedStage
from esper.simic.rewards import (
    ContributionRewardConfig,
    ContributionRewardInputs,
    LossRewardInputs,
    RewardMode,
    SeedInfo,
    compute_loss_reward,
    compute_reward,
)


def test_contribution_reward_golden_simplified_pbrs() -> None:
    """Simplified reward: PBRS + action cost stays stable."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=2,
        seed_params=10_000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=5,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.PRUNE,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=10,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.1,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            num_contributing_fossilized=0,
            config=config,
        )
    )

    assert reward == pytest.approx(0.07610000000000011, abs=1e-6)


def test_contribution_reward_golden_basic_mode() -> None:
    """Basic reward: rent only for non-terminal steps, accuracy at terminal.

    Non-terminal WAIT with no seed: only rent penalty applies.
    rent = param_penalty_weight * (effective_seed_params / param_budget)
         = 0.1 * (25_000 / 500_000) = 0.005
    reward = -rent = -0.005
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            total_params=125_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=2.0,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            effective_seed_params=25_000,
            config=config,
        )
    )

    # DRL Expert design: accuracy bonus only at terminal epoch
    # Non-terminal = rent only = -0.005
    assert reward == pytest.approx(-0.005, abs=1e-6)


def test_loss_reward_golden_terminal() -> None:
    """Loss reward: normalized loss delta + terminal bonus stays stable."""
    config = LossRewardConfig.default()
    reward = compute_loss_reward(
        LossRewardInputs(
            action=LifecycleOp.WAIT,
            loss_delta=-0.05,
            val_loss=1.5,
            seed_info=None,
            epoch=10,
            max_epochs=10,
            total_params=120,
            host_params=100,
            config=config,
        )
    )

    assert reward == pytest.approx(2.890883922160302, abs=1e-6)


def test_basic_mode_prune_forfeits_pbrs() -> None:
    """PRUNE in BASIC mode forfeits accumulated PBRS potential.

    DRL Expert review 2026-01-12: PRUNE must be net-negative to prevent
    GERMINATE→train→PRUNE gaming. The forfeiture is:
        γ × Φ(no_seed) - Φ(current) = 0.995 × 0 - Φ(TRAINING + progress)
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        pbrs_weight=0.3,
        epoch_progress_bonus=0.3,
        gamma=0.995,
    )
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=1.0,
        total_improvement=2.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=6,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.PRUNE,
            seed_contribution=1.5,
            val_acc=72.0,
            seed_info=seed_info,
            epoch=6,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=2.0,
            committed_val_acc=72.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # PBRS forfeiture: -0.3 * (STAGE_POTENTIAL[TRAINING] + min(5*0.3, 2.0))
    # STAGE_POTENTIAL[TRAINING] = 2.0, epoch_bonus = min(1.5, 2.0) = 1.5
    # forfeiture = -0.3 * (2.0 + 1.5) = -1.05
    # rent = 0.1 * (10_000 / 500_000) = 0.002
    # total ≈ -1.05 - 0.002 = -1.052
    assert reward < 0, "PRUNE should be net-negative in BASIC mode"
    assert reward == pytest.approx(-1.052, abs=0.01)


def test_basic_mode_fossilize_good_seed() -> None:
    """FOSSILIZE in BASIC mode rewards f(improvement, contribution).

    DRL Expert review 2026-01-12: Use harmonic mean of improvement and
    contribution. Both must be positive for meaningful reward.
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
        attribution_formula="harmonic",
    )
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,  # Positive improvement
        epochs_in_stage=5,  # Full legitimacy
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=5.0,  # Positive contribution
            val_acc=75.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=5.0,
            committed_val_acc=75.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # harmonic(5.0, 5.0) = 5.0, legitimacy = 1.0
    # fossilize_bonus = (0.3 + 0.5 * 5.0) * 1.0 = 2.8
    # pbrs_bonus for HOLDING stage ≈ 0.12
    # rent = 0.002
    # total ≈ 2.8 + 0.12 - 0.002 ≈ 2.92
    assert reward > 2.0, "Good seed should get substantial reward"
    assert reward == pytest.approx(2.918, abs=0.05)


def test_basic_mode_fossilize_ransomware_penalty() -> None:
    """FOSSILIZE in BASIC mode penalizes ransomware seeds.

    DRL Expert review 2026-01-12: Seeds with high contribution but
    negative improvement are gaming - they made themselves important
    without helping the host.
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        basic_fossilize_invalid_penalty=-0.5,
        attribution_formula="harmonic",
    )
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=-1.0,
        total_improvement=-3.0,  # Negative improvement (ransomware!)
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=15.0,  # High contribution (made itself important)
            val_acc=67.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=-3.0,
            committed_val_acc=67.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # Ransomware: improvement=-3.0 < 0 AND contribution=15.0 > 0.1
    # fossilize_bonus = -0.5 (invalid penalty)
    # This seed should NOT be rewarded despite high contribution
    assert reward < 0, "Ransomware seed should get penalty"
