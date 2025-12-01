"""Tests for loss-primary reward computation."""


def test_compute_loss_reward_basic():
    """Basic loss reward: lower loss = positive reward."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    reward = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    assert reward > 0


def test_compute_loss_reward_regression_penalized():
    """Loss regression gives negative reward."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    reward = compute_loss_reward(
        action=0,
        loss_delta=0.1,
        val_loss=2.5,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    assert reward < 0


def test_asymmetric_regression_penalty():
    """Regression penalty is scaled down (asymmetric)."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    improvement_reward = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        config=config,
    )
    regression_reward = compute_loss_reward(
        action=0,
        loss_delta=0.1,
        val_loss=2.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    assert abs(regression_reward) < abs(improvement_reward)


def test_compute_rent():
    """Compute rent penalizes excess parameters."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    no_params_reward = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        total_params=0,
        host_params=100000,
        config=config,
    )
    with_params_reward = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        total_params=50000,
        host_params=100000,
        config=config,
    )

    assert with_params_reward < no_params_reward


def test_pbrs_stage_bonus():
    """PBRS stage bonus rewards stage progression."""
    from esper.simic.rewards import compute_pbrs_stage_bonus, LossRewardConfig
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()

    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        previous_stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=2.0,
        epochs_in_stage=0,
    )

    bonus = compute_pbrs_stage_bonus(seed_info, config)

    assert bonus > 0


def test_pbrs_stage_bonus_fossilized_highest():
    """FOSSILIZED has highest potential (terminal success)."""
    from esper.simic.rewards import compute_pbrs_stage_bonus, LossRewardConfig
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()

    fossilized_bonus = compute_pbrs_stage_bonus(
        SeedInfo(
            stage=SeedStage.FOSSILIZED.value,
            previous_stage=SeedStage.PROBATIONARY.value,
            improvement_since_stage_start=0.0,
            epochs_in_stage=0,
        ),
        config,
    )

    blending_bonus = compute_pbrs_stage_bonus(
        SeedInfo(
            stage=SeedStage.BLENDING.value,
            previous_stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=0.0,
            epochs_in_stage=0,
        ),
        config,
    )

    assert fossilized_bonus > blending_bonus
