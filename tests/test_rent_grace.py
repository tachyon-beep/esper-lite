"""Tests for rent grace period."""


def test_rent_not_applied_during_grace():
    """No rent during grace period."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig, SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()
    config.grace_epochs = 3

    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        epochs_in_stage=1,
        seed_params=50000,
        seed_age_epochs=1,
    )

    reward_grace = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=5,
        max_epochs=25,
        total_params=50000,
        host_params=100000,
        config=config,
    )

    seed_info_old = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        epochs_in_stage=5,
        seed_params=50000,
        seed_age_epochs=5,
    )

    reward_no_grace = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=seed_info_old,
        epoch=9,
        max_epochs=25,
        total_params=50000,
        host_params=100000,
        config=config,
    )

    assert reward_grace > reward_no_grace


def test_rent_applied_after_grace():
    """Rent applied after grace period ends."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig, SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()
    config.grace_epochs = 3

    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        epochs_in_stage=5,
        seed_params=50000,
        seed_age_epochs=5,
    )

    reward_with_params = compute_loss_reward(
        action=0,
        loss_delta=0.0,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=50000,
        host_params=100000,
        config=config,
    )

    reward_no_params = compute_loss_reward(
        action=0,
        loss_delta=0.0,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=0,
        host_params=100000,
        config=config,
    )

    assert reward_no_params > reward_with_params
