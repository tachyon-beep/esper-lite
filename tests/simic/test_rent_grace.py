"""Tests for rent grace period."""


def test_rent_not_applied_during_grace():
    """No rent during grace period."""
    from esper.leyline import LossRewardConfig, SeedStage
    from esper.simic.rewards import compute_loss_reward, SeedInfo

    config = LossRewardConfig.default()
    config.grace_epochs = 3

    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=1,
        seed_params=50000,
        seed_age_epochs=1,
    )

    # total_params > host_params triggers rent (seed adds 50K overhead)
    reward_grace = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=5,
        max_epochs=25,
        total_params=150000,  # host(100K) + seed(50K) = 150K
        host_params=100000,
        config=config,
    )

    seed_info_old = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=5,
        seed_params=50000,
        seed_age_epochs=5,
    )

    # Same params but seed is older (past grace period) - should have rent
    reward_no_grace = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=seed_info_old,
        epoch=9,
        max_epochs=25,
        total_params=150000,  # host(100K) + seed(50K) = 150K
        host_params=100000,
        config=config,
    )

    # Grace period (age=1) should have higher reward than post-grace (age=5)
    assert reward_grace > reward_no_grace


def test_rent_applied_after_grace():
    """Rent applied after grace period ends."""
    from esper.leyline import LossRewardConfig, SeedStage
    from esper.simic.rewards import compute_loss_reward, SeedInfo

    config = LossRewardConfig.default()
    config.grace_epochs = 3

    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=5,
        seed_params=50000,
        seed_age_epochs=5,
    )

    # With seed overhead: total > host, so rent applies
    reward_with_params = compute_loss_reward(
        action=0,
        loss_delta=0.0,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=150000,  # host(100K) + seed(50K) = 150K overhead
        host_params=100000,
        config=config,
    )

    # No seed overhead: total = host, so no rent
    reward_no_overhead = compute_loss_reward(
        action=0,
        loss_delta=0.0,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=100000,  # No overhead (total = host)
        host_params=100000,
        config=config,
    )

    # No overhead should have higher reward (no rent penalty)
    assert reward_no_overhead > reward_with_params
