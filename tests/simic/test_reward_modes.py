"""Tests for reward mode enum and sparse reward functions."""

from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import RewardMode, RewardFamily, ContributionRewardConfig, compute_reward_for_family, LossRewardConfig


def test_reward_mode_enum_exists():
    """RewardMode enum has three modes."""
    assert RewardMode.SHAPED.value == "shaped"
    assert RewardMode.SPARSE.value == "sparse"
    assert RewardMode.MINIMAL.value == "minimal"


def test_reward_family_enum_exists():
    """RewardFamily enum has two families."""
    assert RewardFamily.CONTRIBUTION.value == "contribution"
    assert RewardFamily.LOSS.value == "loss"


def test_config_has_sparse_fields():
    """ContributionRewardConfig has sparse reward fields."""
    config = ContributionRewardConfig()

    # Default mode is SHAPED
    assert config.reward_mode == RewardMode.SHAPED

    # Sparse reward parameters
    assert config.param_budget == 500_000
    assert config.param_penalty_weight == 0.1
    assert config.sparse_reward_scale == 1.0  # DRL Expert: try 2.0-3.0 if learning fails

    # Minimal mode parameters
    assert config.early_cull_threshold == 5
    assert config.early_cull_penalty == -0.1


def test_reward_mode_exported():
    """RewardMode is in module __all__."""
    from esper.simic import rewards
    assert "RewardMode" in rewards.__all__


def test_sparse_reward_zero_before_terminal():
    """Sparse reward is 0.0 for non-terminal epochs."""
    from esper.simic.rewards import compute_sparse_reward
    config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

    # Epoch 10 of 25 - not terminal
    reward = compute_sparse_reward(
        host_max_acc=75.0,
        total_params=100_000,
        epoch=10,
        max_epochs=25,
        config=config,
    )
    assert reward == 0.0


def test_sparse_reward_nonzero_at_terminal():
    """Sparse reward is non-zero at terminal epoch."""
    from esper.simic.rewards import compute_sparse_reward
    config = ContributionRewardConfig(
        reward_mode=RewardMode.SPARSE,
        param_budget=500_000,
        param_penalty_weight=0.1,
        sparse_reward_scale=1.0,
    )

    # Epoch 25 of 25 - terminal
    reward = compute_sparse_reward(
        host_max_acc=80.0,
        total_params=100_000,
        epoch=25,
        max_epochs=25,
        config=config,
    )

    # Expected: 1.0 * ((80/100) - 0.1 * (100_000 / 500_000)) = 0.8 - 0.02 = 0.78
    assert abs(reward - 0.78) < 0.001


def test_sparse_reward_with_scale():
    """Sparse reward respects scale parameter."""
    from esper.simic.rewards import compute_sparse_reward
    config = ContributionRewardConfig(
        reward_mode=RewardMode.SPARSE,
        param_budget=500_000,
        param_penalty_weight=0.1,
        sparse_reward_scale=2.5,  # DRL Expert recommendation for credit assignment
    )

    reward = compute_sparse_reward(
        host_max_acc=80.0,
        total_params=100_000,
        epoch=25,
        max_epochs=25,
        config=config,
    )

    # H10 FIX: base reward clamped to [-1, 1] BEFORE scaling
    # base = 0.78 (already in [-1, 1]), scaled = 2.5 * 0.78 = 1.95
    # Final reward in [-scale, scale] = [-2.5, 2.5]
    assert abs(reward - 1.95) < 1e-6, f"Scale should be effective, got {reward}"


def test_minimal_reward_no_penalty_for_old_cull():
    """MINIMAL mode: no penalty for culling old seeds."""
    from esper.simic.rewards import compute_minimal_reward
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_cull_threshold=5,
        early_cull_penalty=-0.1,
    )

    # Cull a seed that's old enough (age >= threshold)
    reward = compute_minimal_reward(
        host_max_acc=75.0,
        total_params=100_000,
        epoch=10,
        max_epochs=25,
        action=LifecycleOp.CULL,
        seed_age=5,  # Exactly at threshold
        config=config,
    )

    # Non-terminal, no penalty -> 0.0
    assert reward == 0.0


def test_minimal_reward_penalty_for_young_cull():
    """MINIMAL mode: penalty for culling young seeds."""
    from esper.simic.rewards import compute_minimal_reward
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_cull_threshold=5,
        early_cull_penalty=-0.1,
    )

    # Cull a seed that's too young
    reward = compute_minimal_reward(
        host_max_acc=75.0,
        total_params=100_000,
        epoch=10,
        max_epochs=25,
        action=LifecycleOp.CULL,
        seed_age=3,  # Below threshold
        config=config,
    )

    # Non-terminal but penalty applies -> -0.1
    assert reward == config.early_cull_penalty


def test_compute_reward_for_family_dispatches_contribution():
    """Reward family selects contribution reward path."""
    contrib_cfg = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)
    loss_cfg = LossRewardConfig.default()
    reward = compute_reward_for_family(
        RewardFamily.CONTRIBUTION,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        contribution_config=contrib_cfg,
        loss_config=loss_cfg,
        loss_delta=0.1,
        val_loss=1.0,
    )
    assert isinstance(reward, float)


def test_compute_reward_for_family_dispatches_loss():
    """Reward family selects loss reward path."""
    contrib_cfg = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)
    loss_cfg = LossRewardConfig.default()

    reward = compute_reward_for_family(
        RewardFamily.LOSS,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        contribution_config=contrib_cfg,
        loss_config=loss_cfg,
        loss_delta=-0.5,
        val_loss=1.0,
    )
    # Loss reward should reward improvement (negative delta -> positive reward)
    assert reward > 0


def test_compute_reward_shaped_mode():
    """compute_reward dispatches to shaped reward by default."""
    from esper.simic.rewards import compute_reward
    config = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)

    reward = compute_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        config=config,
    )

    # Shaped reward with no seed should be non-zero (rent, etc.)
    assert isinstance(reward, float)


def test_compute_reward_sparse_mode():
    """compute_reward dispatches to sparse reward when mode is SPARSE."""
    from esper.simic.rewards import compute_reward
    config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

    # Non-terminal epoch
    reward = compute_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        config=config,
    )

    # Sparse reward at non-terminal = 0.0
    assert reward == 0.0


def test_compute_reward_minimal_mode():
    """compute_reward dispatches to minimal reward when mode is MINIMAL."""
    from esper.simic.rewards import compute_reward, SeedInfo
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_cull_threshold=5,
        early_cull_penalty=-0.1,
    )

    # Create a young seed
    seed_info = SeedInfo(
        stage=3,  # TRAINING
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=2,
        seed_params=10_000,
        previous_stage=2,
        previous_epochs_in_stage=1,
        seed_age_epochs=3,  # Young seed
    )

    # Cull action on young seed
    reward = compute_reward(
        action=LifecycleOp.CULL,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=110_000,
        host_params=100_000,
        acc_at_germination=65.0,
        acc_delta=0.5,
        config=config,
    )

    # Should get early-cull penalty
    assert reward == -0.1


def test_parallel_env_state_has_host_max_acc():
    """ParallelEnvState tracks host_max_acc."""
    from esper.simic.training.vectorized import ParallelEnvState
    import inspect

    # Check the dataclass has the field
    hints = inspect.get_annotations(ParallelEnvState)
    assert "host_max_acc" in hints, "ParallelEnvState should have host_max_acc field"
