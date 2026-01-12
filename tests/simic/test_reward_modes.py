"""Tests for reward mode enum and sparse reward functions."""

from esper.leyline import LifecycleOp, LossRewardConfig
from esper.simic.rewards import (
    ContributionRewardConfig,
    ContributionRewardInputs,
    LossRewardInputs,
    RewardFamily,
    RewardMode,
    compute_reward_for_family,
)


def test_reward_mode_enum_exists():
    """RewardMode enum includes all supported modes."""
    assert RewardMode.SHAPED.value == "shaped"
    assert RewardMode.ESCROW.value == "escrow"
    assert RewardMode.BASIC.value == "basic"
    assert RewardMode.SPARSE.value == "sparse"
    assert RewardMode.MINIMAL.value == "minimal"
    assert RewardMode.SIMPLIFIED.value == "simplified"


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
    assert config.early_prune_threshold == 5
    assert config.early_prune_penalty == -0.1


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
        committed_val_acc=75.0,
        fossilized_seed_params=100_000,
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
        committed_val_acc=80.0,
        fossilized_seed_params=100_000,
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
        committed_val_acc=80.0,
        fossilized_seed_params=100_000,
        epoch=25,
        max_epochs=25,
        config=config,
    )

    # H10 FIX: base reward clamped to [-1, 1] BEFORE scaling
    # base = 0.78 (already in [-1, 1]), scaled = 2.5 * 0.78 = 1.95
    # Final reward in [-scale, scale] = [-2.5, 2.5]
    assert abs(reward - 1.95) < 1e-6, f"Scale should be effective, got {reward}"


def test_minimal_reward_no_penalty_for_old_prune():
    """MINIMAL mode: no penalty for pruning old seeds."""
    from esper.simic.rewards import compute_minimal_reward
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_prune_threshold=5,
        early_prune_penalty=-0.1,
    )

    # Cull a seed that's old enough (age >= threshold)
    reward = compute_minimal_reward(
        committed_val_acc=75.0,
        fossilized_seed_params=100_000,
        epoch=10,
        max_epochs=25,
        action=LifecycleOp.PRUNE,
        seed_age=5,  # Exactly at threshold
        config=config,
    )

    # Non-terminal, no penalty -> 0.0
    assert reward == 0.0


def test_minimal_reward_penalty_for_young_prune():
    """MINIMAL mode: penalty for pruning young seeds."""
    from esper.simic.rewards import compute_minimal_reward
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_prune_threshold=5,
        early_prune_penalty=-0.1,
    )

    # Cull a seed that's too young
    reward = compute_minimal_reward(
        committed_val_acc=75.0,
        fossilized_seed_params=100_000,
        epoch=10,
        max_epochs=25,
        action=LifecycleOp.PRUNE,
        seed_age=3,  # Below threshold
        config=config,
    )

    # Non-terminal but penalty applies -> -0.1
    assert reward == config.early_prune_penalty


def test_compute_reward_for_family_dispatches_contribution():
    """Reward family selects contribution reward path."""
    contrib_cfg = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)
    inputs = ContributionRewardInputs(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        committed_val_acc=70.0,
        fossilized_seed_params=0,
        config=contrib_cfg,
    )
    reward = compute_reward_for_family(RewardFamily.CONTRIBUTION, inputs)
    assert isinstance(reward, float)


def test_compute_reward_for_family_dispatches_loss():
    """Reward family selects loss reward path."""
    loss_cfg = LossRewardConfig.default()

    inputs = LossRewardInputs(
        action=LifecycleOp.WAIT,
        loss_delta=-0.5,
        val_loss=1.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=loss_cfg,
    )
    reward = compute_reward_for_family(RewardFamily.LOSS, inputs)
    # Loss reward should reward improvement (negative delta -> positive reward)
    assert reward > 0


def test_compute_reward_shaped_mode():
    """compute_reward dispatches to shaped reward by default."""
    from esper.simic.rewards import compute_reward
    config = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=10,
            max_epochs=25,
            total_params=100_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=0.0,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            config=config,
        )
    )

    # Shaped reward with no seed should be non-zero (rent, etc.)
    assert isinstance(reward, float)


def test_compute_reward_sparse_mode():
    """compute_reward dispatches to sparse reward when mode is SPARSE."""
    from esper.simic.rewards import compute_reward
    config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

    # Non-terminal epoch
    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=10,
            max_epochs=25,
            total_params=100_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=0.0,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            config=config,
        )
    )

    # Sparse reward at non-terminal = 0.0
    assert reward == 0.0


def test_compute_reward_minimal_mode():
    """compute_reward dispatches to minimal reward when mode is MINIMAL."""
    from esper.simic.rewards import compute_reward, SeedInfo
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_prune_threshold=5,
        early_prune_penalty=-0.1,
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

    # Prune action on young seed
    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.PRUNE,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.5,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            config=config,
        )
    )

    # Should get early-prune penalty
    assert reward == -0.1


def test_compute_reward_basic_mode_has_rent_every_step():
    """BASIC mode: per-step rent prevents train-then-purge gaming.

    With per-step rent, the policy pays rent during training epochs,
    not just at terminal. This prevents gaming where the policy
    accumulates rewards then prunes to avoid rent.
    """
    from esper.simic.rewards import compute_reward
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
            epoch=1,  # Non-terminal
            max_epochs=25,
            total_params=100_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=1.0,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            effective_seed_params=50_000,  # 10% of budget
            config=config,
        )
    )

    # Non-terminal: rent is paid every step, no accuracy bonus
    # rent = 0.1 * (50_000 / 500_000) = 0.01
    # No seed_info = no PBRS
    assert abs(reward - (-0.01)) < 1e-8


def test_compute_reward_basic_mode_terminal_includes_accuracy():
    """BASIC mode: terminal step includes accuracy bonus + rent."""
    from esper.simic.rewards import compute_reward
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
            epoch=25,  # Terminal
            max_epochs=25,
            total_params=100_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=1.0,  # +1 percentage point
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            effective_seed_params=50_000,  # 10% of param_budget
            config=config,
        )
    )

    # Terminal: accuracy bonus + rent
    # accuracy = 5.0 * (1/100) = 0.05
    # rent = 0.1 * (50_000 / 500_000) = 0.01
    # reward = 0.05 - 0.01 = 0.04
    assert abs(reward - 0.04) < 1e-8


def test_basic_reward_direct_per_step_rent():
    """Direct compute_basic_reward: rent is paid every step."""
    from esper.simic.rewards import compute_basic_reward

    config = ContributionRewardConfig(
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
    )

    # Non-terminal epoch - rent is still paid
    reward, rent, growth, pbrs, fossilize, new_drip, drip_epoch = compute_basic_reward(
        acc_delta=2.0,
        effective_seed_params=100_000,
        total_params=200_000,
        host_params=100_000,
        config=config,
        epoch=5,
        max_epochs=25,
    )

    # rent = 0.1 * (100_000 / 500_000) = 0.02
    # No seed_info = no PBRS, no accuracy (non-terminal)
    assert abs(rent - 0.02) < 1e-8
    assert abs(reward - (-0.02)) < 1e-8
    assert pbrs == 0.0  # No seed_info
    assert fossilize == 0.0  # No FOSSILIZE action
    assert new_drip is None  # No fossilize
    assert drip_epoch == 0.0  # No drip states


def test_basic_reward_direct_terminal_includes_accuracy():
    """Direct compute_basic_reward: terminal includes accuracy + rent."""
    from esper.simic.rewards import compute_basic_reward

    config = ContributionRewardConfig(
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
    )

    # Terminal epoch
    reward, rent, growth, pbrs, fossilize, new_drip, drip_epoch = compute_basic_reward(
        acc_delta=2.0,  # 2 percentage points improvement
        effective_seed_params=100_000,  # 20% of budget
        total_params=200_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=25,
    )

    # Expected:
    # accuracy = 5.0 * (2.0 / 100) = 0.10
    # rent = 0.1 * (100_000 / 500_000) = 0.02
    # reward = 0.10 - 0.02 = 0.08
    assert abs(reward - 0.08) < 1e-8
    assert abs(rent - 0.02) < 1e-8
    assert growth > 0  # growth_ratio = 100_000 / 100_000 = 1.0
    assert fossilize == 0.0  # No FOSSILIZE action
    assert new_drip is None  # No fossilize
    assert drip_epoch == 0.0  # No drip states


def test_basic_reward_pbrs_provides_state_differentiation():
    """BASIC mode: PBRS gives critic state differentiation.

    The key insight from DRL expert: PBRS is policy-invariant
    (Ng et al., 1999) and provides the critic with intermediate
    signal to distinguish states without changing optimal policy.
    """
    from esper.simic.rewards import compute_basic_reward, SeedInfo
    from esper.simic.rewards.types import STAGE_TRAINING

    config = ContributionRewardConfig(
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
        pbrs_weight=0.3,
    )

    # Seed in TRAINING stage
    seed_info = SeedInfo(
        stage=STAGE_TRAINING,
        improvement_since_stage_start=0.5,
        total_improvement=1.0,
        epochs_in_stage=3,
        seed_params=50_000,
        previous_stage=2,  # GERMINATED
        previous_epochs_in_stage=1,
        seed_age_epochs=4,
    )

    reward_with_seed, rent, growth, pbrs, _, _, _ = compute_basic_reward(
        acc_delta=1.0,
        effective_seed_params=50_000,
        total_params=150_000,
        host_params=100_000,
        config=config,
        epoch=10,
        max_epochs=25,
        seed_info=seed_info,
    )

    reward_no_seed, _, _, pbrs_no_seed, _, _, _ = compute_basic_reward(
        acc_delta=1.0,
        effective_seed_params=0,  # No seed
        total_params=100_000,
        host_params=100_000,
        config=config,
        epoch=10,
        max_epochs=25,
        seed_info=None,
    )

    # PBRS should be non-zero with seed
    assert pbrs != 0.0
    assert pbrs_no_seed == 0.0

    # Critic can now distinguish states with vs without seeds
    assert reward_with_seed != reward_no_seed


def test_basic_reward_prevents_train_then_purge():
    """Verify per-step rent prevents train-then-purge gaming.

    The gaming pattern: accumulate dense attribution rewards during training,
    then prune all seeds at the end to avoid rent.

    With per-step rent, keeping a seed costs rent EVERY step, so the
    cumulative rent paid during training is significant.
    """
    from esper.simic.rewards import compute_basic_reward

    config = ContributionRewardConfig(
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
    )

    # Simulate 24 epochs of having a seed
    cumulative_rent_paid = 0.0
    for epoch in range(1, 25):  # epochs 1-24 (non-terminal)
        reward, rent, _, _, _, _, _ = compute_basic_reward(
            acc_delta=0.0,  # No improvement yet
            effective_seed_params=50_000,
            total_params=150_000,
            host_params=100_000,
            config=config,
            epoch=epoch,
            max_epochs=25,
        )
        cumulative_rent_paid += rent

    # Rent is 0.1 * (50_000 / 500_000) = 0.01 per step
    # Over 24 steps: 0.24
    assert abs(cumulative_rent_paid - 0.24) < 1e-8

    # Train-then-purge now pays significant rent during training!
    # This is the key fix that prevents gaming.


def test_parallel_env_state_has_host_max_acc():
    """ParallelEnvState tracks host_max_acc."""
    from esper.simic.training.vectorized import ParallelEnvState
    import inspect

    # Check the dataclass has the field
    hints = inspect.get_annotations(ParallelEnvState)
    assert "host_max_acc" in hints, "ParallelEnvState should have host_max_acc field"


def test_basic_plus_mode_dispatches_through_compute_reward() -> None:
    """BASIC_PLUS mode dispatches correctly through compute_reward."""
    from esper.simic.rewards.contribution import ContributionRewardConfig, RewardMode
    from esper.simic.rewards.rewards import compute_reward
    from esper.simic.rewards.types import ContributionRewardInputs, SeedInfo
    from esper.leyline import SeedStage

    config = ContributionRewardConfig(reward_mode=RewardMode.BASIC_PLUS)

    # BASIC_PLUS should have drip enabled by default
    assert config.drip_fraction == 0.7

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    inputs = ContributionRewardInputs(
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        val_acc=0.85,
        seed_info=seed_info,
        epoch=20,
        max_epochs=150,
        total_params=110_000,
        host_params=100_000,
        acc_at_germination=0.70,
        acc_delta=5.0,
        config=config,
        return_components=False,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    # Should NOT raise - BASIC_PLUS is now handled
    reward = compute_reward(inputs)
    assert isinstance(reward, float)
    assert reward > 0  # FOSSILIZE with positive contribution should yield positive reward


def test_basic_plus_mode_creates_drip_state() -> None:
    """BASIC_PLUS mode creates drip state on valid FOSSILIZE."""
    from esper.simic.rewards.contribution import (
        ContributionRewardConfig,
        RewardMode,
        compute_basic_reward,
    )
    from esper.simic.rewards.types import SeedInfo
    from esper.leyline import SeedStage

    config = ContributionRewardConfig(reward_mode=RewardMode.BASIC_PLUS)

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    (
        reward,
        rent_penalty,
        growth_ratio,
        pbrs_bonus,
        fossilize_bonus,
        new_drip_state,
        drip_this_epoch,
    ) = compute_basic_reward(
        acc_delta=5.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=20,
        max_epochs=150,
        seed_info=seed_info,
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    # Should create drip state for valid fossilization
    assert new_drip_state is not None
    assert new_drip_state.seed_id == "test-seed"
    assert new_drip_state.slot_id == "r0c1"
    assert new_drip_state.drip_total > 0
    # Immediate bonus should be 30% of full bonus (drip_fraction=0.7)
    assert fossilize_bonus > 0
