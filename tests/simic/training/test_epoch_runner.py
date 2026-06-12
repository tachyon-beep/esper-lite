from esper.leyline import LifecycleOp, LossRewardConfig
from esper.simic.rewards import ContributionRewardConfig, RewardMode
from esper.simic.training.epoch_runner import EpochState


def test_epoch_state_create_allocates_per_env_contracts() -> None:
    env_reward_configs = [
        ContributionRewardConfig(reward_mode=RewardMode.BASIC),
        ContributionRewardConfig(reward_mode=RewardMode.ESCROW),
    ]
    loss_reward_config = LossRewardConfig.default()

    state = EpochState.create(
        n_envs=2,
        max_epochs=7,
        env_reward_configs=env_reward_configs,
        loss_reward_config=loss_reward_config,
    )

    assert state.env_final_accs == [0.0, 0.0]
    assert state.env_total_rewards == [0.0, 0.0]
    assert len(state.reward_summary_accum) == 2
    assert len(state.action_specs) == 2
    assert len(state.action_outcomes) == 2
    assert len(state.action_mask_flags) == 2
    assert state.env_rollback_occurred == [False, False]
    assert state.raw_states_for_normalizer_update == []
    assert state.throughput_step_time_ms_sum == 0.0
    assert state.throughput_dataloader_wait_ms_sum == 0.0

    first_contribution = state.contribution_reward_inputs[0]
    second_contribution = state.contribution_reward_inputs[1]
    assert first_contribution.action is LifecycleOp.WAIT
    assert first_contribution.max_epochs == 7
    assert first_contribution.host_params == 1
    assert first_contribution.config is env_reward_configs[0]
    assert second_contribution.config is env_reward_configs[1]

    for loss_input in state.loss_reward_inputs:
        assert loss_input.action is LifecycleOp.WAIT
        assert loss_input.max_epochs == 7
        assert loss_input.host_params == 1
        assert loss_input.config is loss_reward_config


def test_epoch_state_reset_for_new_batch_reuses_and_resizes_allocations() -> None:
    env_reward_configs = [ContributionRewardConfig(reward_mode=RewardMode.BASIC)]
    state = EpochState.create(
        n_envs=1,
        max_epochs=3,
        env_reward_configs=env_reward_configs,
        loss_reward_config=LossRewardConfig.default(),
    )
    original_action_spec = state.action_specs[0]

    state.env_final_accs[0] = 0.75
    state.env_total_rewards[0] = 4.5
    state.env_rollback_occurred[0] = True
    state.raw_states_for_normalizer_update.append({"state": "seen"})
    state.throughput_step_time_ms_sum = 12.0
    state.throughput_dataloader_wait_ms_sum = 3.0

    state.reset_for_new_batch(n_envs=3)

    assert state.action_specs[0] is original_action_spec
    assert len(state.env_final_accs) == 3
    assert len(state.env_total_rewards) == 3
    assert len(state.reward_summary_accum) == 3
    assert len(state.action_specs) == 3
    assert len(state.action_outcomes) == 3
    assert len(state.action_mask_flags) == 3
    assert state.env_final_accs == [0.0, 0.0, 0.0]
    assert state.env_total_rewards == [0.0, 0.0, 0.0]
    assert state.env_rollback_occurred == [False, False, False]
    assert state.raw_states_for_normalizer_update == []
    assert state.throughput_step_time_ms_sum == 0.0
    assert state.throughput_dataloader_wait_ms_sum == 0.0
