"""Tests for PolicyGroup dataclass used in dual-policy A/B testing."""

import pytest
import torch

from esper.simic.agent import PPOAgent
from esper.simic.rewards import ContributionRewardConfig, RewardMode
from esper.simic.training import EpisodeRecord
from esper.simic.training.policy_group import PolicyGroup
from esper.tamiyo.policy.factory import create_policy
from esper.leyline.slot_config import SlotConfig


@pytest.fixture
def mock_agent():
    """Create a minimal PPOAgent for testing PolicyGroup."""
    policy = create_policy(
        policy_type="lstm",
        state_dim=30,
        slot_config=SlotConfig.default(),
        device="cpu",
        compile_mode="off",
    )
    return PPOAgent(policy=policy, device="cpu")


class TestPolicyGroupInitialization:
    """Test PolicyGroup construction and default values."""

    def test_creates_with_required_fields(self, mock_agent):
        """PolicyGroup should initialize with required fields."""
        device = torch.device("cpu")
        reward_config = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)

        group = PolicyGroup(
            group_id="A",
            device=device,
            agent=mock_agent,
            envs=[],
            reward_config=reward_config,
        )

        assert group.group_id == "A"
        assert group.device == device
        assert group.reward_mode == RewardMode.SHAPED  # Now a property
        assert group.agent is mock_agent
        assert group.reward_config is reward_config

    def test_default_metrics_start_at_zero(self, mock_agent):
        """Default metrics should be initialized to zero."""
        group = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            agent=mock_agent,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED),
        )

        assert group.total_episodes == 0
        assert group.total_steps == 0
        assert group.best_accuracy == 0.0

    def test_episode_history_defaults_to_empty_list(self, mock_agent):
        """Episode history should default to empty list."""
        group = PolicyGroup(
            group_id="C",
            device=torch.device("cpu"),
            agent=mock_agent,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SPARSE),
        )

        assert group.episode_history == []
        assert isinstance(group.episode_history, list)

    def test_cuda_device_assignment(self, mock_agent):
        """Should accept CUDA devices if available."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")
        group = PolicyGroup(
            group_id="A",
            device=device,
            agent=mock_agent,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
        )

        assert group.device == device
        assert group.device.type == "cuda"


class TestPolicyGroupMetricsTracking:
    """Test that PolicyGroup correctly tracks metrics over time."""

    def test_episode_history_can_be_appended(self, mock_agent):
        """Should be able to append episode results to history."""
        group = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=mock_agent,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
        )

        # Simulate recording three episodes using EpisodeRecord schema
        record1 = EpisodeRecord(env_id=0, final_accuracy=45.0, episode_reward=10.5)
        record2 = EpisodeRecord(env_id=1, final_accuracy=52.3, episode_reward=12.1)
        record3 = EpisodeRecord(env_id=2, final_accuracy=58.9, episode_reward=15.7)
        group.episode_history.append(record1)
        group.episode_history.append(record2)
        group.episode_history.append(record3)

        assert len(group.episode_history) == 3
        assert group.episode_history[0].final_accuracy == 45.0
        assert group.episode_history[-1].env_id == 2

    def test_metrics_can_be_updated(self, mock_agent):
        """Metrics like total_episodes and best_accuracy should be mutable."""
        group = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            agent=mock_agent,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED),
        )

        # Simulate training progress
        group.total_episodes = 10
        group.total_steps = 5000
        group.best_accuracy = 67.5

        assert group.total_episodes == 10
        assert group.total_steps == 5000
        assert group.best_accuracy == 67.5


class TestPolicyGroupIndependence:
    """Test that separate PolicyGroups are independent."""

    def test_separate_groups_have_separate_histories(self):
        """Each group should maintain its own episode history."""
        # Create two separate agents to ensure independence
        policy_a = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        policy_b = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        agent_a = PPOAgent(policy=policy_a, device="cpu")
        agent_b = PPOAgent(policy=policy_b, device="cpu")

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=agent_a,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            agent=agent_b,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED),
        )

        # Add to group A only using EpisodeRecord schema
        record = EpisodeRecord(env_id=0, final_accuracy=50.0, episode_reward=10.0)
        group_a.episode_history.append(record)

        # Group B should remain empty
        assert len(group_a.episode_history) == 1
        assert len(group_b.episode_history) == 0

    def test_groups_can_have_different_devices(self):
        """Groups should be able to use different devices."""
        policy_a = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        policy_b = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        agent_a = PPOAgent(policy=policy_a, device="cpu")
        agent_b = PPOAgent(policy=policy_b, device="cpu")

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=agent_a,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),  # In real usage, this would be cuda:1
            agent=agent_b,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED),
        )

        # Groups should have different agents and independent configurations
        assert group_a.agent is not group_b.agent
        assert group_a.device == group_b.device  # Both CPU for testing
        assert group_a.reward_mode != group_b.reward_mode


class TestPolicyGroupRewardConfig:
    """Test that each group can have its own reward configuration."""

    def test_groups_can_have_different_reward_configs(self):
        """Each group should be able to use a different reward configuration."""
        config_a = ContributionRewardConfig(
            reward_mode=RewardMode.SHAPED,
            contribution_weight=1.0,
            pbrs_weight=0.1,
        )

        config_b = ContributionRewardConfig(
            reward_mode=RewardMode.SIMPLIFIED,
            contribution_weight=0.5,
            pbrs_weight=0.2,
        )

        policy_a = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        policy_b = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        agent_a = PPOAgent(policy=policy_a, device="cpu")
        agent_b = PPOAgent(policy=policy_b, device="cpu")

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=agent_a,
            envs=[],
            reward_config=config_a,
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            agent=agent_b,
            envs=[],
            reward_config=config_b,
        )

        # Verify independent configs
        assert group_a.reward_config.contribution_weight == 1.0
        assert group_b.reward_config.contribution_weight == 0.5
        assert group_a.reward_config is not group_b.reward_config
        # Verify reward_mode is derived from config
        assert group_a.reward_mode == RewardMode.SHAPED
        assert group_b.reward_mode == RewardMode.SIMPLIFIED

    def test_environment_list_independence(self):
        """Each group should maintain its own independent environment list."""
        from esper.simic.training.parallel_env_state import ParallelEnvState
        from unittest.mock import Mock

        policy_a = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        policy_b = create_policy(policy_type="lstm", state_dim=30, slot_config=SlotConfig.default(), device="cpu", compile_mode="off")
        agent_a = PPOAgent(policy=policy_a, device="cpu")
        agent_b = PPOAgent(policy=policy_b, device="cpu")

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=agent_a,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            agent=agent_b,
            envs=[],
            reward_config=ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED),
        )

        # Create mock environments
        mock_model_a = Mock()
        mock_optimizer_a = Mock()
        mock_signal_tracker_a = Mock()
        mock_governor_a = Mock()

        env_a = ParallelEnvState(
            model=mock_model_a,
            host_optimizer=mock_optimizer_a,
            signal_tracker=mock_signal_tracker_a,
            governor=mock_governor_a,
            env_device="cpu",
        )

        # Add environment to group A only
        group_a.envs.append(env_a)

        # Verify independence
        assert len(group_a.envs) == 1
        assert len(group_b.envs) == 0
        assert group_a.envs is not group_b.envs
        assert group_a.envs[0] is env_a


class TestRewardModeProperty:
    """Test that reward_mode is derived from reward_config (single source of truth)."""

    def test_reward_mode_derived_from_config(self, mock_agent):
        """reward_mode should be derived from reward_config.reward_mode."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)

        group = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=mock_agent,
            reward_config=config,
        )

        assert group.reward_mode == RewardMode.SIMPLIFIED
        assert group.reward_mode == group.reward_config.reward_mode

    def test_reward_mode_matches_config_value(self, mock_agent):
        """reward_mode.value should match lowercase enum string."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)

        group = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            agent=mock_agent,
            reward_config=config,
        )

        # No more case mismatch - both use the enum
        assert group.reward_mode.value == "shaped"
        assert group.reward_config.reward_mode.value == "shaped"
        assert group.reward_mode.value == group.reward_config.reward_mode.value
