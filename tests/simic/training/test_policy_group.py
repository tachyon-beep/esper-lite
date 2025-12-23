"""Tests for PolicyGroup dataclass used in dual-policy A/B testing."""

import pytest
import torch

from esper.simic.agent import PPOAgent
from esper.simic.rewards import ContributionRewardConfig
from esper.simic.training import PolicyGroup


@pytest.fixture
def mock_agent():
    """Create a minimal PPOAgent for testing PolicyGroup."""
    return PPOAgent(state_dim=30, action_dim=7)


class TestPolicyGroupInitialization:
    """Test PolicyGroup construction and default values."""

    def test_creates_with_required_fields(self, mock_agent):
        """PolicyGroup should initialize with required fields."""
        device = torch.device("cpu")
        reward_config = ContributionRewardConfig()

        group = PolicyGroup(
            group_id="A",
            device=device,
            reward_mode="SHAPED",
            agent=mock_agent,
            reward_config=reward_config,
        )

        assert group.group_id == "A"
        assert group.device == device
        assert group.reward_mode == "SHAPED"
        assert group.agent is mock_agent
        assert group.reward_config is reward_config

    def test_default_metrics_start_at_zero(self, mock_agent):
        """Default metrics should be initialized to zero."""
        group = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            reward_mode="SIMPLIFIED",
            agent=mock_agent,
            reward_config=ContributionRewardConfig(),
        )

        assert group.total_episodes == 0
        assert group.total_steps == 0
        assert group.best_accuracy == 0.0

    def test_episode_history_defaults_to_empty_list(self, mock_agent):
        """Episode history should default to empty list."""
        group = PolicyGroup(
            group_id="C",
            device=torch.device("cpu"),
            reward_mode="SPARSE",
            agent=mock_agent,
            reward_config=ContributionRewardConfig(),
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
            reward_mode="SHAPED",
            agent=mock_agent,
            reward_config=ContributionRewardConfig(),
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
            reward_mode="SHAPED",
            agent=mock_agent,
            reward_config=ContributionRewardConfig(),
        )

        # Simulate recording three episodes
        group.episode_history.append({"episode": 1, "final_accuracy": 45.0, "episode_reward": 10.5})
        group.episode_history.append({"episode": 2, "final_accuracy": 52.3, "episode_reward": 12.1})
        group.episode_history.append({"episode": 3, "final_accuracy": 58.9, "episode_reward": 15.7})

        assert len(group.episode_history) == 3
        assert group.episode_history[0]["final_accuracy"] == 45.0
        assert group.episode_history[-1]["episode"] == 3

    def test_metrics_can_be_updated(self, mock_agent):
        """Metrics like total_episodes and best_accuracy should be mutable."""
        group = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            reward_mode="SIMPLIFIED",
            agent=mock_agent,
            reward_config=ContributionRewardConfig(),
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
        agent_a = PPOAgent(state_dim=30, action_dim=7)
        agent_b = PPOAgent(state_dim=30, action_dim=7)

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            reward_mode="SHAPED",
            agent=agent_a,
            reward_config=ContributionRewardConfig(),
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            reward_mode="SIMPLIFIED",
            agent=agent_b,
            reward_config=ContributionRewardConfig(),
        )

        # Add to group A only
        group_a.episode_history.append({"episode": 1, "final_accuracy": 50.0})

        # Group B should remain empty
        assert len(group_a.episode_history) == 1
        assert len(group_b.episode_history) == 0

    def test_groups_can_have_different_devices(self):
        """Groups should be able to use different devices."""
        agent_a = PPOAgent(state_dim=30, action_dim=7)
        agent_b = PPOAgent(state_dim=30, action_dim=7)

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            reward_mode="SHAPED",
            agent=agent_a,
            reward_config=ContributionRewardConfig(),
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),  # In real usage, this would be cuda:1
            reward_mode="SIMPLIFIED",
            agent=agent_b,
            reward_config=ContributionRewardConfig(),
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
            contribution_weight=1.0,
            pbrs_weight=0.1,
        )

        config_b = ContributionRewardConfig(
            contribution_weight=0.5,
            pbrs_weight=0.2,
        )

        agent_a = PPOAgent(state_dim=30, action_dim=7)
        agent_b = PPOAgent(state_dim=30, action_dim=7)

        group_a = PolicyGroup(
            group_id="A",
            device=torch.device("cpu"),
            reward_mode="SHAPED",
            agent=agent_a,
            reward_config=config_a,
        )

        group_b = PolicyGroup(
            group_id="B",
            device=torch.device("cpu"),
            reward_mode="SIMPLIFIED",
            agent=agent_b,
            reward_config=config_b,
        )

        # Verify independent configs
        assert group_a.reward_config.contribution_weight == 1.0
        assert group_b.reward_config.contribution_weight == 0.5
        assert group_a.reward_config is not group_b.reward_config
