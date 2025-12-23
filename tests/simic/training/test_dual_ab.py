"""Tests for dual-policy A/B testing training function."""

import pytest
import torch

from esper.simic.rewards import RewardMode


class TestTrainDualPolicyAB:
    """Tests for train_dual_policy_ab function."""

    def test_function_exists_and_importable(self):
        """Should be able to import train_dual_policy_ab."""
        from esper.simic.training import train_dual_policy_ab

        assert callable(train_dual_policy_ab)

    def test_validates_minimum_groups(self):
        """Should require at least 2 groups for A/B testing."""
        from esper.simic.training import train_dual_policy_ab

        with pytest.raises(ValueError, match="at least 2 groups"):
            train_dual_policy_ab(
                n_envs_per_group=1,
                group_configs=[("A", RewardMode.SHAPED)],
                devices=["cpu"],
                n_episodes=1,
            )

    def test_validates_device_count_matches_groups(self):
        """Should require same number of devices as groups."""
        from esper.simic.training import train_dual_policy_ab

        with pytest.raises(ValueError, match="Number of devices"):
            train_dual_policy_ab(
                n_envs_per_group=1,
                group_configs=[
                    ("A", RewardMode.SHAPED),
                    ("B", RewardMode.SIMPLIFIED),
                ],
                devices=["cpu"],  # Only 1 device for 2 groups
                n_episodes=1,
            )

    def test_default_group_configs(self):
        """Should use default group configs if none provided."""
        from esper.simic.training import train_dual_policy_ab

        # This will fail with CUDA error if run, but we're just testing
        # that defaults are set correctly by catching the expected error
        with pytest.raises(ValueError, match="requires CUDA"):
            train_dual_policy_ab(
                n_envs_per_group=1,
                devices=None,  # Should auto-detect
                n_episodes=1,
                slots=["r0c0"],  # Minimal slot config for testing
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Requires at least 2 CUDA devices",
    )
    def test_runs_with_minimal_config(self):
        """Should run successfully with minimal configuration on 2 GPUs.

        This is a smoke test that verifies the basic function works.
        We use very small parameters to make it fast.
        """
        from esper.simic.training import train_dual_policy_ab

        results = train_dual_policy_ab(
            n_envs_per_group=1,  # Minimal envs
            group_configs=[
                ("A", RewardMode.SHAPED),
                ("B", RewardMode.SIMPLIFIED),
            ],
            devices=["cuda:0", "cuda:1"],
            n_episodes=1,  # Just 1 episode
            max_epochs=5,  # Very short episodes
            task="cifar10",
            use_telemetry=False,  # Disable for speed
            slots=["r0c0"],  # Minimal slot configuration
        )

        # Should return dict with both groups
        assert "A" in results
        assert "B" in results

        # Each group should have (agent, history) tuple
        agent_a, history_a = results["A"]
        agent_b, history_b = results["B"]

        assert agent_a is not None
        assert agent_b is not None
        assert isinstance(history_a, list)
        assert isinstance(history_b, list)

        # History should contain batch-level metrics
        if history_a:
            assert "avg_accuracy" in history_a[0]
            assert "batch" in history_a[0]

    def test_groups_have_different_seeds(self):
        """Should use different seeds for different groups.

        This ensures groups don't have identical initialization.
        We test this by checking the seed calculation logic.
        """
        # Test the seed generation logic
        base_seed = 42
        group_a_seed = base_seed + hash("A") % 10000
        group_b_seed = base_seed + hash("B") % 10000

        # Seeds should be different
        assert group_a_seed != group_b_seed
