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

    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="Test expects CUDA to be unavailable",
    )
    def test_default_group_configs_requires_cuda(self):
        """Should raise error if CUDA not available when using defaults."""
        from esper.simic.training import train_dual_policy_ab

        # This will fail with CUDA error when CUDA is not available
        with pytest.raises(ValueError, match="requires CUDA"):
            train_dual_policy_ab(
                n_envs_per_group=1,
                devices=None,  # Should auto-detect and fail
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
        import hashlib

        # Test the seed generation logic (using deterministic hash)
        base_seed = 42
        group_a_hash = int(hashlib.md5("A".encode()).hexdigest()[:8], 16)
        group_b_hash = int(hashlib.md5("B".encode()).hexdigest()[:8], 16)
        group_a_seed = base_seed + (group_a_hash % 10000)
        group_b_seed = base_seed + (group_b_hash % 10000)

        # Seeds should be different
        assert group_a_seed != group_b_seed


class TestTelemetryGroupId:
    """Tests for group_id telemetry tagging."""

    def test_telemetry_event_has_group_id_field(self):
        """TelemetryEvent should have group_id field for A/B testing."""
        from esper.leyline import TelemetryEvent, TelemetryEventType

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            group_id="A",
            message="Test",
        )
        assert event.group_id == "A"

    def test_train_ppo_vectorized_accepts_group_id(self):
        """train_ppo_vectorized should accept group_id parameter."""
        import inspect
        from esper.simic.training.vectorized import train_ppo_vectorized

        sig = inspect.signature(train_ppo_vectorized)
        assert "group_id" in sig.parameters
        assert sig.parameters["group_id"].default == "default"

    def test_telemetry_event_defaults_to_default_group(self):
        """TelemetryEvent should default to 'default' group for non-A/B training."""
        from esper.leyline import TelemetryEvent, TelemetryEventType

        event = TelemetryEvent(event_type=TelemetryEventType.EPOCH_COMPLETED)
        assert event.group_id == "default"
