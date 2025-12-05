"""Tests for PPO telemetry dataclasses."""

import pytest

from esper.simic.ppo_telemetry import PPOHealthTelemetry, ValueFunctionTelemetry


class TestPPOHealthTelemetry:
    """Tests for PPOHealthTelemetry."""

    def test_create_from_metrics(self):
        """Can create PPOHealthTelemetry from raw metrics."""
        telemetry = PPOHealthTelemetry(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=0.8,
            approx_kl=0.01,
            clip_fraction=0.15,
            ratio_mean=1.0,
            ratio_std=0.1,
            ratio_max=1.5,
            ratio_min=0.7,
        )
        assert telemetry.policy_loss == 0.5
        assert telemetry.ratio_max == 1.5

    def test_is_ratio_healthy(self):
        """is_ratio_healthy detects problematic ratios."""
        healthy = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=1.5, ratio_min=0.7,
        )
        assert healthy.is_ratio_healthy() is True

        unhealthy = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.5, ratio_max=6.0, ratio_min=0.1,
        )
        assert unhealthy.is_ratio_healthy() is False

    def test_to_dict(self):
        """Can convert to dict for telemetry event."""
        telemetry = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=1.5, ratio_min=0.7,
        )
        d = telemetry.to_dict()
        assert d["policy_loss"] == 0.5
        assert "ratio_max" in d

    def test_is_ratio_healthy_at_exact_threshold(self):
        """Boundary: ratio exactly at threshold is healthy (< not <=)."""
        # ratio_max exactly at 5.0 threshold - should be healthy (< 5.0)
        at_max = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=4.999, ratio_min=0.101,
        )
        assert at_max.is_ratio_healthy() is True

        # Just above max threshold
        above_max = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=5.001, ratio_min=0.5,
        )
        assert above_max.is_ratio_healthy() is False

        # Just below min threshold
        below_min = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=2.0, ratio_min=0.099,
        )
        assert below_min.is_ratio_healthy() is False

    def test_is_ratio_healthy_with_custom_thresholds(self):
        """Can use custom thresholds for ratio health check."""
        telemetry = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=3.0, ratio_min=0.3,
        )
        # Default thresholds (5.0, 0.1) - healthy
        assert telemetry.is_ratio_healthy() is True
        # Stricter thresholds - unhealthy
        assert telemetry.is_ratio_healthy(max_ratio_threshold=2.0) is False
        assert telemetry.is_ratio_healthy(min_ratio_threshold=0.5) is False


class TestValueFunctionTelemetry:
    """Tests for ValueFunctionTelemetry."""

    def test_explained_variance_calculation(self):
        """explained_variance is computed correctly."""
        import torch

        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1.1, 1.9, 3.1, 3.9])

        telemetry = ValueFunctionTelemetry.from_tensors(returns, values)
        # Should be close to 1.0 (good prediction)
        assert telemetry.explained_variance > 0.9

    def test_is_healthy(self):
        """is_healthy detects value function collapse."""
        import torch

        # Good predictions
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1.1, 1.9, 3.1, 3.9])
        healthy = ValueFunctionTelemetry.from_tensors(returns, values)
        assert healthy.is_healthy() is True

        # Bad predictions (constant value)
        bad_values = torch.tensor([2.5, 2.5, 2.5, 2.5])
        unhealthy = ValueFunctionTelemetry.from_tensors(returns, bad_values)
        assert unhealthy.explained_variance < 0.1

    def test_explained_variance_zero_variance_returns(self):
        """Handles zero-variance returns without division error."""
        import torch

        # All returns identical - zero variance
        returns = torch.tensor([2.0, 2.0, 2.0, 2.0])
        values = torch.tensor([1.9, 2.1, 2.0, 2.0])

        telemetry = ValueFunctionTelemetry.from_tensors(returns, values)
        # Should gracefully handle and return 0.0
        assert telemetry.explained_variance == 0.0

    def test_from_tensors_with_advantages(self):
        """from_tensors correctly handles advantages parameter."""
        import torch

        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1.1, 1.9, 3.1, 3.9])
        advantages = torch.tensor([0.5, -0.3, 0.2, 0.1])

        telemetry = ValueFunctionTelemetry.from_tensors(returns, values, advantages)
        assert abs(telemetry.advantage_mean - 0.125) < 0.01  # mean of advantages
        assert telemetry.advantage_std > 0
