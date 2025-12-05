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
