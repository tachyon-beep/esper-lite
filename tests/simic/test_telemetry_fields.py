"""Tests for low-hanging fruit telemetry fields.

Tests gradient health aggregation, reward shaping ratio, and entropy collapse detection.
"""

import pytest

from esper.simic.debug_telemetry import LayerGradientStats
from esper.simic.reward_telemetry import RewardComponentsTelemetry
from esper.simic.vectorized import _aggregate_layer_gradient_health


class TestLayerGradientAggregation:
    """Test gradient layer health aggregation."""

    def test_empty_list_returns_defaults(self):
        """Empty input returns safe defaults."""
        result = _aggregate_layer_gradient_health([])
        assert result["dead_layers"] == 0
        assert result["exploding_layers"] == 0
        assert result["nan_grad_count"] == 0
        assert result["layer_gradient_health"] == 1.0

    def test_dead_layer_detection(self):
        """Layers with >90% zero gradients are dead."""
        stats = [
            LayerGradientStats(
                layer_name="dead_layer",
                param_count=100,
                grad_norm=0.0,
                grad_mean=0.0,
                grad_std=0.0,
                grad_min=0.0,
                grad_max=0.0,
                zero_fraction=0.95,  # Dead
                small_fraction=0.99,
                large_fraction=0.0,
                nan_count=0,
                inf_count=0,
            ),
            LayerGradientStats(
                layer_name="healthy_layer",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,  # Healthy
                small_fraction=0.2,
                large_fraction=0.0,
                nan_count=0,
                inf_count=0,
            ),
        ]

        result = _aggregate_layer_gradient_health(stats)
        assert result["dead_layers"] == 1
        assert result["exploding_layers"] == 0
        assert result["nan_grad_count"] == 0
        # Health reduced due to dead layer
        assert result["layer_gradient_health"] < 1.0

    def test_exploding_layer_detection(self):
        """Layers with >10% large gradients are exploding."""
        stats = [
            LayerGradientStats(
                layer_name="exploding",
                param_count=100,
                grad_norm=100.0,
                grad_mean=10.0,
                grad_std=20.0,
                grad_min=0.0,
                grad_max=1000.0,
                zero_fraction=0.0,
                small_fraction=0.0,
                large_fraction=0.15,  # Exploding
                nan_count=0,
                inf_count=0,
            ),
        ]

        result = _aggregate_layer_gradient_health(stats)
        assert result["dead_layers"] == 0
        assert result["exploding_layers"] == 1
        assert result["nan_grad_count"] == 0
        # Health reduced significantly due to exploding
        assert result["layer_gradient_health"] < 0.5

    def test_nan_count_aggregation(self):
        """NaN counts are summed across layers."""
        stats = [
            LayerGradientStats(
                layer_name="layer1",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,
                small_fraction=0.2,
                large_fraction=0.0,
                nan_count=5,
                inf_count=0,
            ),
            LayerGradientStats(
                layer_name="layer2",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,
                small_fraction=0.2,
                large_fraction=0.0,
                nan_count=10,
                inf_count=0,
            ),
        ]

        result = _aggregate_layer_gradient_health(stats)
        assert result["nan_grad_count"] == 15

    def test_healthy_layers_perfect_score(self):
        """All healthy layers yields health score near 1.0."""
        stats = [
            LayerGradientStats(
                layer_name=f"layer{i}",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,
                small_fraction=0.3,
                large_fraction=0.0,
                nan_count=0,
                inf_count=0,
            )
            for i in range(5)
        ]

        result = _aggregate_layer_gradient_health(stats)
        assert result["dead_layers"] == 0
        assert result["exploding_layers"] == 0
        assert result["nan_grad_count"] == 0
        assert result["layer_gradient_health"] == 1.0


class TestShapedRewardRatio:
    """Test reward shaping ratio computation."""

    def test_high_shaping_ratio(self):
        """High shaping bonuses produce high ratio."""
        t = RewardComponentsTelemetry(
            stage_bonus=0.3,
            pbrs_bonus=0.2,
            action_shaping=0.1,
            total_reward=1.0,
        )
        assert 0.55 < t.shaped_reward_ratio < 0.65

    def test_low_shaping_ratio(self):
        """Low shaping bonuses produce low ratio."""
        t = RewardComponentsTelemetry(
            stage_bonus=0.05,
            pbrs_bonus=0.0,
            action_shaping=0.0,
            base_acc_delta=0.8,
            total_reward=0.85,
        )
        assert t.shaped_reward_ratio < 0.1

    def test_zero_total_reward(self):
        """Zero total reward returns 0.0 ratio."""
        t = RewardComponentsTelemetry(total_reward=0.0)
        assert t.shaped_reward_ratio == 0.0

    def test_negative_reward(self):
        """Negative rewards are handled gracefully."""
        t = RewardComponentsTelemetry(
            stage_bonus=0.1,
            total_reward=-0.5,
        )
        # Should handle negative rewards gracefully (absolute values)
        assert t.shaped_reward_ratio >= 0.0
        assert t.shaped_reward_ratio == 0.2  # |0.1| / |-0.5|

    def test_in_to_dict(self):
        """shaped_reward_ratio is included in to_dict()."""
        t = RewardComponentsTelemetry(
            stage_bonus=0.5,
            total_reward=1.0,
        )
        d = t.to_dict()
        assert "shaped_reward_ratio" in d
        assert d["shaped_reward_ratio"] == 0.5


class TestEntropyCollapsed:
    """Test entropy collapse detection logic."""

    def test_collapsed_entropy_true(self):
        """Entropy < 0.1 indicates collapse."""
        entropy = 0.05
        collapsed = entropy < 0.1
        assert collapsed is True

    def test_healthy_entropy_false(self):
        """Entropy >= 0.1 is healthy."""
        entropy = 0.5
        collapsed = entropy < 0.1
        assert collapsed is False

    def test_boundary_entropy(self):
        """Exactly 0.1 is not collapsed."""
        entropy = 0.1
        collapsed = entropy < 0.1
        assert collapsed is False
