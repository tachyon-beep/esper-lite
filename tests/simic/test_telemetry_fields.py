"""Tests for low-hanging fruit telemetry fields.

Tests gradient health aggregation, reward shaping ratio, and entropy collapse detection.
"""


import dataclasses

from esper.simic.telemetry import LayerGradientStats
from esper.leyline.telemetry import PPOUpdatePayload
from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.telemetry.emitters import aggregate_layer_gradient_health


class TestPPOUpdatePayloadEVRobustnessFields:
    """EV-telemetry-robustness additive contract on PPOUpdatePayload (leyline)."""

    def test_ppo_update_payload_carries_ev_robustness_fields(self):
        """Payload exposes the four new fields, and pre-existing robust signals remain reachable."""
        fields = {f.name: f for f in dataclasses.fields(PPOUpdatePayload)}

        # New additive fields
        assert "value_nrmse" in fields
        assert "ev_low_return_variance" in fields
        assert "ev_return_variance" in fields
        assert "ev_low_return_variance_count" in fields

        # Rollback observability counters (additive int fields, default 0)
        assert "rollback_count" in fields
        assert "rollback_steps_zeroed" in fields
        assert "rollback_attempt_count" in fields
        assert "rollback_unattributed_count" in fields

        # Pre-existing fields the gate already reads (NOT new)
        assert "bellman_error" in fields
        assert "value_loss" in fields
        assert "v_return_correlation" in fields
        assert "return_variance" in fields

        # Defaults match the additive/schema-evolution contract
        payload = PPOUpdatePayload(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        )
        assert payload.value_nrmse is None
        assert payload.ev_low_return_variance is False
        assert payload.ev_return_variance is None
        assert payload.ev_low_return_variance_count == 0
        assert payload.rollback_count == 0
        assert payload.rollback_steps_zeroed == 0
        assert payload.rollback_attempt_count == 0
        assert payload.rollback_unattributed_count == 0

    def test_ppo_update_payload_from_dict_old_event_without_ev_fields(self):
        """B4 schema-evolution: an old persisted event (no EV-robustness keys) deserializes with defaults."""
        # Build a complete current payload, serialize, then strip the new keys to
        # simulate an event persisted BEFORE this plan landed.
        full = PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.2,
            entropy=0.3,
            grad_norm=1.0,
            kl_divergence=0.01,
            clip_fraction=0.1,
            nan_grad_count=0,
        )
        data = dataclasses.asdict(full)
        for key in (
            "value_nrmse",
            "ev_low_return_variance",
            "ev_return_variance",
            "ev_low_return_variance_count",
        ):
            data.pop(key, None)

        restored = PPOUpdatePayload.from_dict(data)

        assert restored.value_nrmse is None
        assert restored.ev_low_return_variance is False
        assert restored.ev_return_variance is None
        assert restored.ev_low_return_variance_count == 0


class TestLayerGradientAggregation:
    """Test gradient layer health aggregation."""

    def test_empty_list_returns_defaults(self):
        """Empty input returns safe defaults."""
        result = aggregate_layer_gradient_health([])
        assert result["dead_layers"] == 0
        assert result["exploding_layers"] == 0
        assert result["nan_grad_count"] == 0
        # Empty input returns empty per-layer dict
        assert result["layer_gradient_health"] == {}

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

        result = aggregate_layer_gradient_health(stats)
        assert result["dead_layers"] == 1
        assert result["exploding_layers"] == 0
        assert result["nan_grad_count"] == 0
        # Per-layer health: dead layer gets 0.0, healthy layer gets 1.0
        per_layer = result["layer_gradient_health"]
        assert per_layer["dead_layer"] == 0.0
        assert per_layer["healthy_layer"] == 1.0

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

        result = aggregate_layer_gradient_health(stats)
        assert result["dead_layers"] == 0
        assert result["exploding_layers"] == 1
        assert result["nan_grad_count"] == 0
        # Per-layer health: exploding layer gets low score (0.1)
        per_layer = result["layer_gradient_health"]
        assert per_layer["exploding"] < 0.5

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

        result = aggregate_layer_gradient_health(stats)
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

        result = aggregate_layer_gradient_health(stats)
        assert result["dead_layers"] == 0
        assert result["exploding_layers"] == 0
        assert result["nan_grad_count"] == 0
        # Per-layer health: all healthy layers get 1.0
        per_layer = result["layer_gradient_health"]
        assert all(v == 1.0 for v in per_layer.values())
        assert len(per_layer) == 5


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
