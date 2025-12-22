"""Integration tests for telemetry collection pipeline.

Tests end-to-end gradient collection, snapshot generation, and feature extraction.
"""

import pytest
import torch

from esper.simic.telemetry import SeedGradientCollector
from esper.leyline import SeedTelemetry
from esper.leyline.alpha import AlphaAlgorithm, AlphaMode


class TestGradientCollectionPipeline:
    """Test gradient collection in training loop."""

    def test_gradient_collection_after_backward(self):
        """Gradients should be collected after loss.backward()."""
        # Simple model
        model = torch.nn.Linear(10, 2)

        # Forward + backward
        x = torch.randn(32, 10)
        y_pred = model(x)
        loss = y_pred.mean()
        loss.backward()

        # Collect gradients
        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        assert 'gradient_norm' in stats
        assert stats['gradient_norm'] > 0  # Non-zero after backward

    def test_gradient_collection_accuracy(self):
        """Gradients should match analytical expectation (exact math).

        This is a STRONG assertion that validates the gradient computation
        is mathematically correct, not just that the pipe is open.
        """
        # Create simple linear model: y = w*x (no bias)
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)  # w = 2

        # Forward pass: y = 2 * 3 = 6
        x = torch.tensor([[3.0]])
        y_pred = model(x)  # y_pred = 6.0

        # Loss = y_pred (identity loss for simple test)
        loss = y_pred.sum()  # loss = 6.0

        # Backward: d(loss)/dw = x = 3.0
        loss.backward()

        # Collect gradients
        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        # The gradient should be exactly 3.0
        # (since d(loss)/dw = d(w*x)/dw = x = 3.0)
        expected_grad_norm = 3.0
        assert abs(stats['gradient_norm'] - expected_grad_norm) < 1e-5, \
            f"Expected gradient norm {expected_grad_norm}, got {stats['gradient_norm']}"

    def test_gradient_collection_no_gradients(self):
        """Collector should handle parameters with no gradients."""
        # Create model but don't run backward
        model = torch.nn.Linear(10, 2)

        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        # Should return zero stats when no gradients
        assert stats['gradient_norm'] == 0.0
        assert stats['gradient_health'] == 1.0
        assert stats['has_vanishing'] is False
        assert stats['has_exploding'] is False

    def test_gradient_collection_vanishing_detection(self):
        """Collector should detect vanishing gradients."""
        # Create parameter with tiny gradient
        param = torch.nn.Parameter(torch.ones(10, 10))
        param.grad = torch.full((10, 10), 1e-8)  # Below default threshold

        collector = SeedGradientCollector()
        stats = collector.collect([param])

        assert stats['has_vanishing'] is True
        assert stats['gradient_health'] < 1.0

    def test_gradient_collection_exploding_detection(self):
        """Collector should detect exploding gradients."""
        # Create parameter with large gradient
        param = torch.nn.Parameter(torch.ones(10, 10))
        param.grad = torch.full((10, 10), 200.0)  # Above default threshold

        collector = SeedGradientCollector()
        stats = collector.collect([param])

        assert stats['has_exploding'] is True
        assert stats['gradient_health'] < 1.0

    def test_gradient_collection_health_scoring(self):
        """Health score should reflect gradient quality."""
        # Healthy gradients
        param_healthy = torch.nn.Parameter(torch.ones(10, 10))
        param_healthy.grad = torch.full((10, 10), 0.01)  # Normal gradient

        collector = SeedGradientCollector()
        healthy_stats = collector.collect([param_healthy])

        # Unhealthy gradients (mix of vanishing and exploding)
        param_unhealthy = torch.nn.Parameter(torch.ones(10, 10))
        param_unhealthy.grad = torch.cat([
            torch.full((5, 10), 1e-8),  # Vanishing
            torch.full((5, 10), 200.0),  # Exploding
        ])

        unhealthy_stats = collector.collect([param_unhealthy])

        # Healthy should have higher health score
        assert healthy_stats['gradient_health'] > unhealthy_stats['gradient_health']


class TestSeedTelemetryFeatures:
    """Test SeedTelemetry feature conversion."""

    def test_telemetry_to_features_17_dim(self):
        """SeedTelemetry.to_features() must return 17-dim vector."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.0,
            gradient_health=0.9,
            accuracy=75.0,
        )

        features = telemetry.to_features()

        assert len(features) == 17

    def test_telemetry_feature_dim_matches(self):
        """SeedTelemetry.feature_dim() should match actual dimension."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.0,
        )

        features = telemetry.to_features()
        expected_dim = SeedTelemetry.feature_dim()

        assert len(features) == expected_dim

    def test_telemetry_features_bounded(self):
        """Telemetry features should be roughly normalized."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=5.0,
            gradient_health=0.8,
            accuracy=65.0,
            accuracy_delta=1.5,
        )

        features = telemetry.to_features()

        # All features should be roughly in [0, 1] or [-1, 1]
        for i, feat in enumerate(features):
            assert -5.0 < feat < 5.0, f"Feature {i} = {feat} out of range"

    def test_telemetry_features_gradient_norm_normalized(self):
        """Gradient norm should be normalized to [0, 1] range."""
        # Gradient norm of 5.0 should map to 0.5
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=5.0,
        )

        features = telemetry.to_features()
        gradient_norm_feature = features[0]

        assert abs(gradient_norm_feature - 0.5) < 1e-5

    def test_telemetry_features_gradient_health_direct(self):
        """Gradient health should pass through directly."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_health=0.75,
        )

        features = telemetry.to_features()
        gradient_health_feature = features[1]

        assert abs(gradient_health_feature - 0.75) < 1e-5

    def test_telemetry_features_accuracy_normalized(self):
        """Accuracy should be normalized to [0, 1] range."""
        # Accuracy of 60% should map to 0.6
        telemetry = SeedTelemetry(
            seed_id="test",
            accuracy=60.0,
        )

        features = telemetry.to_features()
        accuracy_feature = features[5]

        assert abs(accuracy_feature - 0.6) < 1e-5

    def test_telemetry_features_stage_normalized(self):
        """Stage should be normalized to [0, 1] range."""
        # Stage 4 (out of 1-10) should map to 0.333...
        telemetry = SeedTelemetry(
            seed_id="test",
            stage=4,
        )

        features = telemetry.to_features()
        stage_feature = features[7]

        # (4 - 1) / 9 = 3 / 9 = 0.333...
        assert abs(stage_feature - (3.0 / 9.0)) < 1e-5

    def test_telemetry_features_alpha_controller_normalized(self):
        """Alpha controller fields should be normalized and stable."""
        telemetry = SeedTelemetry(
            seed_id="test",
            alpha_target=0.7,
            alpha_mode=AlphaMode.DOWN.value,
            alpha_steps_total=10,
            alpha_steps_done=4,
            time_to_target=6,
            alpha_velocity=-0.1,
            alpha_algorithm=AlphaAlgorithm.GATE.value,
            max_epochs=20,
        )

        features = telemetry.to_features()

        assert abs(features[10] - 0.7) < 1e-6
        assert abs(features[11] - 1.0) < 1e-6  # DOWN -> 2 / 2
        assert abs(features[12] - 0.5) < 1e-6  # 10 / 20
        assert abs(features[13] - 0.2) < 1e-6  # 4 / 20
        assert abs(features[14] - 0.3) < 1e-6  # 6 / 20
        assert abs(features[15] + 0.1) < 1e-6
        assert abs(features[16] - 1.0) < 1e-6  # (3-1)/(3-1)

    def test_telemetry_roundtrip_preserves_alpha_fields(self):
        """to_dict() -> from_dict() preserves alpha controller fields."""
        telemetry = SeedTelemetry(
            seed_id="test",
            alpha_target=0.4,
            alpha_mode=AlphaMode.UP.value,
            alpha_steps_total=12,
            alpha_steps_done=9,
            time_to_target=3,
            alpha_velocity=0.05,
            alpha_algorithm=AlphaAlgorithm.MULTIPLY.value,
        )

        restored = SeedTelemetry.from_dict(telemetry.to_dict())

        assert restored.alpha_target == telemetry.alpha_target
        assert restored.alpha_mode == telemetry.alpha_mode
        assert restored.alpha_steps_total == telemetry.alpha_steps_total
        assert restored.alpha_steps_done == telemetry.alpha_steps_done
        assert restored.time_to_target == telemetry.time_to_target
        assert restored.alpha_velocity == telemetry.alpha_velocity
        assert restored.alpha_algorithm == telemetry.alpha_algorithm

    def test_telemetry_from_dict_rejects_reserved_stage_value(self):
        """from_dict() must reject retired/reserved stage values (e.g., value 5)."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["stage"] = 5  # reserved gap in SeedStage

        with pytest.raises(ValueError, match="SeedTelemetry\\.stage"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_out_of_range_stage_value(self):
        """from_dict() must reject out-of-range stage values."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["stage"] = 999

        with pytest.raises(ValueError, match="SeedTelemetry\\.stage"):
            SeedTelemetry.from_dict(data)


class TestTelemetryToFeaturesIntegration:
    """Test end-to-end flow from gradients to features."""

    def test_end_to_end_gradient_to_features(self):
        """Test complete pipeline: model -> gradients -> telemetry -> features."""
        # Step 1: Create simple model and compute gradients
        model = torch.nn.Linear(10, 2)
        x = torch.randn(32, 10)
        y_pred = model(x)
        loss = y_pred.mean()
        loss.backward()

        # Step 2: Collect gradient stats
        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        # Step 3: Create telemetry from stats
        telemetry = SeedTelemetry(
            seed_id="test-seed",
            gradient_norm=stats['gradient_norm'],
            gradient_health=stats['gradient_health'],
            has_vanishing=stats['has_vanishing'],
            has_exploding=stats['has_exploding'],
            accuracy=75.0,
            epoch=10,
            max_epochs=25,
        )

        # Step 4: Convert to features
        features = telemetry.to_features()

        # Verify complete pipeline
        assert len(features) == 17
        assert all(isinstance(f, float) for f in features)
        assert all(-5.0 < f < 5.0 for f in features)

    def test_telemetry_features_deterministic(self):
        """Same telemetry should produce same features."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.5,
            gradient_health=0.85,
            accuracy=70.0,
            stage=3,
        )

        features1 = telemetry.to_features()
        features2 = telemetry.to_features()

        # Should be identical
        assert features1 == features2

    def test_telemetry_features_different_inputs(self):
        """Different telemetry should produce different features."""
        telemetry1 = SeedTelemetry(
            seed_id="test1",
            gradient_norm=1.0,
            accuracy=70.0,
        )

        telemetry2 = SeedTelemetry(
            seed_id="test2",
            gradient_norm=5.0,
            accuracy=80.0,
        )

        features1 = telemetry1.to_features()
        features2 = telemetry2.to_features()

        # Should be different
        assert features1 != features2

    def test_telemetry_consistent_with_dimension_constant(self):
        """Feature vector should match SeedTelemetry.feature_dim()."""
        # Create multiple telemetry instances with different values
        telemetries = [
            SeedTelemetry(seed_id="test1", gradient_norm=1.0, accuracy=50.0),
            SeedTelemetry(seed_id="test2", gradient_norm=2.0, accuracy=60.0),
            SeedTelemetry(seed_id="test3", gradient_norm=3.0, accuracy=70.0),
        ]

        expected_dim = SeedTelemetry.feature_dim()

        for telemetry in telemetries:
            features = telemetry.to_features()
            assert len(features) == expected_dim, \
                f"Expected {expected_dim} features, got {len(features)}"
