"""Integration tests for telemetry collection pipeline.

Tests end-to-end gradient collection, snapshot generation, and feature extraction.
"""

import torch
from esper.simic.gradient_collector import SeedGradientCollector
from esper.leyline import SeedTelemetry


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

    def test_telemetry_to_features_10_dim(self):
        """SeedTelemetry.to_features() must return 10-dim vector."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.0,
            gradient_health=0.9,
            accuracy=75.0,
        )

        features = telemetry.to_features()

        assert len(features) == 10

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
        # Stage 4 (out of 1-7) should map to 0.5
        telemetry = SeedTelemetry(
            seed_id="test",
            stage=4,
        )

        features = telemetry.to_features()
        stage_feature = features[7]

        # (4 - 1) / 6 = 3 / 6 = 0.5
        assert abs(stage_feature - 0.5) < 1e-5


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
        assert len(features) == 10
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
