"""Tests for SeedTelemetry feature conversion and validation."""

from __future__ import annotations

import pytest
import torch

from esper.leyline import SeedTelemetry
from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.simic.telemetry import SeedGradientCollector


class TestSeedTelemetryFeatures:
    """SeedTelemetry.to_features() schema and normalization tests."""

    def test_telemetry_to_features_26_dim(self):
        """SeedTelemetry.to_features() returns 26-dim vector (schema v1, one-hot stage)."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.0,
            gradient_health=0.9,
            accuracy=75.0,
        )

        features = telemetry.to_features()

        # Schema v1: 10 stage one-hot + 16 other features = 26
        assert len(features) == 26

    def test_telemetry_feature_dim_matches(self):
        """SeedTelemetry.feature_dim() matches actual dimension."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.0,
        )

        features = telemetry.to_features()
        expected_dim = SeedTelemetry.feature_dim()

        assert len(features) == expected_dim

    def test_telemetry_features_bounded(self):
        """Telemetry features are roughly normalized (sanity bounds)."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=5.0,
            gradient_health=0.8,
            accuracy=65.0,
            accuracy_delta=1.5,
        )

        features = telemetry.to_features()

        for i, feat in enumerate(features):
            assert -5.0 < feat < 5.0, f"Feature {i} = {feat} out of range"

    def test_telemetry_features_gradient_norm_normalized(self):
        """Gradient norm is normalized to [0, 1] range."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=5.0,
        )

        features = telemetry.to_features()
        # Feature layout: [0-9] stage one-hot, [10] gradient_norm
        gradient_norm_feature = features[10]

        assert abs(gradient_norm_feature - 0.5) < 1e-5

    def test_telemetry_features_gradient_health_direct(self):
        """Gradient health passes through directly."""
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_health=0.75,
        )

        features = telemetry.to_features()
        # Feature layout: [0-9] stage one-hot, [11] gradient_health
        gradient_health_feature = features[11]

        assert abs(gradient_health_feature - 0.75) < 1e-5

    def test_telemetry_features_accuracy_normalized(self):
        """Accuracy is normalized to [0, 1] range."""
        telemetry = SeedTelemetry(
            seed_id="test",
            accuracy=60.0,
        )

        features = telemetry.to_features()
        # Feature layout: [0-9] stage one-hot, [15] accuracy
        accuracy_feature = features[15]

        assert abs(accuracy_feature - 0.6) < 1e-5

    def test_telemetry_features_stage_one_hot(self):
        """Stage is one-hot encoded in first NUM_STAGES dims."""
        from esper.leyline.stage_schema import STAGE_TO_INDEX, NUM_STAGES

        telemetry = SeedTelemetry(
            seed_id="test",
            stage=4,  # BLENDING
        )

        features = telemetry.to_features()
        stage_one_hot = features[:NUM_STAGES]

        assert sum(stage_one_hot) == 1.0
        expected_idx = STAGE_TO_INDEX[4]
        assert stage_one_hot[expected_idx] == 1.0

    def test_telemetry_features_alpha_controller_normalized(self):
        """Alpha controller fields are normalized and stable."""
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

        assert abs(features[19] - 0.7) < 1e-6
        assert abs(features[20] - 1.0) < 1e-6
        assert abs(features[21] - 0.5) < 1e-6
        assert abs(features[22] - 0.2) < 1e-6
        assert abs(features[23] - 0.3) < 1e-6
        assert abs(features[24] + 0.1) < 1e-6
        assert abs(features[25] - 1.0) < 1e-6

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
        """from_dict() rejects retired/reserved stage values (e.g., value 5)."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["stage"] = 5  # reserved gap in SeedStage

        with pytest.raises(ValueError, match="SeedTelemetry\\.stage"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_out_of_range_stage_value(self):
        """from_dict() rejects out-of-range stage values."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["stage"] = 999

        with pytest.raises(ValueError, match="SeedTelemetry\\.stage"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_invalid_alpha_mode(self):
        """from_dict() rejects invalid alpha_mode values."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["alpha_mode"] = 999

        with pytest.raises(ValueError, match="SeedTelemetry\\.alpha_mode"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_invalid_alpha_algorithm(self):
        """from_dict() rejects invalid alpha_algorithm values."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["alpha_algorithm"] = 999

        with pytest.raises(ValueError, match="SeedTelemetry\\.alpha_algorithm"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_non_int_alpha_mode(self):
        """from_dict() rejects non-int alpha_mode values."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["alpha_mode"] = "HOLD"

        with pytest.raises(ValueError, match="SeedTelemetry\\.alpha_mode"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_bool_alpha_mode(self):
        """from_dict() rejects bool alpha_mode values (bool is subclass of int)."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["alpha_mode"] = True

        with pytest.raises(ValueError, match="SeedTelemetry\\.alpha_mode"):
            SeedTelemetry.from_dict(data)

    def test_telemetry_from_dict_rejects_bool_alpha_algorithm(self):
        """from_dict() rejects bool alpha_algorithm values."""
        telemetry = SeedTelemetry(seed_id="test")
        data = telemetry.to_dict()
        data["alpha_algorithm"] = False

        with pytest.raises(ValueError, match="SeedTelemetry\\.alpha_algorithm"):
            SeedTelemetry.from_dict(data)


class TestTelemetryToFeaturesIntegration:
    """End-to-end smoke: gradients -> SeedTelemetry -> to_features()."""

    def test_end_to_end_gradient_to_features(self):
        model = torch.nn.Linear(10, 2)
        x = torch.randn(32, 10)
        y_pred = model(x)
        loss = y_pred.mean()
        loss.backward()

        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        telemetry = SeedTelemetry(
            seed_id="test-seed",
            gradient_norm=stats["gradient_norm"],
            gradient_health=stats["gradient_health"],
            has_vanishing=stats["has_vanishing"],
            has_exploding=stats["has_exploding"],
            accuracy=75.0,
            epoch=10,
            max_epochs=25,
        )

        features = telemetry.to_features()

        assert len(features) == SeedTelemetry.feature_dim()
        assert all(isinstance(f, float) for f in features)
        assert all(-5.0 < f < 5.0 for f in features)

    def test_telemetry_features_deterministic(self):
        telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.5,
            gradient_health=0.85,
            accuracy=70.0,
            stage=3,
        )

        assert telemetry.to_features() == telemetry.to_features()

    def test_telemetry_features_different_inputs(self):
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

        assert telemetry1.to_features() != telemetry2.to_features()

    def test_telemetry_consistent_with_dimension_constant(self):
        telemetries = [
            SeedTelemetry(seed_id="test1", gradient_norm=1.0, accuracy=50.0),
            SeedTelemetry(seed_id="test2", gradient_norm=2.0, accuracy=60.0),
            SeedTelemetry(seed_id="test3", gradient_norm=3.0, accuracy=70.0),
        ]

        expected_dim = SeedTelemetry.feature_dim()

        for telemetry in telemetries:
            assert len(telemetry.to_features()) == expected_dim

