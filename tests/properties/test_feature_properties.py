"""Property-based tests for feature extraction.

Tests snapshot_to_features() and dimension consistency.
"""

import math
import pytest
from hypothesis import given, assume
from tests.strategies import training_snapshots, seed_telemetries

from esper.simic.comparison import snapshot_to_features
from esper.leyline import SeedTelemetry


class TestFeatureDimensions:
    """Test that feature dimensions are consistent."""

    @given(snapshot=training_snapshots())
    def test_features_without_telemetry_27_dim(self, snapshot):
        """Property: Features without telemetry must be 27-dim."""
        features = snapshot_to_features(snapshot, use_telemetry=False)

        assert len(features) == 27, f"Expected 27-dim, got {len(features)}"

    @given(
        snapshot=training_snapshots(has_active_seed=True),
        telemetry=seed_telemetries(),
    )
    def test_features_with_telemetry_37_dim(self, snapshot, telemetry):
        """Property: Features with telemetry must be 37-dim (27 + 10)."""
        features = snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=telemetry)

        assert len(features) == 37, f"Expected 37-dim, got {len(features)}"

    @given(snapshot=training_snapshots())
    def test_feature_dimensions_deterministic(self, snapshot):
        """Property: Same snapshot → same feature dimension."""
        f1 = snapshot_to_features(snapshot, use_telemetry=False)
        f2 = snapshot_to_features(snapshot, use_telemetry=False)

        assert len(f1) == len(f2), "Feature dimension non-deterministic"


class TestFeatureBounds:
    """Test that features are properly bounded/normalized."""

    @given(snapshot=training_snapshots())
    def test_features_finite(self, snapshot):
        """Property: All features must be finite (no NaN, no Inf)."""
        features = snapshot_to_features(snapshot, use_telemetry=False)

        for i, feat in enumerate(features):
            assert not math.isnan(feat), f"Feature {i} is NaN"
            assert not math.isinf(feat), f"Feature {i} is Inf"

    @given(snapshot=training_snapshots())
    def test_features_roughly_normalized(self, snapshot):
        """Property: Features should be finite and bounded.

        This test validates that features don't contain extreme values that
        could destabilize neural network training. The safe() clamping function
        in to_vector() ensures values are bounded to [-100, 100] or their
        natural ranges (e.g., accuracy in [0, 100]).
        """
        features = snapshot_to_features(snapshot, use_telemetry=False)

        for i, feat in enumerate(features):
            # All features should be finite
            assert math.isfinite(feat), f"Feature {i} is not finite: {feat}"

            # Features should be in a reasonable range for ML
            # (absolute value <= 1e6 to avoid overflow/underflow)
            assert abs(feat) <= 1e6, f"Feature {i} = {feat} has extreme magnitude"


class TestTelemetryEnforcement:
    """Test telemetry enforcement rules."""

    @given(snapshot=training_snapshots(has_active_seed=True))
    def test_telemetry_required_when_seed_active(self, snapshot):
        """Property: ValueError if telemetry required but missing."""
        with pytest.raises(ValueError, match="seed_telemetry is required"):
            snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)

    @given(snapshot=training_snapshots(has_active_seed=False))
    def test_telemetry_optional_when_no_seed(self, snapshot):
        """Property: Zero-padding allowed when no active seed."""
        # Should not raise
        features = snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)

        assert len(features) == 37
