"""Tests for SeedTelemetry contract."""

import pytest
from datetime import datetime, timezone


class TestSeedTelemetry:
    """Tests for SeedTelemetry dataclass."""

    def test_import_from_leyline(self):
        """SeedTelemetry should be importable from leyline."""
        from esper.leyline import SeedTelemetry
        assert SeedTelemetry is not None

    def test_create_with_seed_id(self):
        """SeedTelemetry requires seed_id."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test_seed_1")
        assert telem.seed_id == "test_seed_1"

    def test_default_values(self):
        """SeedTelemetry has sensible defaults."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test")
        assert telem.gradient_norm == 0.0
        assert telem.gradient_health == 1.0
        assert telem.has_vanishing is False
        assert telem.has_exploding is False
        assert telem.accuracy == 0.0
        assert telem.stage == 1
        assert telem.alpha == 0.0

    def test_to_features_returns_10_dims(self):
        """to_features() returns exactly 10 dimensions."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test")
        features = telem.to_features()
        assert len(features) == 10
        assert SeedTelemetry.feature_dim() == 10

    def test_to_features_normalized_range(self):
        """Features should be normalized to approximately [0, 1]."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(
            seed_id="test",
            gradient_norm=5.0,
            gradient_health=0.8,
            has_vanishing=True,
            has_exploding=False,
            accuracy=75.0,
            accuracy_delta=2.5,
            epochs_in_stage=10,
            stage=4,  # BLENDING
            alpha=0.5,
            epoch=12,
            max_epochs=25,
        )
        features = telem.to_features()

        # All features should be in reasonable range
        for i, f in enumerate(features):
            assert -1.0 <= f <= 1.0, f"Feature {i} out of range: {f}"

    def test_stage_normalization(self):
        """Stage should normalize to [0, 1] for stages 1-7."""
        from esper.leyline import SeedTelemetry

        # Stage 1 (DORMANT) -> 0.0
        telem1 = SeedTelemetry(seed_id="test", stage=1)
        assert telem1.to_features()[7] == 0.0

        # Stage 7 (FOSSILIZED) -> 1.0
        telem7 = SeedTelemetry(seed_id="test", stage=7)
        assert telem7.to_features()[7] == 1.0

    def test_temporal_position(self):
        """Temporal position should be epoch/max_epochs."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test", epoch=10, max_epochs=20)
        features = telem.to_features()
        assert features[9] == 0.5  # 10/20

    def test_captured_at_timestamp(self):
        """SeedTelemetry should have a captured_at timestamp."""
        from esper.leyline import SeedTelemetry
        before = datetime.now(timezone.utc)
        telem = SeedTelemetry(seed_id="test")
        after = datetime.now(timezone.utc)
        assert before <= telem.captured_at <= after
