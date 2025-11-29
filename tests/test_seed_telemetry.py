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


class TestSeedStateTelemetry:
    """Tests for SeedState.telemetry integration."""

    def test_seed_state_has_telemetry(self):
        """SeedState should have a telemetry field."""
        from esper.kasmina.slot import SeedState
        state = SeedState(seed_id="test", blueprint_id="conv_enhance")
        assert hasattr(state, 'telemetry')
        assert state.telemetry is not None

    def test_telemetry_initialized_with_seed_info(self):
        """Telemetry should be initialized with seed_id and blueprint_id."""
        from esper.kasmina.slot import SeedState
        state = SeedState(seed_id="seed_1", blueprint_id="attention")
        assert state.telemetry.seed_id == "seed_1"
        assert state.telemetry.blueprint_id == "attention"

    def test_sync_telemetry_updates_from_metrics(self):
        """sync_telemetry should copy values from metrics."""
        from esper.kasmina.slot import SeedState
        from esper.leyline import SeedStage

        state = SeedState(seed_id="test", blueprint_id="conv")
        state.stage = SeedStage.TRAINING
        state.metrics.current_val_accuracy = 75.0
        state.metrics.epochs_in_current_stage = 5
        state.alpha = 0.3

        state.sync_telemetry(
            gradient_norm=2.5,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            epoch=10,
            max_epochs=25,
        )

        assert state.telemetry.accuracy == 75.0
        assert state.telemetry.epochs_in_stage == 5
        assert state.telemetry.stage == SeedStage.TRAINING.value
        assert state.telemetry.alpha == 0.3
        assert state.telemetry.gradient_norm == 2.5
        assert state.telemetry.gradient_health == 0.9
        assert state.telemetry.epoch == 10
        assert state.telemetry.max_epochs == 25
