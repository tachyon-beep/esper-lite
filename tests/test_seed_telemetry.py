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


class TestSeedGradientCollector:
    """Tests for lightweight gradient collection."""

    def test_import_collector(self):
        """SeedGradientCollector should be importable."""
        from esper.simic.gradient_collector import SeedGradientCollector
        assert SeedGradientCollector is not None

    def test_collect_gradient_stats(self):
        """Collector should compute gradient stats from parameters."""
        import torch
        import torch.nn as nn
        from esper.simic.gradient_collector import SeedGradientCollector

        # Create a simple model with gradients
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        assert 'gradient_norm' in stats
        assert 'gradient_health' in stats
        assert 'has_vanishing' in stats
        assert 'has_exploding' in stats
        assert stats['gradient_norm'] >= 0
        assert 0 <= stats['gradient_health'] <= 1

    def test_detect_vanishing_gradients(self):
        """Collector should detect vanishing gradients."""
        import torch
        import torch.nn as nn
        from esper.simic.gradient_collector import SeedGradientCollector

        # Create model with tiny gradients
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.grad = torch.zeros_like(p) + 1e-10

        collector = SeedGradientCollector(vanishing_threshold=1e-7)
        stats = collector.collect(model.parameters())

        assert stats['has_vanishing'] is True

    def test_detect_exploding_gradients(self):
        """Collector should detect exploding gradients."""
        import torch
        import torch.nn as nn
        from esper.simic.gradient_collector import SeedGradientCollector

        # Create model with huge gradients
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 1000

        collector = SeedGradientCollector(exploding_threshold=100)
        stats = collector.collect(model.parameters())

        assert stats['has_exploding'] is True


class TestComparisonTelemetry:
    """Tests for comparison.py telemetry integration."""

    def test_snapshot_to_features_with_seed_telemetry(self):
        """snapshot_to_features should use seed telemetry when provided."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot
        from esper.leyline import SeedTelemetry

        snapshot = TrainingSnapshot(
            epoch=5,
            global_step=500,
            train_loss=0.5,
            val_loss=0.6,
            loss_delta=-0.1,
            train_accuracy=80.0,
            val_accuracy=75.0,
            accuracy_delta=2.0,
            plateau_epochs=0,
            best_val_accuracy=75.0,
            best_val_loss=0.6,
            loss_history_5=(0.9, 0.8, 0.7, 0.6, 0.6),
            accuracy_history_5=(60.0, 65.0, 70.0, 73.0, 75.0),
            has_active_seed=True,
            seed_stage=3,
            seed_epochs_in_stage=2,
            seed_alpha=0.0,
            seed_improvement=5.0,
            available_slots=0,
        )

        seed_telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.5,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            accuracy=75.0,
            accuracy_delta=5.0,
            epochs_in_stage=2,
            stage=3,
            alpha=0.0,
            epoch=5,
            max_epochs=25,
        )

        features = snapshot_to_features(
            snapshot,
            use_telemetry=True,
            seed_telemetry=seed_telemetry
        )

        # Should be 27 base + 10 seed = 37 dims
        assert len(features) == 37

        # Last 10 should be from seed telemetry
        seed_features = features[-10:]
        expected = seed_telemetry.to_features()
        assert seed_features == expected

    def test_snapshot_to_features_no_telemetry(self):
        """Without telemetry, should return 27 dims."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot

        snapshot = TrainingSnapshot(
            epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
            loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
            accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
            best_val_loss=1.0, loss_history_5=(1.0,)*5,
            accuracy_history_5=(50.0,)*5, has_active_seed=False,
            seed_stage=0, seed_epochs_in_stage=0, seed_alpha=0.0,
            seed_improvement=0.0, available_slots=1,
        )

        features = snapshot_to_features(snapshot, use_telemetry=False)
        assert len(features) == 27
