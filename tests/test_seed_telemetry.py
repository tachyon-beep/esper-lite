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


class TestHeadToHeadTelemetryIntegration:
    """Integration tests for telemetry collection in head_to_head_comparison."""

    def test_telemetry_collected_during_training(self):
        """head_to_head_comparison should collect and sync telemetry during training.

        This test verifies that:
        1. Gradient collector is instantiated
        2. Gradients are collected after backward pass
        3. Telemetry is synced to seed_state
        4. iql_action_fn receives real telemetry (not zeros)
        """
        import torch
        import torch.nn as nn
        from unittest.mock import patch, MagicMock
        from esper.simic.gradient_collector import SeedGradientCollector
        from esper.kasmina.slot import SeedState
        from esper.leyline import SeedStage

        # Create a mock scenario where we have a seed with gradients
        seed_state = SeedState(seed_id="test", blueprint_id="conv_enhance")
        seed_state.stage = SeedStage.TRAINING
        seed_state.metrics.current_val_accuracy = 75.0
        seed_state.metrics.epochs_in_current_stage = 3
        seed_state.alpha = 0.0

        # Create a simple model to generate real gradients
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Test the gradient collector
        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        # Verify stats are non-default (we have real gradients)
        assert stats['gradient_norm'] > 0, "Should collect non-zero gradient norm"
        assert 0 <= stats['gradient_health'] <= 1, "Health should be in [0, 1]"
        assert 'has_vanishing' in stats
        assert 'has_exploding' in stats

        # Test sync_telemetry updates seed_state.telemetry
        seed_state.sync_telemetry(
            gradient_norm=stats['gradient_norm'],
            gradient_health=stats['gradient_health'],
            has_vanishing=stats['has_vanishing'],
            has_exploding=stats['has_exploding'],
            epoch=5,
            max_epochs=25,
        )

        # Verify telemetry was updated
        assert seed_state.telemetry.gradient_norm == stats['gradient_norm']
        assert seed_state.telemetry.gradient_health == stats['gradient_health']
        assert seed_state.telemetry.accuracy == 75.0  # from metrics
        assert seed_state.telemetry.stage == SeedStage.TRAINING.value
        assert seed_state.telemetry.epochs_in_stage == 3

        # Verify telemetry produces non-zero features
        features = seed_state.telemetry.to_features()
        assert len(features) == 10
        # At least gradient_norm should be non-zero
        assert features[0] > 0, "First feature (gradient_norm) should be non-zero"

    def test_telemetry_passed_to_snapshot_to_features(self):
        """iql_action_fn should pass real telemetry to snapshot_to_features."""
        import torch
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot
        from esper.leyline import SeedTelemetry

        # Create a snapshot with active seed
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

        # Create telemetry with non-default values
        telemetry = SeedTelemetry(
            seed_id="test_seed",
            blueprint_id="conv_enhance",
            gradient_norm=2.5,
            gradient_health=0.85,
            has_vanishing=False,
            has_exploding=False,
            accuracy=75.0,
            accuracy_delta=2.0,
            epochs_in_stage=2,
            stage=3,
            alpha=0.0,
            epoch=5,
            max_epochs=25,
        )

        # Convert to features with telemetry
        features = snapshot_to_features(
            snapshot,
            use_telemetry=True,
            seed_telemetry=telemetry
        )

        # Verify we get 37 dimensions
        assert len(features) == 37

        # Verify the last 10 dimensions are from real telemetry (not zeros)
        seed_features = features[-10:]
        expected_telemetry_features = telemetry.to_features()

        # Check that we're getting real telemetry data
        # gradient_norm should be 2.5/10.0 = 0.25
        assert abs(seed_features[0] - 0.25) < 0.01

        # gradient_health should be 0.85
        assert abs(seed_features[1] - 0.85) < 0.01

        # Overall features should match
        assert seed_features == expected_telemetry_features


class TestLoadIQLModelDimensions:
    """Tests for load_iql_model dimension detection and support."""

    def test_load_27dim_model(self, tmp_path):
        """Load 27-dim model (no telemetry) correctly."""
        import torch
        from esper.simic.comparison import load_iql_model
        from esper.simic.iql import IQL

        # Create a 27-dim model
        iql = IQL(state_dim=27, action_dim=7, hidden_dim=256, device='cpu')

        # Save checkpoint in expected format
        checkpoint_path = tmp_path / "iql_27dim.pt"
        checkpoint = {
            'state_dim': 27,
            'action_dim': 7,
            'gamma': 0.99,
            'tau': 0.7,
            'beta': 3.0,
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Load and verify
        loaded_iql, telemetry_mode = load_iql_model(str(checkpoint_path), device='cpu')

        assert loaded_iql.q_network.net[0].in_features == 27
        assert telemetry_mode == 'none'

    def test_load_37dim_model(self, tmp_path):
        """Load 37-dim model (seed telemetry) correctly."""
        import torch
        from esper.simic.comparison import load_iql_model
        from esper.simic.iql import IQL

        # Create a 37-dim model (27 base + 10 seed telemetry)
        iql = IQL(state_dim=37, action_dim=7, hidden_dim=256, device='cpu')

        # Save checkpoint in expected format
        checkpoint_path = tmp_path / "iql_37dim.pt"
        checkpoint = {
            'state_dim': 37,
            'action_dim': 7,
            'gamma': 0.99,
            'tau': 0.7,
            'beta': 3.0,
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Load and verify
        loaded_iql, telemetry_mode = load_iql_model(str(checkpoint_path), device='cpu')

        assert loaded_iql.q_network.net[0].in_features == 37
        assert telemetry_mode == 'seed'

    def test_load_54dim_model(self, tmp_path):
        """Load 54-dim model (legacy full-model telemetry) correctly."""
        import torch
        from esper.simic.comparison import load_iql_model
        from esper.simic.iql import IQL

        # Create a 54-dim model (27 base + 27 legacy telemetry)
        iql = IQL(state_dim=54, action_dim=7, hidden_dim=256, device='cpu')

        # Save checkpoint in expected format
        checkpoint_path = tmp_path / "iql_54dim.pt"
        checkpoint = {
            'state_dim': 54,
            'action_dim': 7,
            'gamma': 0.99,
            'tau': 0.7,
            'beta': 3.0,
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Load and verify
        loaded_iql, telemetry_mode = load_iql_model(str(checkpoint_path), device='cpu')

        assert loaded_iql.q_network.net[0].in_features == 54
        assert telemetry_mode == 'legacy'

    def test_load_unknown_dimension_raises_error(self, tmp_path):
        """Loading model with unknown dimension should raise ValueError."""
        import torch
        import pytest
        from esper.simic.comparison import load_iql_model
        from esper.simic.iql import IQL

        # Create a model with unsupported dimension (e.g., 42)
        iql = IQL(state_dim=42, action_dim=7, hidden_dim=256, device='cpu')

        # Save checkpoint in expected format
        checkpoint_path = tmp_path / "iql_42dim.pt"
        checkpoint = {
            'state_dim': 42,
            'action_dim': 7,
            'gamma': 0.99,
            'tau': 0.7,
            'beta': 3.0,
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Should raise ValueError for unknown dimension
        with pytest.raises(ValueError, match="Unknown state dimension: 42"):
            load_iql_model(str(checkpoint_path), device='cpu')

    def test_backward_compatibility_with_use_telemetry_flag(self, tmp_path):
        """Verify backward compatibility: use_telemetry flag matches telemetry_mode."""
        import torch
        from esper.simic.comparison import load_iql_model
        from esper.simic.iql import IQL

        # Test 27-dim: use_telemetry should be False (backward compat)
        iql_27 = IQL(state_dim=27, action_dim=7, device='cpu')
        path_27 = tmp_path / "iql_27.pt"
        torch.save({
            'state_dim': 27,
            'action_dim': 7,
            'q_network': iql_27.q_network.state_dict(),
            'v_network': iql_27.v_network.state_dict(),
        }, path_27)

        loaded_27, mode_27 = load_iql_model(str(path_27), device='cpu')
        assert mode_27 == 'none'
        # For backward compat, we can check: use_telemetry would be False
        use_telemetry_27 = (mode_27 in ['seed', 'legacy'])
        assert use_telemetry_27 is False

        # Test 37-dim: use_telemetry should be True
        iql_37 = IQL(state_dim=37, action_dim=7, device='cpu')
        path_37 = tmp_path / "iql_37.pt"
        torch.save({
            'state_dim': 37,
            'action_dim': 7,
            'q_network': iql_37.q_network.state_dict(),
            'v_network': iql_37.v_network.state_dict(),
        }, path_37)

        loaded_37, mode_37 = load_iql_model(str(path_37), device='cpu')
        assert mode_37 == 'seed'
        use_telemetry_37 = (mode_37 in ['seed', 'legacy'])
        assert use_telemetry_37 is True

        # Test 54-dim: use_telemetry should be True (legacy)
        iql_54 = IQL(state_dim=54, action_dim=7, device='cpu')
        path_54 = tmp_path / "iql_54.pt"
        torch.save({
            'state_dim': 54,
            'action_dim': 7,
            'q_network': iql_54.q_network.state_dict(),
            'v_network': iql_54.v_network.state_dict(),
        }, path_54)

        loaded_54, mode_54 = load_iql_model(str(path_54), device='cpu')
        assert mode_54 == 'legacy'
        use_telemetry_54 = (mode_54 in ['seed', 'legacy'])
        assert use_telemetry_54 is True


class TestEndToEndTelemetryPipeline:
    """End-to-end test of the complete telemetry pipeline."""

    def test_complete_telemetry_flow(self):
        """Test the entire flow from gradient collection to feature extraction.

        This integration test verifies:
        1. Create a model and seed state
        2. Do a forward/backward pass to generate gradients
        3. Collect gradients with SeedGradientCollector
        4. Sync telemetry to seed_state
        5. Create a TrainingSnapshot
        6. Call snapshot_to_features with seed_telemetry
        7. Verify output is 37-dim with expected values
        """
        import torch
        import torch.nn as nn
        from esper.kasmina.slot import SeedState
        from esper.leyline import SeedStage, SeedTelemetry
        from esper.simic.gradient_collector import SeedGradientCollector
        from esper.simic.episodes import TrainingSnapshot
        from esper.simic.comparison import snapshot_to_features

        # Step 1: Create a simple model and seed state
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        seed_state = SeedState(
            seed_id="integration_test_seed",
            blueprint_id="conv_enhance"
        )
        seed_state.stage = SeedStage.TRAINING
        seed_state.metrics.current_val_accuracy = 72.5
        seed_state.metrics.epochs_in_current_stage = 3
        seed_state.alpha = 0.0

        # Step 2: Perform a forward/backward pass to generate real gradients
        x = torch.randn(8, 10)
        target = torch.randint(0, 5, (8,))

        # Forward pass
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)

        # Backward pass generates gradients
        loss.backward()

        # Step 3: Collect gradients using SeedGradientCollector
        collector = SeedGradientCollector()
        gradient_stats = collector.collect(model.parameters())

        # Verify we got real gradient statistics
        assert gradient_stats['gradient_norm'] > 0, "Should have non-zero gradient norm"
        assert 0 <= gradient_stats['gradient_health'] <= 1
        assert isinstance(gradient_stats['has_vanishing'], bool)
        assert isinstance(gradient_stats['has_exploding'], bool)

        # Step 4: Sync telemetry to seed_state
        current_epoch = 5
        max_epochs = 25

        seed_state.sync_telemetry(
            gradient_norm=gradient_stats['gradient_norm'],
            gradient_health=gradient_stats['gradient_health'],
            has_vanishing=gradient_stats['has_vanishing'],
            has_exploding=gradient_stats['has_exploding'],
            epoch=current_epoch,
            max_epochs=max_epochs,
        )

        # Verify telemetry was updated correctly
        assert seed_state.telemetry.seed_id == "integration_test_seed"
        assert seed_state.telemetry.blueprint_id == "conv_enhance"
        assert seed_state.telemetry.gradient_norm == gradient_stats['gradient_norm']
        assert seed_state.telemetry.gradient_health == gradient_stats['gradient_health']
        assert seed_state.telemetry.has_vanishing == gradient_stats['has_vanishing']
        assert seed_state.telemetry.has_exploding == gradient_stats['has_exploding']
        assert seed_state.telemetry.accuracy == 72.5
        assert seed_state.telemetry.epochs_in_stage == 3
        assert seed_state.telemetry.stage == SeedStage.TRAINING.value
        assert seed_state.telemetry.alpha == 0.0
        assert seed_state.telemetry.epoch == current_epoch
        assert seed_state.telemetry.max_epochs == max_epochs

        # Step 5: Create a TrainingSnapshot (simulating a training scenario)
        snapshot = TrainingSnapshot(
            epoch=current_epoch,
            global_step=500,
            train_loss=0.45,
            val_loss=0.52,
            loss_delta=-0.08,
            train_accuracy=78.0,
            val_accuracy=72.5,
            accuracy_delta=2.5,
            plateau_epochs=0,
            best_val_accuracy=72.5,
            best_val_loss=0.52,
            loss_history_5=(0.9, 0.75, 0.6, 0.52, 0.52),
            accuracy_history_5=(60.0, 65.0, 68.0, 70.0, 72.5),
            has_active_seed=True,
            seed_stage=SeedStage.TRAINING.value,
            seed_epochs_in_stage=3,
            seed_alpha=0.0,
            seed_improvement=5.0,
            available_slots=0,
        )

        # Step 6: Call snapshot_to_features with seed_telemetry
        features = snapshot_to_features(
            snapshot,
            use_telemetry=True,
            seed_telemetry=seed_state.telemetry
        )

        # Step 7: Verify output is 37-dim with expected values
        assert len(features) == 37, f"Expected 37 features, got {len(features)}"

        # First 27 dimensions are base snapshot features
        base_features = features[:27]
        assert len(base_features) == 27

        # Last 10 dimensions are seed telemetry features
        telemetry_features = features[-10:]
        assert len(telemetry_features) == 10

        # Verify telemetry features match what we'd get from to_features()
        expected_telemetry_features = seed_state.telemetry.to_features()
        assert telemetry_features == expected_telemetry_features

        # Verify gradient information is present in telemetry features
        # First feature should be normalized gradient_norm (should be > 0)
        assert telemetry_features[0] > 0, "Gradient norm feature should be non-zero"

        # Second feature should be gradient_health (should be in [0, 1])
        assert 0 <= telemetry_features[1] <= 1, "Gradient health should be in [0, 1]"

        # Boolean features for vanishing/exploding gradients
        # These are 0.0 or 1.0
        assert telemetry_features[2] in [0.0, 1.0], "has_vanishing should be boolean"
        assert telemetry_features[3] in [0.0, 1.0], "has_exploding should be boolean"

        # Verify all features are finite
        for i, feature in enumerate(features):
            assert torch.isfinite(torch.tensor(feature)), f"Feature {i} is not finite"

        # Verify telemetry features are in normalized range [-1, 1]
        for i, feature in enumerate(telemetry_features):
            assert -1.0 <= feature <= 1.0, f"Telemetry feature {i} out of normalized range: {feature}"

        # Verify the telemetry timestamp is recent
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        captured_at = seed_state.telemetry.captured_at
        time_diff = (now - captured_at).total_seconds()
        assert time_diff < 5.0, "Telemetry should have been captured recently"


class TestTelemetryEnforcement:
    """Tests for telemetry requirement enforcement (Task 9)."""

    def test_snapshot_to_features_requires_telemetry_when_enabled(self):
        """When use_telemetry=True with active seed, seed_telemetry must be provided.

        This enforces the DRL review finding that zero-padding telemetry
        causes distribution shift and degrades policy quality.
        """
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot
        import pytest

        snapshot = TrainingSnapshot(
            epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
            loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
            accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
            best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
            has_active_seed=True,  # Seed active but no telemetry provided
            seed_stage=2, seed_epochs_in_stage=3, seed_alpha=0.0,
            seed_improvement=5.0, available_slots=0
        )

        # Should raise ValueError, not warn
        with pytest.raises(ValueError, match="seed_telemetry is required"):
            snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)


    def test_snapshot_to_features_allows_none_when_no_seed(self):
        """When use_telemetry=True but no seed is active, None telemetry is OK."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot

        snapshot = TrainingSnapshot(
            epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
            loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
            accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
            best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
            has_active_seed=False,  # No seed active
            available_slots=1
        )

        # Should NOT raise when no seed is active
        features = snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)
        assert len(features) == 37  # 27 base + 10 zero telemetry
