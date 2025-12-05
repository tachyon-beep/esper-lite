"""Tests for feature extraction."""

import pytest


class TestCounterfactualInObservation:
    """Counterfactual contribution should be in observation space."""

    def test_tensor_schema_has_counterfactual(self):
        """TensorSchema should include SEED_COUNTERFACTUAL."""
        from esper.leyline.signals import TensorSchema, TENSOR_SCHEMA_SIZE

        assert hasattr(TensorSchema, 'SEED_COUNTERFACTUAL')
        # Should be index 27 (after AVAILABLE_SLOTS at 26)
        assert TensorSchema.SEED_COUNTERFACTUAL == 27
        # Schema size should be 35 (V3 with host state + blueprint one-hot)
        assert TENSOR_SCHEMA_SIZE == 35

    def test_fast_signals_has_counterfactual(self):
        """FastTrainingSignals should include seed_counterfactual."""
        from esper.leyline.signals import FastTrainingSignals

        # Check the field exists
        empty = FastTrainingSignals.empty()
        assert hasattr(empty, 'seed_counterfactual')
        assert empty.seed_counterfactual == 0.0

    def test_to_vector_includes_counterfactual(self):
        """to_vector() should include counterfactual at correct index.

        Note: FastTrainingSignals.to_vector() returns 30 features (base features
        without blueprint encoding). Blueprint one-hot is only added by
        obs_to_base_features() which gets blueprint info from the observation dict.
        """
        from esper.leyline.signals import FastTrainingSignals, TensorSchema

        signals = FastTrainingSignals.empty()._replace(seed_counterfactual=1.5)
        vec = signals.to_vector()

        assert len(vec) == 30  # Base features without blueprint one-hot
        assert vec[TensorSchema.SEED_COUNTERFACTUAL] == pytest.approx(1.5 / 10.0, rel=0.01)

    def test_obs_to_base_features_includes_counterfactual(self):
        """obs_to_base_features should handle seed_counterfactual."""
        from esper.simic.features import obs_to_base_features

        obs = {
            'epoch': 10,
            'global_step': 1000,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'loss_delta': -0.01,
            'train_accuracy': 80.0,
            'val_accuracy': 75.0,
            'accuracy_delta': 1.0,
            'plateau_epochs': 2,
            'best_val_accuracy': 76.0,
            'best_val_loss': 0.55,
            'loss_history_5': [0.7, 0.65, 0.6, 0.58, 0.55],
            'accuracy_history_5': [70, 72, 73, 74, 75],
            'has_active_seed': 1,
            'seed_stage': 6,  # PROBATIONARY
            'seed_epochs_in_stage': 3,
            'seed_alpha': 0.8,
            'seed_improvement': 2.0,
            'available_slots': 1,
            'seed_counterfactual': 1.5,  # NEW
        }

        features = obs_to_base_features(obs)

        assert len(features) == 35, f"Expected 35 features, got {len(features)}"
        # Counterfactual should be at index 27 (before host state at 28-29, blueprint at 30-34)
        assert features[27] == pytest.approx(1.5 / 10.0, rel=0.01)


class TestHostStateObservability:
    """Host network state should be observable by policy."""

    def test_tensor_schema_has_host_signals(self):
        """TensorSchema should include host state signals."""
        from esper.leyline.signals import TensorSchema, TENSOR_SCHEMA_SIZE

        assert hasattr(TensorSchema, 'HOST_GRAD_NORM')
        assert hasattr(TensorSchema, 'HOST_LEARNING_PHASE')
        assert TensorSchema.HOST_GRAD_NORM == 28
        assert TensorSchema.HOST_LEARNING_PHASE == 29
        assert TENSOR_SCHEMA_SIZE == 35

    def test_fast_signals_has_host_state(self):
        """FastTrainingSignals should include host state."""
        from esper.leyline.signals import FastTrainingSignals

        empty = FastTrainingSignals.empty()
        assert hasattr(empty, 'host_grad_norm')
        assert hasattr(empty, 'host_learning_phase')

    def test_obs_to_base_features_includes_host_state(self):
        """obs_to_base_features should handle host state."""
        from esper.simic.features import obs_to_base_features

        obs = {
            'epoch': 10,
            'global_step': 1000,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'loss_delta': -0.01,
            'train_accuracy': 80.0,
            'val_accuracy': 75.0,
            'accuracy_delta': 1.0,
            'plateau_epochs': 2,
            'best_val_accuracy': 76.0,
            'best_val_loss': 0.55,
            'loss_history_5': [0.7, 0.65, 0.6, 0.58, 0.55],
            'accuracy_history_5': [70, 72, 73, 74, 75],
            'has_active_seed': 1,
            'seed_stage': 6,
            'seed_epochs_in_stage': 3,
            'seed_alpha': 0.8,
            'seed_improvement': 2.0,
            'available_slots': 1,
            'seed_counterfactual': 1.5,
            'host_grad_norm': 0.5,
            'host_learning_phase': 0.4,
        }

        features = obs_to_base_features(obs)

        assert len(features) == 35, f"Expected 35 features, got {len(features)}"


class TestBlueprintOneHotEncoding:
    """Blueprint ID should be in observation space as one-hot encoding."""

    def test_base_features_includes_blueprint_one_hot(self):
        """Base features should include one-hot blueprint encoding."""
        from esper.simic.features import obs_to_base_features

        obs = {
            'epoch': 10,
            'global_step': 500,
            'train_loss': 1.5,
            'val_loss': 1.6,
            'loss_delta': -0.1,
            'train_accuracy': 60.0,
            'val_accuracy': 58.0,
            'accuracy_delta': 2.0,
            'plateau_epochs': 3,
            'best_val_accuracy': 60.0,
            'best_val_loss': 1.4,
            'loss_history_5': [2.0, 1.8, 1.7, 1.6, 1.5],
            'accuracy_history_5': [40.0, 45.0, 50.0, 55.0, 58.0],
            'has_active_seed': 1.0,
            'seed_stage': 3,
            'seed_epochs_in_stage': 5,
            'seed_alpha': 0.0,
            'seed_improvement': 2.0,
            'available_slots': 0,
            'seed_counterfactual': 0.0,
            'host_grad_norm': 0.5,
            'host_learning_phase': 0.4,
            # New: blueprint encoding (0=none, 1-5=blueprint index)
            'seed_blueprint_id': 2,  # attention blueprint
            'num_blueprints': 5,
        }

        features = obs_to_base_features(obs, max_epochs=25)

        # Should now be 35 features (30 base + 5 one-hot blueprint)
        assert len(features) == 35

        # Blueprint one-hot should be at indices 30-34
        blueprint_one_hot = features[30:35]
        assert blueprint_one_hot == [0.0, 1.0, 0.0, 0.0, 0.0]  # Index 1 is hot (blueprint_id=2, 0-indexed)

        # Test with no active seed (blueprint_id=0)
        obs['seed_blueprint_id'] = 0
        features_no_seed = obs_to_base_features(obs, max_epochs=25)
        assert features_no_seed[30:35] == [0.0, 0.0, 0.0, 0.0, 0.0]  # All zeros
