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



