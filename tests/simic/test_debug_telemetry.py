"""Tests for debug-level telemetry."""

import torch
import torch.nn as nn
import pytest

from esper.simic.debug_telemetry import (
    LayerGradientStats,
    collect_per_layer_gradients,
    NumericalStabilityReport,
    check_numerical_stability,
)


class TestPerLayerGradients:
    """Tests for per-layer gradient collection."""

    @pytest.fixture
    def model_with_grads(self):
        """Create model with gradients."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        return model

    def test_collect_returns_list_of_stats(self, model_with_grads):
        """collect_per_layer_gradients returns list of LayerGradientStats."""
        stats = collect_per_layer_gradients(model_with_grads)
        assert len(stats) > 0
        assert all(isinstance(s, LayerGradientStats) for s in stats)

    def test_stats_have_layer_names(self, model_with_grads):
        """Each stats entry has layer name."""
        stats = collect_per_layer_gradients(model_with_grads)
        names = [s.layer_name for s in stats]
        assert "0.weight" in names or any("weight" in n for n in names)

    def test_stats_have_valid_values(self, model_with_grads):
        """Stats have reasonable gradient values."""
        stats = collect_per_layer_gradients(model_with_grads)
        for s in stats:
            assert s.grad_norm >= 0
            assert 0 <= s.zero_fraction <= 1
            assert s.nan_count == 0  # No NaNs in normal model


class TestNumericalStability:
    """Tests for numerical stability checking."""

    @pytest.fixture
    def healthy_model(self):
        """Create healthy model."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        return model, loss

    def test_healthy_model_passes(self, healthy_model):
        """Healthy model has clean stability report."""
        model, loss = healthy_model
        report = check_numerical_stability(model, loss)

        assert isinstance(report, NumericalStabilityReport)
        assert len(report.nan_in_weights) == 0
        assert len(report.nan_in_gradients) == 0
        assert report.loss_is_finite is True

    def test_detects_nan_in_gradients(self):
        """Detects NaN in gradients."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        # Inject NaN
        model.weight.grad[0, 0] = float('nan')

        report = check_numerical_stability(model)
        assert len(report.nan_in_gradients) > 0
