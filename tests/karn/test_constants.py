"""Tests for Karn constants module."""

import pytest
from esper.karn.constants import (
    AnomalyThresholds,
    PolicyThresholds,
    HealthThresholds,
    TUIThresholds,
)


class TestAnomalyThresholds:
    """Test anomaly detection thresholds."""

    def test_loss_spike_threshold_is_positive(self) -> None:
        assert AnomalyThresholds.LOSS_SPIKE_MULTIPLIER > 0

    def test_accuracy_drop_threshold_is_positive(self) -> None:
        assert AnomalyThresholds.ACCURACY_DROP_POINTS > 0

    def test_gradient_explosion_multiplier_is_large(self) -> None:
        # Should be at least 10x to count as "explosion"
        assert AnomalyThresholds.GRADIENT_EXPLOSION_MULTIPLIER >= 10.0


class TestPolicyThresholds:
    """Test PPO policy anomaly thresholds."""

    def test_value_std_threshold_is_small(self) -> None:
        # Value collapse threshold should be small (near zero)
        assert 0 < PolicyThresholds.VALUE_STD_COLLAPSE < 0.1

    def test_entropy_threshold_is_reasonable(self) -> None:
        # Entropy collapse should trigger before reaching 0
        assert 0 < PolicyThresholds.ENTROPY_COLLAPSE < 0.5

    def test_kl_threshold_is_positive(self) -> None:
        assert PolicyThresholds.KL_SPIKE > 0


class TestHealthThresholds:
    """Test system health thresholds."""

    def test_gpu_warning_is_high(self) -> None:
        # GPU warning should be above 80% utilization
        assert HealthThresholds.GPU_UTILIZATION_WARNING > 0.8

    def test_grad_norm_warning_less_than_error(self) -> None:
        assert HealthThresholds.GRAD_NORM_WARNING < HealthThresholds.GRAD_NORM_ERROR


class TestTUIThresholds:
    """Test TUI display thresholds."""

    def test_entropy_warning_less_than_max(self) -> None:
        assert TUIThresholds.ENTROPY_WARNING < TUIThresholds.ENTROPY_MAX

    def test_entropy_critical_less_than_warning(self) -> None:
        assert TUIThresholds.ENTROPY_CRITICAL < TUIThresholds.ENTROPY_WARNING
