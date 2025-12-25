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


def test_explained_variance_thresholds_drl_correct():
    """EV thresholds should follow DRL best practices.

    Theory: EV=0 means value function explains nothing (useless baseline).
    EV<0 means value function increases variance (actively harmful).
    WARNING at 0.3, CRITICAL at 0.0 per DRL expert review.
    """
    from esper.karn.constants import TUIThresholds

    # Warning: value function weak (not useless, but not helping much)
    assert TUIThresholds.EXPLAINED_VAR_WARNING == 0.3

    # Critical: value function useless or harmful
    assert TUIThresholds.EXPLAINED_VAR_CRITICAL == 0.0


def test_kl_thresholds_exist():
    """KL divergence should have both warning and critical thresholds."""
    from esper.karn.constants import TUIThresholds

    assert hasattr(TUIThresholds, 'KL_WARNING')
    assert hasattr(TUIThresholds, 'KL_CRITICAL')
    assert TUIThresholds.KL_WARNING == 0.015
    assert TUIThresholds.KL_CRITICAL == 0.03


def test_advantage_thresholds_exist():
    """Advantage std should have tiered thresholds."""
    from esper.karn.constants import TUIThresholds

    # Normal range: ~1.0 (normalized advantages)
    # Warning: too high (>2.0) or too low (<0.5)
    # Critical: extremely high (>3.0) or collapsed (<0.1)
    assert TUIThresholds.ADVANTAGE_STD_WARNING == 2.0
    assert TUIThresholds.ADVANTAGE_STD_CRITICAL == 3.0
    assert TUIThresholds.ADVANTAGE_STD_LOW_WARNING == 0.5
    assert TUIThresholds.ADVANTAGE_STD_COLLAPSED == 0.1
