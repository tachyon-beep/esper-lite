"""Tests for LSTM hidden state health monitoring.

Tests the LSTMHealthMetrics dataclass and compute_lstm_health function.
"""

from __future__ import annotations

import pytest
import torch

from esper.simic.telemetry.lstm_health import LSTMHealthMetrics, compute_lstm_health


def _make_lstm_metrics(**overrides) -> LSTMHealthMetrics:
    """Create LSTMHealthMetrics with sane defaults for unit tests."""
    base = {
        # Capacity/load (not scale-free)
        "h_l2_total": 1.0,
        "c_l2_total": 1.0,
        # Scale-free health (RMS)
        "h_rms": 1.0,
        "c_rms": 1.0,
        # Per-env RMS stats
        "h_env_rms_mean": 1.0,
        "h_env_rms_max": 1.0,
        "c_env_rms_mean": 1.0,
        "c_env_rms_max": 1.0,
        # Extremes / finiteness
        "h_max": 0.0,
        "c_max": 0.0,
        "has_nan": False,
        "has_inf": False,
    }
    base.update(overrides)
    return LSTMHealthMetrics(**base)


class TestLSTMHealthMetrics:
    """Tests for LSTMHealthMetrics dataclass."""

    def test_healthy_state(self) -> None:
        """Normal metrics should be considered healthy."""
        metrics = _make_lstm_metrics(h_rms=2.0, c_rms=2.0, h_max=2.0, c_max=2.0)
        assert metrics.is_healthy()

    def test_nan_detected_unhealthy(self) -> None:
        """NaN in hidden state makes it unhealthy."""
        metrics = _make_lstm_metrics(h_rms=2.0, c_rms=2.0, has_nan=True)
        assert not metrics.is_healthy()

    def test_inf_detected_unhealthy(self) -> None:
        """Inf in hidden state makes it unhealthy."""
        metrics = _make_lstm_metrics(h_rms=2.0, c_rms=2.0, has_inf=True)
        assert not metrics.is_healthy()

    def test_exploding_h_rms_unhealthy(self) -> None:
        """h_rms above max_rms is unhealthy (explosion/saturation)."""
        metrics = _make_lstm_metrics(h_rms=11.0)  # > 10.0 default
        assert not metrics.is_healthy()

    def test_exploding_c_rms_unhealthy(self) -> None:
        """c_rms above max_rms is unhealthy (explosion/saturation)."""
        metrics = _make_lstm_metrics(c_rms=11.0)  # > 10.0 default
        assert not metrics.is_healthy()

    def test_vanishing_h_rms_unhealthy(self) -> None:
        """h_rms below min_rms is unhealthy (vanishing)."""
        metrics = _make_lstm_metrics(h_rms=1e-8, h_max=1e-9)
        assert not metrics.is_healthy()

    def test_vanishing_c_rms_unhealthy(self) -> None:
        """c_rms below min_rms is unhealthy (vanishing)."""
        metrics = _make_lstm_metrics(c_rms=1e-8, c_max=1e-9)
        assert not metrics.is_healthy()

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        metrics = _make_lstm_metrics(h_rms=5.0, c_rms=5.0)
        # With defaults, 5.0 is healthy (< 10)
        assert metrics.is_healthy()
        # With stricter threshold, 50.0 is unhealthy
        assert not metrics.is_healthy(max_rms=4.0)

    def test_to_dict(self) -> None:
        """to_dict produces expected telemetry format."""
        metrics = _make_lstm_metrics(
            h_l2_total=10.0,
            c_l2_total=20.0,
            h_rms=1.5,
            c_rms=2.5,
            h_env_rms_mean=1.1,
            h_env_rms_max=1.9,
            c_env_rms_mean=2.0,
            c_env_rms_max=3.1,
            h_max=0.8,
            c_max=1.2,
            has_inf=True,
        )
        d = metrics.to_dict()
        assert d["lstm_h_l2_total"] == 10.0
        assert d["lstm_c_l2_total"] == 20.0
        assert d["lstm_h_rms"] == 1.5
        assert d["lstm_c_rms"] == 2.5
        assert d["lstm_h_env_rms_mean"] == 1.1
        assert d["lstm_h_env_rms_max"] == 1.9
        assert d["lstm_c_env_rms_mean"] == 2.0
        assert d["lstm_c_env_rms_max"] == 3.1
        assert d["lstm_h_max"] == 0.8
        assert d["lstm_c_max"] == 1.2
        assert d["lstm_has_nan"] is False
        assert d["lstm_has_inf"] is True


class TestComputeLstmHealth:
    """Tests for compute_lstm_health function."""

    def test_none_hidden_returns_none(self) -> None:
        """No hidden state returns None."""
        assert compute_lstm_health(None) is None

    def test_normal_hidden_state(self) -> None:
        """Normal tensors produce valid metrics."""
        h = torch.randn(1, 1, 64)  # [num_layers, batch, hidden_dim]
        c = torch.randn(1, 1, 64)
        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.h_l2_total > 0
        assert metrics.c_l2_total > 0
        assert metrics.h_rms > 0
        assert metrics.c_rms > 0
        assert not metrics.has_nan
        assert not metrics.has_inf

    def test_nan_in_h_detected(self) -> None:
        """NaN in h tensor is detected."""
        h = torch.randn(1, 1, 64)
        h[0, 0, 0] = float("nan")
        c = torch.randn(1, 1, 64)
        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.has_nan

    def test_nan_in_c_detected(self) -> None:
        """NaN in c tensor is detected."""
        h = torch.randn(1, 1, 64)
        c = torch.randn(1, 1, 64)
        c[0, 0, 0] = float("nan")
        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.has_nan

    def test_inf_in_h_detected(self) -> None:
        """Inf in h tensor is detected."""
        h = torch.randn(1, 1, 64)
        h[0, 0, 0] = float("inf")
        c = torch.randn(1, 1, 64)
        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.has_inf

    def test_inf_in_c_detected(self) -> None:
        """Inf in c tensor is detected."""
        h = torch.randn(1, 1, 64)
        c = torch.randn(1, 1, 64)
        c[0, 0, 0] = float("-inf")
        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.has_inf

    def test_multi_layer_lstm(self) -> None:
        """Multi-layer LSTM hidden states work correctly."""
        h = torch.randn(3, 1, 128)  # 3 layers
        c = torch.randn(3, 1, 128)
        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.h_l2_total > 0
        assert metrics.c_l2_total > 0

    def test_max_values_tracked(self) -> None:
        """h_max and c_max track maximum absolute values."""
        h = torch.zeros(1, 1, 10)
        c = torch.zeros(1, 1, 10)
        h[0, 0, 5] = 42.0
        c[0, 0, 3] = -37.0

        metrics = compute_lstm_health((h, c))

        assert metrics is not None
        assert metrics.h_max == 42.0
        assert metrics.c_max == 37.0  # Absolute value
