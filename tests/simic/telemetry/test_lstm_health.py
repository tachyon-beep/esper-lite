"""Tests for LSTM hidden state health monitoring.

Tests the LSTMHealthMetrics dataclass and compute_lstm_health function.
"""

from __future__ import annotations

import pytest
import torch

from esper.simic.telemetry.lstm_health import LSTMHealthMetrics, compute_lstm_health


class TestLSTMHealthMetrics:
    """Tests for LSTMHealthMetrics dataclass."""

    def test_healthy_state(self) -> None:
        """Normal metrics should be considered healthy."""
        metrics = LSTMHealthMetrics(
            h_norm=5.0,
            c_norm=5.0,
            h_max=2.0,
            c_max=2.0,
            has_nan=False,
            has_inf=False,
        )
        assert metrics.is_healthy()

    def test_nan_detected_unhealthy(self) -> None:
        """NaN in hidden state makes it unhealthy."""
        metrics = LSTMHealthMetrics(
            h_norm=5.0,
            c_norm=5.0,
            h_max=2.0,
            c_max=2.0,
            has_nan=True,
            has_inf=False,
        )
        assert not metrics.is_healthy()

    def test_inf_detected_unhealthy(self) -> None:
        """Inf in hidden state makes it unhealthy."""
        metrics = LSTMHealthMetrics(
            h_norm=5.0,
            c_norm=5.0,
            h_max=2.0,
            c_max=2.0,
            has_nan=False,
            has_inf=True,
        )
        assert not metrics.is_healthy()

    def test_exploding_h_norm_unhealthy(self) -> None:
        """h_norm above max_norm is unhealthy (explosion)."""
        metrics = LSTMHealthMetrics(
            h_norm=150.0,  # > 100.0 default
            c_norm=5.0,
            h_max=50.0,
            c_max=2.0,
            has_nan=False,
            has_inf=False,
        )
        assert not metrics.is_healthy()

    def test_exploding_c_norm_unhealthy(self) -> None:
        """c_norm above max_norm is unhealthy (explosion)."""
        metrics = LSTMHealthMetrics(
            h_norm=5.0,
            c_norm=150.0,  # > 100.0 default
            h_max=2.0,
            c_max=50.0,
            has_nan=False,
            has_inf=False,
        )
        assert not metrics.is_healthy()

    def test_vanishing_h_norm_unhealthy(self) -> None:
        """h_norm below min_norm is unhealthy (vanishing)."""
        metrics = LSTMHealthMetrics(
            h_norm=1e-8,  # < 1e-6 default
            c_norm=5.0,
            h_max=1e-9,
            c_max=2.0,
            has_nan=False,
            has_inf=False,
        )
        assert not metrics.is_healthy()

    def test_vanishing_c_norm_unhealthy(self) -> None:
        """c_norm below min_norm is unhealthy (vanishing)."""
        metrics = LSTMHealthMetrics(
            h_norm=5.0,
            c_norm=1e-8,  # < 1e-6 default
            h_max=2.0,
            c_max=1e-9,
            has_nan=False,
            has_inf=False,
        )
        assert not metrics.is_healthy()

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        metrics = LSTMHealthMetrics(
            h_norm=50.0,
            c_norm=50.0,
            h_max=25.0,
            c_max=25.0,
            has_nan=False,
            has_inf=False,
        )
        # With defaults, 50.0 is healthy (< 100)
        assert metrics.is_healthy()
        # With stricter threshold, 50.0 is unhealthy
        assert not metrics.is_healthy(max_norm=40.0)

    def test_to_dict(self) -> None:
        """to_dict produces expected telemetry format."""
        metrics = LSTMHealthMetrics(
            h_norm=1.5,
            c_norm=2.5,
            h_max=0.8,
            c_max=1.2,
            has_nan=False,
            has_inf=True,
        )
        d = metrics.to_dict()
        assert d["lstm_h_norm"] == 1.5
        assert d["lstm_c_norm"] == 2.5
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
        assert metrics.h_norm > 0
        assert metrics.c_norm > 0
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
        assert metrics.h_norm > 0
        assert metrics.c_norm > 0

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
