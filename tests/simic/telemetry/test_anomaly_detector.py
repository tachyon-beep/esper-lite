"""Tests for AnomalyDetector scheduling logic."""

from __future__ import annotations

from esper.simic.telemetry import AnomalyDetector


def test_ev_threshold_non_decreasing_over_training() -> None:
    """EV threshold should get stricter later in training (or stay equal)."""
    detector = AnomalyDetector()

    early = detector.get_ev_threshold(current_episode=100, total_episodes=1000)
    late = detector.get_ev_threshold(current_episode=900, total_episodes=1000)

    assert isinstance(early, float)
    assert isinstance(late, float)
    assert late >= early


class TestCheckLstmHealth:
    """Tests for LSTM hidden state health checking (B7-DRL-04)."""

    def test_healthy_state_no_anomaly(self) -> None:
        """Healthy LSTM state should not trigger anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=5.0,
            c_norm=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert not report.has_anomaly
        assert report.anomaly_types == []

    def test_nan_triggers_anomaly(self) -> None:
        """NaN in hidden state should trigger lstm_nan anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=5.0,
            c_norm=5.0,
            has_nan=True,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_nan" in report.anomaly_types

    def test_inf_triggers_anomaly(self) -> None:
        """Inf in hidden state should trigger lstm_inf anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=5.0,
            c_norm=5.0,
            has_nan=False,
            has_inf=True,
        )
        assert report.has_anomaly
        assert "lstm_inf" in report.anomaly_types

    def test_h_explosion_triggers_anomaly(self) -> None:
        """h_norm above threshold should trigger lstm_h_explosion anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=150.0,  # > 100.0 default
            c_norm=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_explosion" in report.anomaly_types

    def test_c_explosion_triggers_anomaly(self) -> None:
        """c_norm above threshold should trigger lstm_c_explosion anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=5.0,
            c_norm=150.0,  # > 100.0 default
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_c_explosion" in report.anomaly_types

    def test_h_vanishing_triggers_anomaly(self) -> None:
        """h_norm below threshold should trigger lstm_h_vanishing anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=1e-8,  # < 1e-6 default
            c_norm=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_vanishing" in report.anomaly_types

    def test_c_vanishing_triggers_anomaly(self) -> None:
        """c_norm below threshold should trigger lstm_c_vanishing anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=5.0,
            c_norm=1e-8,  # < 1e-6 default
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_c_vanishing" in report.anomaly_types

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        detector = AnomalyDetector(
            lstm_max_norm=50.0,
            lstm_min_norm=1e-4,
        )
        # 75.0 is under default (100.0) but over custom (50.0)
        report = detector.check_lstm_health(
            h_norm=75.0,
            c_norm=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_explosion" in report.anomaly_types

    def test_multiple_anomalies(self) -> None:
        """Multiple LSTM issues should all be reported."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_norm=150.0,  # Explosion
            c_norm=1e-8,   # Vanishing
            has_nan=True,  # NaN
            has_inf=False,
        )
        assert report.has_anomaly
        assert len(report.anomaly_types) >= 3
        assert "lstm_nan" in report.anomaly_types
        assert "lstm_h_explosion" in report.anomaly_types
        assert "lstm_c_vanishing" in report.anomaly_types

