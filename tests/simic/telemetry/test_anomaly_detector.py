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
            h_rms=5.0,
            c_rms=5.0,
            h_env_rms_max=5.0,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert not report.has_anomaly
        assert report.anomaly_types == []

    def test_nan_triggers_anomaly(self) -> None:
        """NaN in hidden state should trigger lstm_nan anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=5.0,
            h_env_rms_max=5.0,
            c_env_rms_max=5.0,
            has_nan=True,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_nan" in report.anomaly_types

    def test_inf_triggers_anomaly(self) -> None:
        """Inf in hidden state should trigger lstm_inf anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=5.0,
            h_env_rms_max=5.0,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=True,
        )
        assert report.has_anomaly
        assert "lstm_inf" in report.anomaly_types

    def test_h_explosion_triggers_anomaly(self) -> None:
        """h_rms above threshold should trigger lstm_h_explosion anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=15.0,  # > 10.0 default
            c_rms=5.0,
            h_env_rms_max=15.0,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_explosion" in report.anomaly_types

    def test_c_explosion_triggers_anomaly(self) -> None:
        """c_rms above threshold should trigger lstm_c_explosion anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=15.0,  # > 10.0 default
            h_env_rms_max=5.0,
            c_env_rms_max=15.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_c_explosion" in report.anomaly_types

    def test_h_vanishing_triggers_anomaly(self) -> None:
        """h_rms below threshold should trigger lstm_h_vanishing anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=1e-8,  # < 1e-6 default
            c_rms=5.0,
            h_env_rms_max=1e-8,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_vanishing" in report.anomaly_types

    def test_c_vanishing_triggers_anomaly(self) -> None:
        """c_rms below threshold should trigger lstm_c_vanishing anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=1e-8,  # < 1e-6 default
            h_env_rms_max=5.0,
            c_env_rms_max=1e-8,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_c_vanishing" in report.anomaly_types

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        detector = AnomalyDetector(
            lstm_max_rms=5.0,
            lstm_min_rms=1e-4,
        )
        # 7.5 is under default (10.0) but over custom (5.0)
        report = detector.check_lstm_health(
            h_rms=7.5,
            c_rms=5.0,
            h_env_rms_max=7.5,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_explosion" in report.anomaly_types

    def test_multiple_anomalies(self) -> None:
        """Multiple LSTM issues should all be reported."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=15.0,  # Explosion
            c_rms=1e-8,   # Vanishing
            h_env_rms_max=15.0,
            c_env_rms_max=1e-8,
            has_nan=True,  # NaN
            has_inf=False,
        )
        assert report.has_anomaly
        assert len(report.anomaly_types) >= 3
        assert "lstm_nan" in report.anomaly_types
        assert "lstm_h_explosion" in report.anomaly_types
        assert "lstm_c_vanishing" in report.anomaly_types


class TestPerHeadEntropyCollapse:
    """Tests for per-head entropy collapse detection with hysteresis."""

    def test_no_warning_on_single_collapse(self) -> None:
        """Single collapse should not trigger warning (hysteresis)."""
        detector = AnomalyDetector()
        head_entropies = {"blueprint": 0.01}  # Below 0.05

        report = detector.check_per_head_entropy_collapse(head_entropies)

        # First collapse - no warning yet (need N consecutive)
        assert not report.has_anomaly

    def test_warning_after_consecutive_collapses(self) -> None:
        """Warning should fire after N=3 consecutive collapses."""
        detector = AnomalyDetector()
        head_entropies = {"blueprint": 0.01}

        # First two - no warning
        detector.check_per_head_entropy_collapse(head_entropies)
        detector.check_per_head_entropy_collapse(head_entropies)

        # Third - warning fires
        report = detector.check_per_head_entropy_collapse(head_entropies)
        assert "entropy_collapse_blueprint" in report.anomaly_types

    def test_no_detection_when_healthy(self) -> None:
        """Should not flag healthy heads."""
        detector = AnomalyDetector()

        head_entropies = {
            "op": 0.5,
            "blueprint": 0.3,
            "slot": 0.4,
        }

        report = detector.check_per_head_entropy_collapse(head_entropies)

        assert not report.has_anomaly

    def test_recovery_requires_margin(self) -> None:
        """Recovery should require entropy above threshold * 1.5."""
        detector = AnomalyDetector()

        # Build up streak of 2 collapses
        detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        detector.check_per_head_entropy_collapse({"blueprint": 0.01})

        # Barely above threshold (0.06 > 0.05) - should NOT clear streak
        detector.check_per_head_entropy_collapse({"blueprint": 0.06})

        # Back to collapse - should continue streak and fire (now 3rd)
        report = detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        assert "entropy_collapse_blueprint" in report.anomaly_types

        # Well above threshold (0.10 > 0.075) - should clear
        detector.check_per_head_entropy_collapse({"blueprint": 0.10})

        # New collapse - should start fresh (only 1 in streak, no warning)
        report = detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        assert not report.has_anomaly

    def test_uses_per_head_thresholds(self) -> None:
        """Different heads should have different collapse thresholds."""
        detector = AnomalyDetector()

        # op threshold is 0.08, blueprint is 0.05
        head_entropies = {
            "op": 0.07,        # Below 0.08 -> collapse streak
            "blueprint": 0.06,  # Above 0.05 -> OK
        }

        # Need 3 consecutive to trigger
        for _ in range(3):
            report = detector.check_per_head_entropy_collapse(head_entropies)

        assert "entropy_collapse_op" in report.anomaly_types
        assert "entropy_collapse_blueprint" not in report.anomaly_types
