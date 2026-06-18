"""Tests for anomaly detection in PPO training."""


import pytest

from esper.simic.telemetry import AnomalyDetector


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def test_detect_ratio_explosion(self):
        """Detects ratio explosion."""
        detector = AnomalyDetector()
        report = detector.check_ratios(
            ratio_max=6.0,  # > 5.0 threshold
            ratio_min=0.5,
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types

    def test_detect_ratio_collapse(self):
        """Detects ratio collapse (min too low)."""
        detector = AnomalyDetector()
        report = detector.check_ratios(
            ratio_max=2.0,
            ratio_min=0.05,  # < 0.1 threshold
        )
        assert report.has_anomaly is True
        assert "ratio_collapse" in report.anomaly_types

    def test_healthy_ratios_no_anomaly(self):
        """Healthy ratios produce no anomaly."""
        detector = AnomalyDetector()
        report = detector.check_ratios(
            ratio_max=2.0,
            ratio_min=0.5,
        )
        assert report.has_anomaly is False
        assert len(report.anomaly_types) == 0

    def test_detect_ratio_nan_inf(self):
        """Detects NaN/Inf ratio values (C3 fix: IEEE 754 comparison trap).

        NaN comparisons always return False in IEEE 754, so NaN > threshold
        would silently pass without explicit check. This test verifies the
        math.isfinite() guard catches non-finite values.
        """
        detector = AnomalyDetector()

        # NaN ratio_max
        report = detector.check_ratios(ratio_max=float("nan"), ratio_min=0.5)
        assert report.has_anomaly is True
        assert "ratio_nan_inf" in report.anomaly_types
        assert "ratio_max=nan" in report.details["ratio_nan_inf"]

        # +Inf ratio_max (common from exp() overflow in PPO ratio computation)
        report = detector.check_ratios(ratio_max=float("inf"), ratio_min=0.5)
        assert report.has_anomaly is True
        assert "ratio_nan_inf" in report.anomaly_types
        assert "ratio_max=inf" in report.details["ratio_nan_inf"]

        # -Inf ratio_min
        report = detector.check_ratios(ratio_max=2.0, ratio_min=float("-inf"))
        assert report.has_anomaly is True
        assert "ratio_nan_inf" in report.anomaly_types
        assert "ratio_min=-inf" in report.details["ratio_nan_inf"]

        # Both NaN
        report = detector.check_ratios(ratio_max=float("nan"), ratio_min=float("nan"))
        assert report.has_anomaly is True
        assert "ratio_nan_inf" in report.anomaly_types
        assert "ratio_max=nan" in report.details["ratio_nan_inf"]
        assert "ratio_min=nan" in report.details["ratio_nan_inf"]

    def test_detect_value_collapse_late_training(self):
        """Detects value function collapse via the scale-anchored robust signal.

        EV-telemetry-robustness (GATE = ROBUST-ANCHORED ONLY): explained_variance is a
        pure diagnostic and never gates. value_collapse fires solely on the robust signal
        (value_loss PRIMARY / bellman_error SECONDARY). Here a bad value_loss fires it.
        """
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=0.05,  # diagnostic only — does not gate
            current_episode=80,
            total_episodes=100,
            value_loss=10.0,  # bad robust signal (> value_loss_threshold 5.0)
            ev_low_return_variance=False,
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_value_collapse_does_not_fire_on_ev_alone(self):
        """GATE = ROBUST-ANCHORED ONLY: low EV with healthy robust signals must NOT fire."""
        detector = AnomalyDetector()
        # EV is low but the scale-anchored value fit is genuinely healthy.
        report = detector.check_value_function(
            explained_variance=0.05,
            current_episode=80,
            total_episodes=100,
            bellman_error=0.5,  # healthy
            value_loss=0.099,   # healthy (median)
            ev_low_return_variance=False,
        )
        assert report.has_anomaly is False
        assert "value_collapse" not in report.anomaly_types

    def test_no_value_collapse_with_healthy_robust_signal(self):
        """A healthy robust signal produces no value collapse regardless of EV / phase."""
        detector = AnomalyDetector()
        # EV is negative but the robust signals are healthy (and default to quiescent 0.0).
        report = detector.check_value_function(
            explained_variance=-0.3,  # diagnostic only
            current_episode=5,
            total_episodes=100,
        )
        assert report.has_anomaly is False

    def test_get_ev_threshold_remains_a_diagnostic_accessor(self):
        """get_ev_threshold still tightens by phase but no longer gates value_collapse.

        The phase-dependent EV thresholds are retained for diagnostic display only; the
        firing decision is robust-anchored and ignores EV entirely.
        """
        detector = AnomalyDetector()

        warmup = detector.get_ev_threshold(current_episode=5, total_episodes=100)
        late = detector.get_ev_threshold(current_episode=90, total_episodes=100)
        # Phase thresholds still tighten over training (warmup -0.5 -> late +0.1).
        assert late > warmup

        # But a crater EV below every phase threshold does NOT fire with healthy robust
        # signals — EV is diagnostic only.
        report = detector.check_value_function(
            explained_variance=-5.0,  # below warmup, early, mid, and late thresholds
            current_episode=90,
            total_episodes=100,
            bellman_error=0.5,
            value_loss=0.099,
        )
        assert report.has_anomaly is False
        assert "value_collapse" not in report.anomaly_types

    def test_requires_episode_info_for_value_collapse_thresholds(self):
        """Value collapse thresholds require explicit episode context (no shims).

        B7-CR-01 fix: Arguments are now required (no trap defaults).
        Missing arguments raise TypeError at call time, not ValueError.
        """
        detector = AnomalyDetector()
        with pytest.raises(TypeError):
            detector.check_value_function(explained_variance=0.05)  # type: ignore[call-arg]

    def test_detect_numerical_instability(self):
        """Detects NaN/Inf in metrics."""
        detector = AnomalyDetector()
        report = detector.check_numerical_stability(
            has_nan=True,
            has_inf=False,
        )
        assert report.has_anomaly is True
        assert "numerical_instability" in report.anomaly_types

    def test_combined_check(self):
        """Can check all anomalies at once."""
        detector = AnomalyDetector()
        report = detector.check_all(
            ratio_max=6.0,
            ratio_min=0.5,
            explained_variance=0.5,
            has_nan=False,
            has_inf=False,
            current_episode=1,
            total_episodes=100,
            value_collapse_applicable=True,
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types

    def test_combined_check_skips_value_collapse_when_not_applicable(self):
        """Combined checks should preserve other anomalies when value collapse is out of scope."""
        detector = AnomalyDetector()
        report = detector.check_all(
            ratio_max=6.0,
            ratio_min=0.5,
            explained_variance=-1.0,
            has_nan=False,
            has_inf=False,
            current_episode=100,
            total_episodes=100,
            value_collapse_applicable=False,
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types
        assert "value_collapse" not in report.anomaly_types

    def test_requires_total_episodes_for_value_collapse_thresholds(self):
        """Value collapse thresholds require total_episodes (no shims)."""
        detector = AnomalyDetector()
        with pytest.raises(ValueError):
            detector.check_value_function(
                explained_variance=0.05,
                current_episode=10,
                total_episodes=0,
            )

    def test_detect_gradient_norm_drift(self):
        """Detects gradient norm drift exceeding threshold (B7-DRL-01)."""
        detector = AnomalyDetector()
        report = detector.check_gradient_drift(
            norm_drift=0.6,  # > 0.5 threshold
            health_drift=0.1,
        )
        assert report.has_anomaly is True
        assert "gradient_norm_drift" in report.anomaly_types

    def test_detect_gradient_health_drift(self):
        """Detects gradient health drift exceeding threshold (B7-DRL-01)."""
        detector = AnomalyDetector()
        report = detector.check_gradient_drift(
            norm_drift=0.1,
            health_drift=0.7,  # > 0.5 threshold
        )
        assert report.has_anomaly is True
        assert "gradient_health_drift" in report.anomaly_types

    def test_no_gradient_drift_below_threshold(self):
        """No anomaly when drift is within acceptable range (B7-DRL-01)."""
        detector = AnomalyDetector()
        report = detector.check_gradient_drift(
            norm_drift=0.3,  # Below 0.5 threshold
            health_drift=0.2,  # Below 0.5 threshold
        )
        assert report.has_anomaly is False
        assert len(report.anomaly_types) == 0

    def test_gradient_drift_custom_threshold(self):
        """Custom threshold works for gradient drift (B7-DRL-01)."""
        detector = AnomalyDetector()
        # With stricter threshold of 0.2
        report = detector.check_gradient_drift(
            norm_drift=0.3,  # > 0.2 threshold
            health_drift=0.1,
            drift_threshold=0.2,
        )
        assert report.has_anomaly is True
        assert "gradient_norm_drift" in report.anomaly_types
