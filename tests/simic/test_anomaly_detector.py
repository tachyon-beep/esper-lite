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

    def test_detect_value_collapse_late_training(self):
        """Detects value function collapse in late training phase."""
        detector = AnomalyDetector()
        # Late training (80% complete) - threshold is 0.1
        report = detector.check_value_function(
            explained_variance=0.05,  # Below 0.1 threshold
            current_episode=80,
            total_episodes=100,
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_no_value_collapse_early_training(self):
        """Low EV during warmup is expected, not anomalous."""
        detector = AnomalyDetector()
        # Warmup phase (5% complete) - threshold is -0.5
        report = detector.check_value_function(
            explained_variance=-0.3,  # Would fail late threshold (0.1) but passes warmup (-0.5)
            current_episode=5,
            total_episodes=100,
        )
        assert report.has_anomaly is False

    def test_value_collapse_threshold_progression(self):
        """Thresholds tighten as training progresses."""
        detector = AnomalyDetector()

        # Same EV (-0.3) tested at different phases
        ev = -0.3

        # Warmup (5%): threshold=-0.5, ev=-0.3 > -0.5 → OK
        warmup = detector.check_value_function(ev, current_episode=5, total_episodes=100)
        assert warmup.has_anomaly is False

        # Early (15%): threshold=-0.2, ev=-0.3 < -0.2 → FAIL
        early = detector.check_value_function(ev, current_episode=15, total_episodes=100)
        assert early.has_anomaly is True

    def test_phase_thresholds_scale_with_total_episodes(self):
        """Phase boundaries are proportional to total_episodes."""
        detector = AnomalyDetector()
        ev = -0.3

        # 10% of 100 episodes = episode 10 (warmup)
        small_run_warmup = detector.check_value_function(ev, current_episode=10, total_episodes=100)
        # 10% of 1000 episodes = episode 100 (warmup)
        large_run_warmup = detector.check_value_function(ev, current_episode=100, total_episodes=1000)

        # Both should be in warmup phase with same result
        assert small_run_warmup.has_anomaly == large_run_warmup.has_anomaly

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
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types

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
