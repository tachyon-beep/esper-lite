"""Tests for anomaly detection in PPO training."""

import pytest

from esper.simic.anomaly_detector import AnomalyDetector, AnomalyReport


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

    def test_detect_value_collapse(self):
        """Detects value function collapse."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=-0.5,  # Negative = worse than mean
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

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
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types
