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

