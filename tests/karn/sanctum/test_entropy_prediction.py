"""Tests for entropy velocity and collapse risk prediction.

Task 3 of the Tamiyo debugging enhancements plan.
These functions help operators detect entropy collapse before it happens.
"""

import pytest
from collections import deque

from esper.karn.sanctum.schema import compute_entropy_velocity, compute_collapse_risk


class TestEntropyVelocity:
    """Test entropy velocity calculation."""

    def test_stable_entropy_zero_velocity(self):
        """Constant entropy should have ~0 velocity."""
        history = deque([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        velocity = compute_entropy_velocity(history)
        assert abs(velocity) < 0.01

    def test_declining_entropy_negative_velocity(self):
        """Declining entropy should have negative velocity."""
        history = deque([1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55])
        velocity = compute_entropy_velocity(history)
        assert velocity < -0.03  # About -0.05 per step

    def test_rising_entropy_positive_velocity(self):
        """Rising entropy should have positive velocity."""
        history = deque([0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        velocity = compute_entropy_velocity(history)
        assert velocity > 0.03

    def test_short_history_returns_zero(self):
        """With <5 samples, return 0 (insufficient data)."""
        history = deque([1.0, 0.9, 0.8])
        velocity = compute_entropy_velocity(history)
        assert velocity == 0.0

    def test_noisy_declining_entropy(self):
        """Declining entropy with realistic noise should still detect trend."""
        history = deque([0.82, 0.78, 0.81, 0.74, 0.69, 0.72, 0.65, 0.62, 0.58, 0.55])
        velocity = compute_entropy_velocity(history)
        assert velocity < -0.02  # Clear downward trend despite noise


class TestCollapseRisk:
    """Test entropy collapse risk scoring."""

    def test_stable_high_entropy_no_risk(self):
        """Stable entropy at healthy level should have low risk."""
        history = deque([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk < 0.1

    def test_declining_entropy_high_risk(self):
        """Rapidly declining entropy should have high risk."""
        history = deque([0.8, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.38])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk > 0.5

    def test_already_collapsed_max_risk(self):
        """Entropy already at critical should have risk=1.0."""
        history = deque([0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk == 1.0  # Exactly 1.0 when already collapsed

    def test_rising_entropy_low_risk(self):
        """Rising entropy should have minimal risk (just proximity-based)."""
        history = deque([0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk < 0.2  # Small proximity risk but no velocity risk

    def test_zero_velocity_no_crash(self):
        """Zero velocity should not cause divide-by-zero."""
        history = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk < 0.3  # Just proximity risk, no velocity component

    def test_hysteresis_prevents_minor_fluctuation(self):
        """Risk score should not change if delta < hysteresis threshold."""
        history = deque([0.6, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42])

        # First call establishes baseline
        risk1 = compute_collapse_risk(history, previous_risk=0.0, hysteresis=0.08)

        # Slightly modified history (should produce similar but not identical base_risk)
        history2 = deque([0.6, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44, 0.41])
        risk2 = compute_collapse_risk(history2, previous_risk=risk1, hysteresis=0.08)

        # Risk should be sticky due to hysteresis (returns previous_risk)
        assert risk2 == risk1

    def test_hysteresis_allows_significant_change(self):
        """Risk score should update if delta > hysteresis threshold."""
        # Start with moderate decline
        history1 = deque([0.8, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62])
        risk1 = compute_collapse_risk(history1, previous_risk=0.0, hysteresis=0.08)

        # Significant change - now rapidly approaching critical
        history2 = deque([0.45, 0.42, 0.39, 0.36, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29])
        risk2 = compute_collapse_risk(history2, previous_risk=risk1, hysteresis=0.08)

        # Risk should have increased significantly (>0.08 change)
        assert risk2 > risk1 + 0.08
