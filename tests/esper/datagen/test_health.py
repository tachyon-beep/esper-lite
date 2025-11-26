"""Tests for dataset health checks."""

import pytest
from esper.datagen.health import (
    DatasetHealthCheck,
    HealthCheckResult,
    check_action_coverage,
    check_action_entropy,
    check_policy_diversity,
)


class TestActionCoverage:
    def test_good_coverage(self):
        action_counts = {"WAIT": 800, "GERMINATE": 80, "ADVANCE": 70, "CULL": 50}
        result = check_action_coverage(action_counts, min_pct=0.02)
        assert result.passed
        assert result.level == "ok"

    def test_warning_coverage(self):
        # CULL is 1% - above error threshold (0.005) but below warning threshold (0.05)
        action_counts = {"WAIT": 900, "GERMINATE": 50, "ADVANCE": 40, "CULL": 10}
        result = check_action_coverage(action_counts, min_pct=0.005, warn_pct=0.05)
        assert result.passed  # Still passes (above error threshold)
        assert result.level == "warning"
        assert "CULL" in result.message

    def test_error_coverage(self):
        action_counts = {"WAIT": 990, "GERMINATE": 5, "ADVANCE": 4, "CULL": 1}
        result = check_action_coverage(action_counts, min_pct=0.02)
        assert not result.passed
        assert result.level == "error"


class TestActionEntropy:
    def test_high_entropy(self):
        # Uniform distribution = max entropy
        action_counts = {"WAIT": 250, "GERMINATE": 250, "ADVANCE": 250, "CULL": 250}
        result = check_action_entropy(action_counts, min_entropy=0.5)
        assert result.passed
        assert result.details["entropy"] > 1.3  # Near max of log(4) ≈ 1.39

    def test_low_entropy(self):
        # Single action dominates
        action_counts = {"WAIT": 950, "GERMINATE": 20, "ADVANCE": 20, "CULL": 10}
        result = check_action_entropy(action_counts, min_entropy=0.5, warn_entropy=0.3)
        assert result.level in ["warning", "error"]


class TestPolicyDiversity:
    def test_good_diversity(self):
        episodes = [
            {"behavior_policy": {"policy_id": "baseline"}},
            {"behavior_policy": {"policy_id": "aggressive"}},
            {"behavior_policy": {"policy_id": "conservative"}},
            {"behavior_policy": {"policy_id": "random"}},
            {"behavior_policy": {"policy_id": "early-intervener"}},
        ]
        result = check_policy_diversity(episodes, min_policies=5)
        assert result.passed

    def test_low_diversity(self):
        episodes = [
            {"behavior_policy": {"policy_id": "baseline"}},
            {"behavior_policy": {"policy_id": "baseline"}},
            {"behavior_policy": {"policy_id": "aggressive"}},
        ]
        result = check_policy_diversity(episodes, min_policies=5)
        assert not result.passed or result.level == "warning"


class TestDatasetHealthCheck:
    def test_full_check(self):
        # Create mock episodes with good coverage
        episodes = self._make_diverse_episodes()
        checker = DatasetHealthCheck()
        results = checker.run_all(episodes)

        assert "action_coverage" in results
        assert "action_entropy" in results
        assert "policy_diversity" in results

    def test_has_blocking_errors(self):
        checker = DatasetHealthCheck()
        # Episodes with terrible action coverage
        episodes = self._make_single_action_episodes()
        results = checker.run_all(episodes)

        assert checker.has_blocking_errors(results)

    def _make_diverse_episodes(self):
        return [
            {
                "behavior_policy": {"policy_id": f"policy_{i % 6}"},
                "decisions": [
                    {"action": {"action": "WAIT"}},
                    {"action": {"action": "GERMINATE"}},
                    {"action": {"action": "ADVANCE"}},
                    {"action": {"action": "CULL"}},
                ] * 5
            }
            for i in range(20)
        ]

    def _make_single_action_episodes(self):
        return [
            {
                "behavior_policy": {"policy_id": "baseline"},
                "decisions": [{"action": {"action": "WAIT"}}] * 100
            }
            for _ in range(10)
        ]
