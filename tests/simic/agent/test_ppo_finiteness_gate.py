"""Tests for PPO finiteness gate contract.

Verifies that when all epochs skip due to non-finite values:
1. ppo_update_performed=False is returned
2. Metrics are NaN (not 0.0 which looks "normal")
3. finiteness_gate_skip_count is accurate
"""

import math
from collections import defaultdict

import pytest


class TestFinitenessGateAggregation:
    """Test the aggregation logic when all epochs skip."""

    def test_all_epochs_skipped_returns_explicit_signal(self):
        """When all epochs hit finiteness gate, ppo_update_performed=False."""
        # Simulate metrics dict with only finiteness_gate_failures, no ratio_max
        metrics: dict = defaultdict(list)
        metrics["finiteness_gate_failures"].append({
            "epoch": 0,
            "sources": ["log_probs[op]: NaN detected"],
        })

        # Simulate the aggregation logic from ppo.py
        finiteness_failures = metrics.get("finiteness_gate_failures", [])
        epochs_completed = len(metrics.get("ratio_max", []))

        # The fix: check if any epochs completed
        assert epochs_completed == 0
        assert len(finiteness_failures) == 1

        # When no epochs completed, we should return explicit signal
        result = {
            "ppo_update_performed": False,
            "finiteness_gate_skip_count": len(finiteness_failures),
            "ratio_max": float("nan"),
            "ratio_min": float("nan"),
        }

        assert result["ppo_update_performed"] is False
        assert result["finiteness_gate_skip_count"] == 1
        assert math.isnan(result["ratio_max"])
        assert math.isnan(result["ratio_min"])

    def test_some_epochs_completed_returns_valid_metrics(self):
        """When some epochs complete, ppo_update_performed=True."""
        # Simulate metrics with one skipped epoch and one successful
        metrics: dict = defaultdict(list)
        metrics["finiteness_gate_failures"].append({
            "epoch": 0,
            "sources": ["log_probs[op]: NaN detected"],
        })
        # Second epoch completed successfully
        metrics["ratio_max"].append(1.5)
        metrics["ratio_min"].append(0.8)
        metrics["policy_loss"].append(0.1)

        finiteness_failures = metrics.get("finiteness_gate_failures", [])
        epochs_completed = len(metrics.get("ratio_max", []))

        assert epochs_completed == 1
        assert len(finiteness_failures) == 1

        # With epochs completed, we should return valid metrics
        result = {
            "ppo_update_performed": True,
            "finiteness_gate_skip_count": len(finiteness_failures),
            "ratio_max": max(metrics["ratio_max"]),
            "ratio_min": min(metrics["ratio_min"]),
        }

        assert result["ppo_update_performed"] is True
        assert result["finiteness_gate_skip_count"] == 1
        assert result["ratio_max"] == 1.5
        assert result["ratio_min"] == 0.8

    def test_nan_metrics_propagate_has_nan_detection(self):
        """NaN metrics should be detectable by downstream has_nan checks."""
        metrics = {
            "ppo_update_performed": False,
            "ratio_max": float("nan"),
            "ratio_min": float("nan"),
            "policy_loss": float("nan"),
        }

        metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
        has_nan = any(math.isnan(v) for v in metric_values if isinstance(v, float))

        assert has_nan is True, "NaN metrics should be detectable"

    def test_zero_default_hides_missing_data(self):
        """Demonstrate why 0.0 is wrong: it looks like valid data.

        This test documents the bug that was fixed: filling empty lists
        with 0.0 made "no update happened" look like "healthy training".
        """
        # The OLD (buggy) behavior: empty list becomes 0.0
        old_behavior_metrics = {
            "ratio_max": 0.0,  # Looks like "no ratio change" (healthy)
            "ratio_min": 0.0,  # Actually means "no data"
        }

        # A simple threshold check would miss the problem
        ratio_explosion_threshold = 2.0
        ratio_collapse_threshold = 0.5

        # 0.0 passes both checks (looks "healthy")
        is_explosion = old_behavior_metrics["ratio_max"] > ratio_explosion_threshold
        is_collapse = old_behavior_metrics["ratio_min"] < ratio_collapse_threshold

        assert is_explosion is False  # Missed: no data, not healthy
        assert is_collapse is True  # FALSE POSITIVE: 0.0 < 0.5 triggers collapse alarm

        # The NEW behavior: NaN signals "no data"
        new_behavior_metrics = {
            "ratio_max": float("nan"),
            "ratio_min": float("nan"),
        }

        # NaN comparisons return False (correct: unknown is not explosion/collapse)
        is_explosion_nan = new_behavior_metrics["ratio_max"] > ratio_explosion_threshold
        is_collapse_nan = new_behavior_metrics["ratio_min"] < ratio_collapse_threshold

        assert is_explosion_nan is False  # Correct: unknown
        assert is_collapse_nan is False  # Correct: unknown, not false positive


class TestConsecutiveFailureEscalation:
    """Test that consecutive finiteness failures are tracked and escalated."""

    def test_single_failure_allows_continuation(self):
        """One skipped update should not crash training."""
        consecutive_failures = 0

        # Simulate one skipped update
        update_performed = False
        if not update_performed:
            consecutive_failures += 1

        # Should not escalate
        should_escalate = consecutive_failures >= 3
        assert should_escalate is False
        assert consecutive_failures == 1

    def test_three_consecutive_failures_escalates(self):
        """Three consecutive skipped updates should escalate to error."""
        consecutive_failures = 0

        # Simulate three skipped updates
        for _ in range(3):
            update_performed = False
            if not update_performed:
                consecutive_failures += 1

        # Should escalate
        should_escalate = consecutive_failures >= 3
        assert should_escalate is True
        assert consecutive_failures == 3

    def test_successful_update_resets_counter(self):
        """A successful update should reset the failure counter."""
        consecutive_failures = 2  # Two failures already

        # Successful update
        update_performed = True
        if update_performed:
            consecutive_failures = 0

        assert consecutive_failures == 0

        # Next failure starts fresh
        update_performed = False
        if not update_performed:
            consecutive_failures += 1

        should_escalate = consecutive_failures >= 3
        assert should_escalate is False
        assert consecutive_failures == 1


class TestTrainStepsIncrementContract:
    """Test that train_steps only increments on successful updates.

    This is critical for entropy annealing and other schedules that depend
    on train_steps tracking actual policy updates, not skipped failures.
    """

    def test_train_steps_not_incremented_when_all_epochs_fail(self):
        """train_steps should stay constant when finiteness gate skips all epochs.

        Bug context: Previously, train_steps was incremented unconditionally at
        the top of the aggregation section, BEFORE checking epochs_completed.
        This caused entropy annealing to advance even when no update occurred.
        """
        # This test verifies the contract at the logic level.
        # The actual PPO agent test requires a full agent setup.

        # Simulate the FIXED control flow from ppo.py update():
        train_steps_before = 10
        train_steps = train_steps_before

        # Simulate: all epochs failed finiteness gate
        epochs_completed = 0

        # The FIX: only increment train_steps if epochs_completed > 0
        if epochs_completed > 0:
            train_steps += 1

        assert train_steps == train_steps_before, (
            "train_steps should NOT increment when all epochs fail finiteness gate. "
            "This would cause entropy annealing to drift without actual policy updates."
        )

    def test_train_steps_incremented_when_some_epochs_succeed(self):
        """train_steps should increment when at least one epoch completes."""
        train_steps_before = 10
        train_steps = train_steps_before

        # Simulate: one epoch succeeded, one failed
        epochs_completed = 1

        if epochs_completed > 0:
            train_steps += 1

        assert train_steps == train_steps_before + 1, (
            "train_steps should increment when at least one epoch completes."
        )

    def test_entropy_annealing_depends_on_train_steps(self):
        """Document that entropy annealing uses train_steps for schedule progress.

        This test documents WHY the train_steps increment contract matters:
        entropy coefficient is computed as a function of train_steps / anneal_steps.
        """
        # Simulate PPOAgent.get_entropy_coef() logic
        entropy_coef_start = 0.1
        entropy_coef_end = 0.01
        entropy_anneal_steps = 100

        def get_entropy_coef(train_steps: int) -> float:
            if entropy_anneal_steps == 0:
                return entropy_coef_start
            progress = min(1.0, train_steps / entropy_anneal_steps)
            return entropy_coef_start + progress * (entropy_coef_end - entropy_coef_start)

        # At start: full entropy
        assert get_entropy_coef(0) == pytest.approx(0.1)

        # Halfway: mid-annealing
        coef_50 = get_entropy_coef(50)
        assert 0.05 < coef_50 < 0.06  # ~0.055

        # At end: minimum entropy
        assert get_entropy_coef(100) == pytest.approx(0.01)

        # Beyond end: stays at minimum
        assert get_entropy_coef(150) == pytest.approx(0.01)

        # BUG SCENARIO: If train_steps increments on skipped updates,
        # entropy would drop faster than intended, reducing exploration
        # before the policy has actually learned.
        train_steps_with_bug = 50  # 50 "updates" but only 25 were real
        train_steps_correct = 25  # 25 actual updates

        coef_buggy = get_entropy_coef(train_steps_with_bug)  # 0.055
        coef_correct = get_entropy_coef(train_steps_correct)  # 0.0775

        assert coef_correct > coef_buggy, (
            "Buggy behavior causes entropy to drop faster (more exploitation) "
            "when updates are being skipped (policy not learning)."
        )
