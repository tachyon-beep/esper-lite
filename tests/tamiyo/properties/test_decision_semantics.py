"""Property-based tests for Tamiyo decision semantics.

Tier 2 properties verify semantic correctness of decisions:
1. Decision completeness - decide() always returns valid TamiyoDecision
2. Action coverage - Every action type is reachable from some input
3. Target consistency - CULL/FOSSILIZE always specify target_seed_id
4. WAIT as default - WAIT returned when no criteria met
5. Stage appropriateness - Actions appropriate for seed stage
6. Reason non-empty - Decisions always have reasons

These properties ensure the decision logic is semantically sound and complete.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from esper.leyline import SeedStage
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig
from esper.tamiyo.decisions import TamiyoDecision

# Import Tamiyo-specific strategies
from tests.tamiyo.strategies import (
    tamiyo_configs,
    mock_seed_states,
    mock_seed_states_at_stage,
    mock_training_signals,
)
from tests.tamiyo.strategies.decision_strategies import (
    failing_seed_states,
    succeeding_seed_states,
    probationary_seed_states,
    unstabilized_signals,
    plateau_signals,
    early_training_signals,
)


# =============================================================================
# Property: Decision Completeness
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestDecisionCompleteness:
    """Every valid input produces a valid decision without exceptions."""

    @given(
        config=tamiyo_configs(),
        signals=mock_training_signals(),
        seeds=st.lists(mock_seed_states(), max_size=3),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_decide_always_returns_decision(self, config, signals, seeds):
        """Property: decide() never raises, always returns TamiyoDecision."""
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Must not raise
        decision = policy.decide(signals, seeds)

        # Must return correct type
        assert isinstance(decision, TamiyoDecision)
        assert decision.action is not None

    @given(signals=mock_training_signals())
    @settings(max_examples=100)
    def test_decide_with_empty_seeds(self, signals):
        """Property: decide() works with empty seed list."""
        policy = HeuristicTamiyo(topology="cnn")

        decision = policy.decide(signals, active_seeds=[])

        assert isinstance(decision, TamiyoDecision)
        # With no seeds, can only WAIT or GERMINATE
        assert decision.action.name in ["WAIT", "GERMINATE_CONV_LIGHT",
                                         "GERMINATE_CONV_HEAVY", "GERMINATE_ATTENTION",
                                         "GERMINATE_NORM", "GERMINATE_DEPTHWISE"]

    @given(
        signals=mock_training_signals(),
        seed=mock_seed_states(),
    )
    @settings(max_examples=100)
    def test_decide_with_single_seed(self, signals, seed):
        """Property: decide() works with single seed."""
        policy = HeuristicTamiyo(topology="cnn")

        decision = policy.decide(signals, active_seeds=[seed])

        assert isinstance(decision, TamiyoDecision)


# =============================================================================
# Property: Action Reachability
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestActionReachability:
    """Every action type can be reached from valid inputs."""

    @given(signals=plateau_signals(min_plateau=5))
    @settings(max_examples=50)
    def test_germinate_reachable(self, signals):
        """Property: GERMINATE is reachable when stabilized + plateau + no seeds."""
        # Use config that allows early germination
        config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=3,
            min_epochs_before_germinate=0,
            embargo_epochs_after_cull=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Ensure stabilization
        signals.metrics.host_stabilized = 1
        signals.metrics.epoch = 10  # Past min_epochs

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name.startswith("GERMINATE_"), \
            f"Expected GERMINATE, got {decision.action.name}: {decision.reason}"

    @given(seed=probationary_seed_states(with_counterfactual=True))
    @settings(max_examples=50)
    def test_fossilize_reachable(self, seed):
        """Property: FOSSILIZE is reachable for contributing HOLDING seeds."""
        policy = HeuristicTamiyo(topology="cnn")

        # Ensure positive contribution AND positive total_improvement
        # (to avoid triggering ransomware detection - P2-B)
        seed.metrics.counterfactual_contribution = 5.0
        seed.metrics.total_improvement = 3.0  # Positive to avoid ransomware pattern

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        assert decision.action.name == "FOSSILIZE", \
            f"Expected FOSSILIZE, got {decision.action.name}: {decision.reason}"

    @given(seed=failing_seed_states())
    @settings(max_examples=50)
    def test_cull_reachable(self, seed):
        """Property: CULL is reachable for failing seeds."""
        # Use strict config that culls quickly
        config = HeuristicPolicyConfig(
            cull_after_epochs_without_improvement=1,
            cull_if_accuracy_drops_by=0.1,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Ensure seed is failing hard and long enough
        seed.metrics.improvement_since_stage_start = -10.0
        seed.epochs_in_stage = 10

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = -5.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        assert decision.action.name == "CULL", \
            f"Expected CULL, got {decision.action.name}: {decision.reason}"

    @given(signals=early_training_signals(max_epoch=2))
    @settings(max_examples=50)
    def test_wait_reachable(self, signals):
        """Property: WAIT is reachable (default when no criteria met)."""
        # Use lenient config that rarely acts
        config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=100,
            min_epochs_before_germinate=100,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT", \
            f"Expected WAIT, got {decision.action.name}: {decision.reason}"


# =============================================================================
# Property: Target Consistency
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestTargetConsistency:
    """Decisions that act on seeds always specify targets."""

    @given(seed=probationary_seed_states(with_counterfactual=True))
    @settings(max_examples=100)
    def test_fossilize_has_target(self, seed):
        """Property: FOSSILIZE decisions always have target_seed_id."""
        policy = HeuristicTamiyo(topology="cnn")

        # Force positive contribution
        seed.metrics.counterfactual_contribution = 5.0

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        if decision.action.name == "FOSSILIZE":
            assert decision.target_seed_id is not None, \
                "FOSSILIZE must have target_seed_id"
            assert decision.target_seed_id == seed.seed_id

    @given(seed=failing_seed_states())
    @settings(max_examples=100)
    def test_cull_has_target(self, seed):
        """Property: CULL decisions always have target_seed_id."""
        config = HeuristicPolicyConfig(
            cull_after_epochs_without_improvement=1,
            cull_if_accuracy_drops_by=0.1,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed.metrics.improvement_since_stage_start = -10.0
        seed.epochs_in_stage = 10

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = -5.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        if decision.action.name == "CULL":
            assert decision.target_seed_id is not None, \
                "CULL must have target_seed_id"
            assert decision.target_seed_id == seed.seed_id

    @given(signals=plateau_signals(min_plateau=5))
    @settings(max_examples=50)
    def test_germinate_no_target(self, signals):
        """Property: GERMINATE decisions don't need target_seed_id (creating new)."""
        config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=3,
            min_epochs_before_germinate=0,
            embargo_epochs_after_cull=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        signals.metrics.host_stabilized = 1
        signals.metrics.epoch = 10

        decision = policy.decide(signals, active_seeds=[])

        if decision.action.name.startswith("GERMINATE_"):
            # target_seed_id can be None for germination (new seed)
            # This is acceptable - the seed doesn't exist yet
            pass  # No assertion needed, just verifying it doesn't crash


# =============================================================================
# Property: WAIT as Default
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestWaitDefault:
    """WAIT is the default when no action criteria are met."""

    @given(seed=succeeding_seed_states())
    @settings(max_examples=100)
    def test_wait_for_healthy_seed(self, seed):
        """Property: Healthy seeds in non-terminal stages get WAIT."""
        policy = HeuristicTamiyo(topology="cnn")

        # Ensure seed is healthy but not ready to fossilize
        if seed.stage == SeedStage.HOLDING:
            seed.stage = SeedStage.BLENDING  # Move to earlier stage

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 1.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        assert decision.action.name == "WAIT", \
            f"Expected WAIT for healthy seed, got {decision.action.name}"

    @given(signals=unstabilized_signals())
    @settings(max_examples=100)
    def test_wait_when_unstabilized(self, signals):
        """Property: No germination when host not stabilized."""
        config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=1,
            min_epochs_before_germinate=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Even with high plateau, should WAIT if not stabilized
        signals.metrics.plateau_epochs = 100
        signals.metrics.epoch = 100

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT", \
            f"Expected WAIT when unstabilized, got {decision.action.name}"
        assert "stabil" in decision.reason.lower()


# =============================================================================
# Property: Stage Appropriateness
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestStageAppropriateness:
    """Actions are only taken from appropriate seed stages."""

    @given(seed=mock_seed_states_at_stage(SeedStage.GERMINATED))
    @settings(max_examples=50)
    def test_germinated_seed_waits(self, seed):
        """Property: GERMINATED seeds always get WAIT (awaiting auto-advance)."""
        policy = HeuristicTamiyo(topology="cnn")

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        assert decision.action.name == "WAIT", \
            f"GERMINATED should WAIT, got {decision.action.name}"
        assert "auto-advance" in decision.reason.lower() or "await" in decision.reason.lower()

    @given(seed=mock_seed_states_at_stage(SeedStage.TRAINING))
    @settings(max_examples=50)
    def test_training_seed_cannot_fossilize(self, seed):
        """Property: TRAINING seeds cannot be fossilized directly."""
        policy = HeuristicTamiyo(topology="cnn")

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        # TRAINING seeds should either WAIT or CULL (if failing), never FOSSILIZE
        assert decision.action.name != "FOSSILIZE", \
            "TRAINING seeds cannot be fossilized directly"

    @given(seed=mock_seed_states_at_stage(SeedStage.BLENDING))
    @settings(max_examples=50)
    def test_blending_seed_cannot_fossilize(self, seed):
        """Property: BLENDING seeds cannot be fossilized directly."""
        policy = HeuristicTamiyo(topology="cnn")

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        # BLENDING seeds should WAIT or CULL, never FOSSILIZE
        assert decision.action.name != "FOSSILIZE", \
            "BLENDING seeds cannot be fossilized directly"


# =============================================================================
# Property: Reason Non-Empty
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestReasonProvided:
    """Every decision includes a reason explaining why."""

    @given(
        signals=mock_training_signals(),
        seeds=st.lists(mock_seed_states(), max_size=2),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_decision_has_reason(self, signals, seeds):
        """Property: All decisions have non-empty reason strings."""
        policy = HeuristicTamiyo(topology="cnn")

        decision = policy.decide(signals, seeds)

        assert decision.reason, "Decision must have a reason"
        assert len(decision.reason) > 0, "Reason must be non-empty"
        assert isinstance(decision.reason, str), "Reason must be string"


# =============================================================================
# Property: Confidence Bounds
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestConfidenceBounds:
    """Confidence values are always in valid range."""

    @given(
        signals=mock_training_signals(),
        seeds=st.lists(mock_seed_states(), max_size=2),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_confidence_in_range(self, signals, seeds):
        """Property: Confidence is always in [0, 1]."""
        policy = HeuristicTamiyo(topology="cnn")

        decision = policy.decide(signals, seeds)

        assert 0.0 <= decision.confidence <= 1.0, \
            f"Confidence {decision.confidence} out of range [0, 1]"


# =============================================================================
# Property: Decision History Integrity
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestDecisionHistory:
    """Policy maintains accurate decision history."""

    @given(
        num_decisions=st.integers(min_value=1, max_value=10),
        signals=mock_training_signals(),
    )
    @settings(max_examples=50)
    def test_history_length_matches_calls(self, num_decisions, signals):
        """Property: decisions property length equals number of decide() calls."""
        policy = HeuristicTamiyo(topology="cnn")

        for i in range(num_decisions):
            # Vary epoch to avoid exact duplicate detection issues
            signals.metrics.epoch = i
            policy.decide(signals, active_seeds=[])

        assert len(policy.decisions) == num_decisions, \
            f"Expected {num_decisions} decisions, got {len(policy.decisions)}"

    @given(signals=mock_training_signals())
    @settings(max_examples=50)
    def test_history_returns_copy(self, signals):
        """Property: decisions property returns copy, not internal list."""
        policy = HeuristicTamiyo(topology="cnn")
        policy.decide(signals, active_seeds=[])

        history1 = policy.decisions
        history2 = policy.decisions

        # Should be equal but not the same object
        assert history1 == history2
        assert history1 is not history2, "decisions should return copy"

        # Modifying returned list shouldn't affect internal state
        history1.clear()
        assert len(policy.decisions) == 1, "Internal history should be unchanged"


def test_fossilize_requires_meaningful_improvement():
    """Fossilization should require meaningful improvement, not just any positive value."""
    from esper.tamiyo.heuristic import HeuristicPolicyConfig
    from esper.leyline import DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE

    # Default threshold should be meaningful (at least 0.5%)
    assert DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE >= 0.5, (
        f"Default fossilize threshold {DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE} is too low. "
        "Seeds with negligible improvement can be fossilized, enabling reward hacking."
    )

    # HeuristicPolicyConfig should use the default
    config = HeuristicPolicyConfig()
    assert config.min_improvement_to_fossilize >= 0.5
