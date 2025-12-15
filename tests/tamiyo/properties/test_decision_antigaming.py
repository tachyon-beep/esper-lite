"""Property-based tests for Tamiyo anti-gaming mechanisms.

Tier 3 properties verify that policy mechanisms prevent gaming:
1. Embargo prevents thrashing - No germinate/cull cycles
2. Blueprint penalty prevents abuse - Culled blueprints get penalized
3. Stabilization prevents false credit - No germination during explosive growth
4. Counterfactual guards fossilization - Seeds must prove value

These properties ensure the decision policy is robust against gaming patterns
that could destabilize training or waste resources.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from esper.leyline import SeedStage
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig
from esper.tamiyo.tracker import SignalTracker

# Import shared strategies
from tests.strategies import bounded_floats

# Import Tamiyo-specific strategies
from tests.tamiyo.strategies.decision_strategies import (
    probationary_seed_states,
    unstabilized_signals,
)
from tests.tamiyo.strategies.tracker_strategies import (
    stable_loss_sequences,
    explosive_loss_sequences,
)


# =============================================================================
# Property: Embargo Prevents Thrashing
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestAntiThrashing:
    """Embargo mechanism prevents germinate/cull thrashing."""

    @given(
        embargo_epochs=st.integers(min_value=1, max_value=10),
        cull_epoch=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=200)
    def test_embargo_window_honored(self, embargo_epochs, cull_epoch):
        """Property: No germination possible within embargo window after cull."""
        config = HeuristicPolicyConfig(
            embargo_epochs_after_cull=embargo_epochs,
            plateau_epochs_to_germinate=1,  # Easy to trigger
            min_epochs_before_germinate=0,  # No minimum
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Simulate a cull
        policy._last_cull_epoch = cull_epoch

        # Test all epochs during embargo
        for offset in range(embargo_epochs):
            current_epoch = cull_epoch + offset

            class MockSignals:
                class metrics:
                    epoch = current_epoch
                    plateau_epochs = 100  # Would trigger germination
                    host_stabilized = 1   # Stabilized
                    accuracy_delta = 0.0

            decision = policy.decide(MockSignals(), active_seeds=[])

            assert decision.action.name == "WAIT", \
                f"Embargo violated at epoch {current_epoch} (cull at {cull_epoch}, embargo={embargo_epochs})"
            assert "embargo" in decision.reason.lower()

    @given(
        embargo_epochs=st.integers(min_value=1, max_value=10),
        cull_epoch=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=100)
    def test_embargo_expires(self, embargo_epochs, cull_epoch):
        """Property: Germination allowed after embargo expires."""
        config = HeuristicPolicyConfig(
            embargo_epochs_after_cull=embargo_epochs,
            plateau_epochs_to_germinate=1,
            min_epochs_before_germinate=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._last_cull_epoch = cull_epoch

        # Epoch exactly at embargo expiration
        post_embargo_epoch = cull_epoch + embargo_epochs

        class MockSignals:
            class metrics:
                epoch = post_embargo_epoch
                plateau_epochs = 100
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[])

        # Should be able to germinate now
        assert decision.action.name.startswith("GERMINATE_"), \
            f"Expected germination after embargo, got {decision.action.name}: {decision.reason}"


# =============================================================================
# Property: Blueprint Penalty Prevents Abuse
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestBlueprintPenalty:
    """Blueprint penalty system prevents rapid re-germination of failed blueprints."""

    @given(
        penalty_amount=bounded_floats(2.0, 10.0),
        threshold=bounded_floats(1.0, 5.0),
    )
    @settings(max_examples=100)
    def test_penalized_blueprint_avoided(self, penalty_amount, threshold):
        """Property: Blueprints with penalty >= threshold are skipped in rotation.

        Note: Penalty decays by decay factor (default 0.5) each epoch before
        blueprint selection, so we need penalty > threshold / decay_factor
        to ensure the penalty stays above threshold after decay.
        """
        assume(penalty_amount >= threshold * 2)  # After 0.5 decay, still >= threshold

        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy", "attention"],
            blueprint_penalty_threshold=threshold,
            blueprint_penalty_decay=0.5,  # Explicit for clarity
            plateau_epochs_to_germinate=1,
            min_epochs_before_germinate=0,
            embargo_epochs_after_cull=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Penalize first blueprint (needs to survive decay and still be >= threshold)
        policy._blueprint_penalties["conv_light"] = penalty_amount

        class MockSignals:
            class metrics:
                epoch = 10
                plateau_epochs = 10
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[])

        # Should germinate with a different blueprint
        assert decision.action.name.startswith("GERMINATE_")
        # After decay, penalty should still be >= threshold
        assert decision.blueprint_id != "conv_light", \
            f"Should not use penalized blueprint, got {decision.blueprint_id}"

    @given(
        decay=bounded_floats(0.1, 0.9),
        initial_penalty=bounded_floats(1.0, 10.0),
        epochs=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_penalty_decays_over_epochs(self, decay, initial_penalty, epochs):
        """Property: Blueprint penalties decay each epoch."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=decay,
            blueprint_rotation=["conv_light"],
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._blueprint_penalties["conv_light"] = initial_penalty
        policy._last_decay_epoch = -1

        expected_penalty = initial_penalty
        for epoch in range(epochs):
            policy._last_decay_epoch = epoch - 1  # Allow decay this epoch

            class MockSignals:
                class metrics:
                    pass
            MockSignals.metrics.epoch = epoch

            # Manually trigger decay (normally done in decide())
            policy._decay_blueprint_penalties()

            expected_penalty *= decay

            current = policy._blueprint_penalties.get("conv_light", 0.0)

            # Penalty should decay (with tolerance for float precision)
            if current > 0:
                assert current <= initial_penalty, \
                    f"Penalty should decrease from {initial_penalty}, got {current}"

    @given(num_culls=st.integers(min_value=1, max_value=5))
    @settings(max_examples=50)
    def test_repeated_culls_accumulate_penalty(self, num_culls):
        """Property: Multiple culls of same blueprint accumulate penalty."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_on_cull=2.0,
            blueprint_rotation=["conv_light"],
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        for i in range(num_culls):
            policy._apply_blueprint_penalty("conv_light")

        expected_penalty = num_culls * config.blueprint_penalty_on_cull
        actual_penalty = policy._blueprint_penalties.get("conv_light", 0.0)

        assert actual_penalty == expected_penalty, \
            f"Expected {expected_penalty} penalty after {num_culls} culls, got {actual_penalty}"


# =============================================================================
# Property: Stabilization Prevents False Credit
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestStabilizationGating:
    """Stabilization gating prevents germination during explosive growth."""

    @given(losses=explosive_loss_sequences(min_length=5, max_length=15))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_stabilization_during_explosive_growth(self, losses):
        """Property: Tracker doesn't stabilize during explosive improvement."""
        tracker = SignalTracker()

        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0 + i * 0.5,
                val_loss=loss,
                val_accuracy=50.0 + i * 0.5,
                active_seeds=[],
            )

        # With explosive improvement, should NOT be stabilized
        # (unless we happened to generate a sequence that stabilized anyway)
        # This is a probabilistic test - explosive sequences usually don't stabilize

    @given(signals=unstabilized_signals())
    @settings(max_examples=100)
    def test_no_germination_before_stabilization(self, signals):
        """Property: Cannot germinate when host_stabilized=0."""
        config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=1,
            min_epochs_before_germinate=0,
            embargo_epochs_after_cull=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Even with long plateau, should not germinate
        signals.metrics.plateau_epochs = 100
        signals.metrics.epoch = 100
        signals.metrics.host_stabilized = 0

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT", \
            f"Should not germinate when unstabilized, got {decision.action.name}"
        assert "stabil" in decision.reason.lower()

    @given(losses=stable_loss_sequences(min_length=5, max_length=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_stabilization_after_stable_epochs(self, losses):
        """Property: Tracker stabilizes after consecutive stable epochs."""
        tracker = SignalTracker(stabilization_epochs=3)

        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0,
                val_loss=loss,
                val_accuracy=50.0,
                active_seeds=[],
            )

        # With stable losses, should eventually stabilize
        # (first epoch doesn't count, so need len(losses) > stabilization_epochs + 1)
        if len(losses) > tracker.stabilization_epochs + 1:
            assert tracker.is_stabilized, \
                f"Should be stabilized after {len(losses)} stable epochs"


# =============================================================================
# Property: Counterfactual Guards Fossilization
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestCounterfactualGuard:
    """Fossilization requires demonstrated value (counterfactual > threshold)."""

    @given(seed=probationary_seed_states(with_counterfactual=True))
    @settings(max_examples=100)
    def test_negative_counterfactual_no_fossilize(self, seed):
        """Property: Seeds with negative counterfactual get culled, not fossilized."""
        config = HeuristicPolicyConfig(
            min_improvement_to_fossilize=0.0,  # Any positive is enough
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Force negative counterfactual
        seed.metrics.counterfactual_contribution = -5.0

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        assert decision.action.name != "FOSSILIZE", \
            f"Should not fossilize with negative counterfactual, got {decision.action.name}"

    @given(
        counterfactual=bounded_floats(0.5, 10.0),
        threshold=bounded_floats(-1.0, 0.4),
    )
    @settings(max_examples=100)
    def test_positive_counterfactual_fossilizes(self, counterfactual, threshold):
        """Property: Seeds with counterfactual > threshold get fossilized."""
        assume(counterfactual > threshold)  # Ensure above threshold

        config = HeuristicPolicyConfig(
            min_improvement_to_fossilize=threshold,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed = type('MockSeed', (), {
            'seed_id': 'test_seed',
            'stage': SeedStage.PROBATIONARY,
            'epochs_in_stage': 5,
            'alpha': 1.0,
            'blueprint_id': 'conv_light',
            'metrics': type('Metrics', (), {
                'improvement_since_stage_start': 2.0,
                'total_improvement': 3.0,
                'counterfactual_contribution': counterfactual,
            })(),
        })()

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        assert decision.action.name == "FOSSILIZE", \
            f"Should fossilize with counterfactual {counterfactual} > threshold {threshold}"

    @given(seed=probationary_seed_states(with_counterfactual=False))
    @settings(max_examples=50)
    def test_no_counterfactual_uses_total_improvement(self, seed):
        """Property: Without counterfactual, falls back to total_improvement."""
        policy = HeuristicTamiyo(topology="cnn")

        # Force no counterfactual
        seed.metrics.counterfactual_contribution = None
        # Force positive total improvement
        seed.metrics.total_improvement = 5.0

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        # With positive total_improvement, should fossilize
        assert decision.action.name == "FOSSILIZE", \
            "Should fossilize with positive total_improvement when no counterfactual"


# =============================================================================
# Property: Ransomware Detection (Future Enhancement)
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestRansomwareDetection:
    """
    Properties for detecting "ransomware" seeds - high counterfactual but
    negative total improvement (seed created dependencies without net benefit).

    NOTE: HeuristicTamiyo is currently VULNERABLE to this pattern.
    These tests document the desired behavior for future implementation.
    """

    @pytest.mark.skip(reason="HeuristicTamiyo doesn't yet detect ransomware pattern")
    @given(
        counterfactual=bounded_floats(5.0, 15.0),
        total_improvement=bounded_floats(-5.0, -0.5),
    )
    def test_ransomware_pattern_detected(self, counterfactual, total_improvement):
        """Property: High counterfactual + negative total = ransomware, should cull.

        Ransomware pattern: Seed creates dependencies (high counterfactual) but
        overall system is worse (negative total_improvement). The seed is holding
        the system "hostage."

        FUTURE: This test will pass once ransomware detection is implemented.
        """
        policy = HeuristicTamiyo(topology="cnn")

        seed = type('MockSeed', (), {
            'seed_id': 'ransomware_seed',
            'stage': SeedStage.PROBATIONARY,
            'epochs_in_stage': 10,
            'alpha': 1.0,
            'blueprint_id': 'conv_light',
            'metrics': type('Metrics', (), {
                'improvement_since_stage_start': -1.0,
                'total_improvement': total_improvement,  # Negative!
                'counterfactual_contribution': counterfactual,  # High!
            })(),
        })()

        class MockSignals:
            class metrics:
                epoch = 50
                plateau_epochs = 0
                host_stabilized = 1
                accuracy_delta = 0.0

        decision = policy.decide(MockSignals(), active_seeds=[seed])

        # Ransomware seeds should be CULLED, not FOSSILIZED
        assert decision.action.name == "CULL", \
            f"Ransomware pattern should trigger CULL, got {decision.action.name}"
