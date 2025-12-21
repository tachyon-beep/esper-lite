"""Property-based tests for Tamiyo strategic controller.

These tests use Hypothesis to generate hundreds of random but valid inputs,
finding edge cases that are "impossible but happen regularly."

Properties tested:
1. Stabilization latch monotonicity - once True, stays True
2. SignalTracker invariants - best_accuracy, history bounds, plateau_count
3. Decision idempotence - same inputs → same outputs
4. Blueprint penalty decay monotonicity
5. Embargo enforcement invariant
6. Stateful decision sequences

Usage:
    pytest tests/tamiyo/properties/ -v --hypothesis-show-statistics
    HYPOTHESIS_PROFILE=tamiyo_ci pytest tests/tamiyo/properties/  # More examples in CI
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize

from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig

# Import shared strategies from tests.strategies
from tests.strategies import bounded_floats

# Import Tamiyo-specific strategies from the new strategies module
from tests.tamiyo.strategies import (
    mock_seed_states,
    mock_training_signals,
    loss_sequences,
)


# =============================================================================
# Property 1: Stabilization Latch Monotonicity
# =============================================================================

class TestStabilizationLatchProperties:
    """Property: Once is_stabilized=True, it MUST stay True (until reset)."""

    @given(loss_sequences(min_length=5, max_length=30))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_latch_monotonicity(self, losses):
        """Property: is_stabilized can transition False→True but never True→False."""
        tracker = SignalTracker()

        was_stabilized = False
        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0 + i * 0.1,
                val_loss=loss,
                val_accuracy=50.0 + i * 0.1,
                active_seeds=[],
            )

            current = tracker.is_stabilized

            # Monotonicity: if was True, must still be True
            if was_stabilized:
                assert current, f"Latch violation at epoch {i}: was True, now False"

            was_stabilized = current

    @given(st.integers(min_value=1, max_value=10))
    def test_stabilization_requires_consecutive_epochs(self, required_epochs):
        """Property: Stabilization requires exactly stabilization_epochs consecutive stable epochs.

        Note: First epoch can't count toward stabilization (no previous loss to compare).
        So we need required_epochs + 1 total epochs to trigger stabilization.

        Stable = < 3% relative improvement per epoch.
        """
        tracker = SignalTracker(stabilization_epochs=required_epochs)

        # Epoch 0: Establish baseline (can't count toward stabilization)
        baseline_loss = 1.0
        tracker.update(
            epoch=0, global_step=0,
            train_loss=baseline_loss, train_accuracy=50.0,
            val_loss=baseline_loss, val_accuracy=50.0,
            active_seeds=[],
        )
        assert not tracker.is_stabilized, "Should not stabilize on first epoch"

        # Feed required_epochs stable epochs (2% improvement each - below 3% threshold)
        current_loss = baseline_loss
        for i in range(required_epochs):
            # 2% improvement (well below 3% threshold)
            current_loss = current_loss * 0.98
            tracker.update(
                epoch=i + 1, global_step=(i + 1) * 100,
                train_loss=current_loss,
                train_accuracy=50.0,
                val_loss=current_loss,
                val_accuracy=50.0,
                active_seeds=[],
            )

            # Check: should stabilize exactly when we've had required_epochs stable epochs
            if i < required_epochs - 1:
                assert not tracker.is_stabilized, f"Premature stabilization at epoch {i + 1}"

        # After required_epochs stable epochs, should be stabilized
        assert tracker.is_stabilized, f"Should be stabilized after {required_epochs} stable epochs"


# =============================================================================
# Property 2: SignalTracker Invariants
# =============================================================================

class TestSignalTrackerInvariants:
    """Invariants that must hold for SignalTracker at all times."""

    @given(loss_sequences(min_length=3, max_length=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_best_accuracy_invariant(self, losses):
        """Property: best_accuracy >= any accuracy seen so far."""
        tracker = SignalTracker()

        max_seen = 0.0
        for i, loss in enumerate(losses):
            accuracy = 100.0 - loss * 20  # Derived accuracy
            max_seen = max(max_seen, accuracy)

            signals = tracker.update(
                epoch=i, global_step=i * 100,
                train_loss=loss, train_accuracy=accuracy,
                val_loss=loss, val_accuracy=accuracy,
                active_seeds=[],
            )

            assert signals.metrics.best_val_accuracy >= accuracy, \
                f"best_val_accuracy ({signals.metrics.best_val_accuracy}) < current ({accuracy})"

    @given(
        st.integers(min_value=1, max_value=20),
        loss_sequences(min_length=5, max_length=50)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_history_window_invariant(self, window_size, losses):
        """Property: History length never exceeds history_window."""
        tracker = SignalTracker(history_window=window_size)

        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i, global_step=i * 100,
                train_loss=loss, train_accuracy=50.0,
                val_loss=loss, val_accuracy=50.0,
                active_seeds=[],
            )

            assert len(tracker._loss_history) <= window_size, \
                f"Loss history {len(tracker._loss_history)} > window {window_size}"
            assert len(tracker._accuracy_history) <= window_size, \
                f"Accuracy history {len(tracker._accuracy_history)} > window {window_size}"

    @given(loss_sequences(min_length=5, max_length=30))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_plateau_count_non_negative(self, losses):
        """Property: plateau_count is always >= 0."""
        tracker = SignalTracker()

        for i, loss in enumerate(losses):
            signals = tracker.update(
                epoch=i, global_step=i * 100,
                train_loss=loss, train_accuracy=50.0 + i * 0.5,
                val_loss=loss, val_accuracy=50.0 + i * 0.5,
                active_seeds=[],
            )

            assert signals.metrics.plateau_epochs >= 0, \
                f"Negative plateau count: {signals.metrics.plateau_epochs}"


# =============================================================================
# Property 3: Decision Idempotence
# =============================================================================

class TestDecisionIdempotence:
    """Property: Same inputs produce same outputs (determinism)."""

    @given(mock_training_signals(), st.lists(mock_seed_states(), max_size=3))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_decide_is_deterministic(self, signals, seeds):
        """Property: decide(s, seeds) == decide(s, seeds) for identical inputs."""
        policy = HeuristicTamiyo(topology="cnn")

        # Make two identical calls
        decision1 = policy.decide(signals, seeds)

        # Reset to same state
        policy.reset()

        decision2 = policy.decide(signals, seeds)

        # Decisions should be identical
        assert decision1.action == decision2.action, \
            f"Non-deterministic action: {decision1.action} vs {decision2.action}"
        assert decision1.target_seed_id == decision2.target_seed_id, \
            f"Non-deterministic target: {decision1.target_seed_id} vs {decision2.target_seed_id}"


# =============================================================================
# Property 4: Blueprint Penalty Decay Monotonicity
# =============================================================================

class TestBlueprintPenaltyProperties:
    """Property: Penalty decay is monotonically decreasing."""

    @given(
        bounded_floats(0.1, 10.0),  # initial penalty
        bounded_floats(0.1, 0.9),   # decay factor
        st.integers(min_value=1, max_value=20),  # epochs
    )
    def test_penalty_decay_monotonic(self, initial_penalty, decay, epochs):
        """Property: penalty_t+1 <= penalty_t after each decay."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=decay,
            blueprint_rotation=["conv_light"],
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Set initial penalty
        policy._blueprint_penalties["conv_light"] = initial_penalty

        prev_penalty = initial_penalty
        for epoch in range(epochs):
            policy._last_decay_epoch = epoch - 1  # Allow decay
            policy._decay_blueprint_penalties()

            current = policy._blueprint_penalties.get("conv_light", 0.0)

            assert current <= prev_penalty + 1e-9, \
                f"Penalty increased: {prev_penalty} -> {current}"

            prev_penalty = current

    @given(bounded_floats(0.1, 10.0), st.integers(min_value=10, max_value=50))
    def test_penalty_eventually_clears(self, initial_penalty, decay_cycles):
        """Property: After enough decays, penalty drops below threshold."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=0.5,  # Halves each time
            blueprint_rotation=["conv_light"],
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._blueprint_penalties["conv_light"] = initial_penalty

        for i in range(decay_cycles):
            policy._last_decay_epoch = i - 1
            policy._decay_blueprint_penalties()

        # After many decays, penalty should be gone (< 0.1 threshold)
        assert "conv_light" not in policy._blueprint_penalties or \
               policy._blueprint_penalties["conv_light"] < 0.1


# =============================================================================
# Property 6: Embargo Enforcement Invariant
# =============================================================================

class TestEmbargoEnforcementProperties:
    """Property: Germination is impossible during embargo period."""

    @given(
        st.integers(min_value=1, max_value=10),  # embargo_epochs
        st.integers(min_value=0, max_value=100),  # cull_epoch
    )
    def test_embargo_blocks_germination(self, embargo_epochs, cull_epoch):
        """Property: If epoch - last_cull_epoch < embargo, decision MUST be WAIT."""
        config = HeuristicPolicyConfig(
            embargo_epochs_after_prune=embargo_epochs,
            plateau_epochs_to_germinate=1,  # Easy to trigger
            min_epochs_before_germinate=0,  # No minimum
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Set cull epoch
        policy._last_prune_epoch = cull_epoch

        # Test epochs during embargo
        for offset in range(embargo_epochs):
            current_epoch = cull_epoch + offset

            class MockMetrics:
                epoch = current_epoch
                plateau_epochs = 10  # Would normally trigger germination
                host_stabilized = 1  # Stabilized

            class MockSignals:
                metrics = MockMetrics()

            decision = policy.decide(MockSignals(), active_seeds=[])

            assert decision.action.name == "WAIT", \
                f"Embargo violated at epoch {current_epoch} (cull was {cull_epoch})"
            assert "Embargo" in decision.reason


# =============================================================================
# Property 7: Stateful Machine Testing
# =============================================================================

class TamiyoStateMachine(RuleBasedStateMachine):
    """Stateful test: Random decision sequences don't violate invariants."""

    def __init__(self):
        super().__init__()
        self.policy = None
        self.tracker = None
        self.epoch = 0
        self.decisions = []

    @initialize()
    def setup(self):
        self.policy = HeuristicTamiyo(topology="cnn")
        self.tracker = SignalTracker()
        self.epoch = 0
        self.decisions = []

    @rule(
        loss=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        accuracy=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def step_epoch(self, loss, accuracy):
        """Simulate one training epoch."""
        signals = self.tracker.update(
            epoch=self.epoch,
            global_step=self.epoch * 100,
            train_loss=loss,
            train_accuracy=accuracy,
            val_loss=loss,
            val_accuracy=accuracy,
            active_seeds=[],
        )

        decision = self.policy.decide(signals, active_seeds=[])
        self.decisions.append(decision)
        self.epoch += 1

    @rule()
    def reset_policy(self):
        """Reset policy state."""
        self.policy.reset()

    @invariant()
    def policy_state_valid(self):
        """Invariant: Policy internal state is always valid."""
        assert self.policy._blueprint_index >= 0
        assert self.policy._germination_count >= 0
        assert len(self.policy._blueprint_penalties) >= 0
        for penalty in self.policy._blueprint_penalties.values():
            assert penalty >= 0, f"Negative penalty: {penalty}"

    @invariant()
    def tracker_state_valid(self):
        """Invariant: Tracker internal state is always valid."""
        assert self.tracker._plateau_count >= 0
        assert self.tracker._stable_count >= 0
        assert len(self.tracker._loss_history) <= self.tracker.history_window
        assert len(self.tracker._accuracy_history) <= self.tracker.history_window

    @invariant()
    def decisions_have_valid_actions(self):
        """Invariant: All decisions have valid action types."""
        for decision in self.decisions:
            action_name = decision.action.name
            valid = (
                action_name == "WAIT" or
                action_name == "FOSSILIZE" or
                action_name == "PRUNE" or
                action_name.startswith("GERMINATE_")
            )
            assert valid, f"Invalid action: {action_name}"


# Run the stateful test
TestTamiyoSequences = TamiyoStateMachine.TestCase


# =============================================================================
# Regression Properties (from known bugs)
# =============================================================================

class TestRegressionProperties:
    """Properties derived from past bugs."""

    @given(st.integers(min_value=0, max_value=100))
    def test_first_epoch_no_crash(self, epoch):
        """Property: First update never crashes regardless of epoch number."""
        tracker = SignalTracker()

        # First update with any epoch should work
        signals = tracker.update(
            epoch=epoch,
            global_step=epoch * 100,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.0,
            val_accuracy=50.0,
            active_seeds=[],
        )

        assert signals is not None
        assert signals.metrics.epoch == epoch

    @given(bounded_floats(0.0, 100.0))
    def test_zero_loss_no_division_error(self, accuracy):
        """Property: Zero or near-zero loss doesn't cause division by zero."""
        tracker = SignalTracker()

        # Set up previous loss
        tracker._prev_loss = 0.0001  # Near zero

        # Should not raise ZeroDivisionError
        signals = tracker.update(
            epoch=1,
            global_step=100,
            train_loss=0.0001,
            train_accuracy=accuracy,
            val_loss=0.0001,
            val_accuracy=accuracy,
            active_seeds=[],
        )

        assert signals is not None
