"""Property-based stateful tests for Tamiyo components.

Tier 4 properties use Hypothesis state machines to verify:
1. Policy state consistency - Internal state stays valid through any sequence
2. Tracker state consistency - History, counts, latch remain consistent
3. Reset completeness - reset() fully clears all state
4. Cross-component interactions - Policy + Tracker work together correctly

These tests generate random sequences of operations and verify invariants
hold at every step, catching edge cases that single-input tests miss.
"""

from __future__ import annotations

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    invariant,
    initialize,
)

from esper.kasmina.alpha_controller import AlphaController
from esper.leyline import SeedStage
from esper.leyline.alpha import AlphaMode
from esper.tamiyo.heuristic import HeuristicTamiyo
from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.decisions import TamiyoDecision

# Import strategies


# =============================================================================
# State Machine: HeuristicTamiyo Policy
# =============================================================================

class HeuristicPolicyStateMachine(RuleBasedStateMachine):
    """Stateful test for HeuristicTamiyo through random operation sequences.

    This state machine:
    - Makes decisions with various inputs
    - Tracks germinations and culls
    - Resets state periodically
    - Verifies invariants after every operation
    """

    def __init__(self):
        super().__init__()
        self.policy = None
        self.epoch = 0
        self.decision_count = 0
        self.germination_count = 0
        self.cull_count = 0
        self.seeds_active = []

    @initialize()
    def setup(self):
        """Initialize fresh policy."""
        self.policy = HeuristicTamiyo(topology="cnn")
        self.epoch = 0
        self.decision_count = 0
        self.germination_count = 0
        self.cull_count = 0
        self.seeds_active = []

    @rule(
        plateau=st.integers(min_value=0, max_value=20),
        stabilized=st.booleans(),
    )
    def make_decision_no_seeds(self, plateau, stabilized):
        """Make decision with no active seeds."""
        class MockSignals:
            class metrics:
                pass

        MockSignals.metrics.epoch = self.epoch
        MockSignals.metrics.plateau_epochs = plateau
        MockSignals.metrics.host_stabilized = 1 if stabilized else 0
        MockSignals.metrics.accuracy_delta = 0.0

        decision = self.policy.decide(MockSignals(), active_seeds=[])
        self.decision_count += 1
        self.epoch += 1

        if decision.action.name.startswith("GERMINATE_"):
            self.germination_count += 1

    @rule(
        stage=st.sampled_from([SeedStage.TRAINING, SeedStage.BLENDING]),
        improvement=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        epochs_in_stage=st.integers(min_value=0, max_value=20),
    )
    def make_decision_with_seed(self, stage, improvement, epochs_in_stage):
        """Make decision with an active seed."""
        alpha = 0.5 if stage == SeedStage.BLENDING else 0.0
        alpha_controller = AlphaController(alpha=alpha)
        alpha_controller.alpha_target = 1.0
        alpha_controller.alpha_mode = AlphaMode.UP

        seed = type('MockSeed', (), {
            'seed_id': f'seed_{self.epoch}',
            'stage': stage,
            'epochs_in_stage': epochs_in_stage,
            'alpha': alpha,
            'blueprint_id': 'conv_light',
            'alpha_controller': alpha_controller,
            'metrics': type('Metrics', (), {
                'improvement_since_stage_start': improvement,
                'total_improvement': improvement,
                'counterfactual_contribution': None,
            })(),
        })()

        class MockSignals:
            class metrics:
                pass

        MockSignals.metrics.epoch = self.epoch
        MockSignals.metrics.plateau_epochs = 0
        MockSignals.metrics.host_stabilized = 1
        MockSignals.metrics.accuracy_delta = 0.0

        decision = self.policy.decide(MockSignals(), active_seeds=[seed])
        self.decision_count += 1
        self.epoch += 1

        if decision.action.name == "PRUNE":
            self.cull_count += 1
            self.policy._last_prune_epoch = self.epoch

    @rule(
        counterfactual=st.floats(min_value=-5.0, max_value=10.0, allow_nan=False),
    )
    def make_decision_probationary(self, counterfactual):
        """Make decision with HOLDING seed."""
        alpha_controller = AlphaController(alpha=1.0)
        alpha_controller.alpha_target = 1.0
        alpha_controller.alpha_mode = AlphaMode.HOLD

        seed = type('MockSeed', (), {
            'seed_id': f'seed_{self.epoch}',
            'stage': SeedStage.HOLDING,
            'epochs_in_stage': 5,
            'alpha': 1.0,
            'blueprint_id': 'conv_light',
            'alpha_controller': alpha_controller,
            'metrics': type('Metrics', (), {
                'improvement_since_stage_start': counterfactual,
                'total_improvement': counterfactual,
                'counterfactual_contribution': counterfactual,
            })(),
        })()

        class MockSignals:
            class metrics:
                pass

        MockSignals.metrics.epoch = self.epoch
        MockSignals.metrics.plateau_epochs = 0
        MockSignals.metrics.host_stabilized = 1
        MockSignals.metrics.accuracy_delta = 0.0

        decision = self.policy.decide(MockSignals(), active_seeds=[seed])
        self.decision_count += 1
        self.epoch += 1

        if decision.action.name == "PRUNE":
            self.cull_count += 1

    @rule()
    def reset_policy(self):
        """Reset policy state."""
        self.policy.reset()
        self.decision_count = 0
        # Note: epoch continues to advance (external time)

    # =========================================================================
    # Invariants
    # =========================================================================

    @invariant()
    def blueprint_index_valid(self):
        """Blueprint index is always non-negative."""
        assert self.policy._blueprint_index >= 0, \
            f"Blueprint index {self.policy._blueprint_index} < 0"

    @invariant()
    def germination_count_valid(self):
        """Germination count is always non-negative."""
        assert self.policy._germination_count >= 0, \
            f"Germination count {self.policy._germination_count} < 0"

    @invariant()
    def penalties_non_negative(self):
        """All blueprint penalties are non-negative."""
        for bp, penalty in self.policy._blueprint_penalties.items():
            assert penalty >= 0, f"Penalty for {bp} is negative: {penalty}"

    @invariant()
    def decisions_list_valid(self):
        """Decisions list length matches our count (after reset: 0)."""
        # After reset, decisions list is cleared
        # We track our own count that resets with the policy
        expected = self.decision_count
        actual = len(self.policy._decisions_made)
        assert actual == expected, \
            f"Expected {expected} decisions, got {actual}"

    @invariant()
    def decisions_are_valid_type(self):
        """All stored decisions are TamiyoDecision instances."""
        for d in self.policy._decisions_made:
            assert isinstance(d, TamiyoDecision), \
                f"Invalid decision type: {type(d)}"


# Create test class for pytest discovery
TestHeuristicPolicyStates = HeuristicPolicyStateMachine.TestCase


# =============================================================================
# State Machine: SignalTracker
# =============================================================================

class SignalTrackerStateMachine(RuleBasedStateMachine):
    """Stateful test for SignalTracker through random update sequences.

    This state machine:
    - Updates tracker with various loss/accuracy values
    - Resets tracker periodically
    - Verifies invariants after every operation
    """

    def __init__(self):
        super().__init__()
        self.tracker = None
        self.epoch = 0
        self.update_count = 0
        self.max_accuracy_seen = 0.0
        self.was_stabilized = False

    @initialize()
    def setup(self):
        """Initialize fresh tracker."""
        self.tracker = SignalTracker(history_window=10)
        self.epoch = 0
        self.update_count = 0
        self.max_accuracy_seen = 0.0
        self.was_stabilized = False

    @rule(
        loss=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        accuracy=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def update_tracker(self, loss, accuracy):
        """Update tracker with new values."""
        self.tracker.update(
            epoch=self.epoch,
            global_step=self.epoch * 100,
            train_loss=loss,
            train_accuracy=accuracy,
            val_loss=loss,
            val_accuracy=accuracy,
            active_seeds=[],
        )
        self.epoch += 1
        self.update_count += 1
        self.max_accuracy_seen = max(self.max_accuracy_seen, accuracy)

        # Track stabilization latch
        if self.tracker.is_stabilized:
            self.was_stabilized = True

    @rule()
    def reset_tracker(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.update_count = 0
        self.max_accuracy_seen = 0.0
        self.was_stabilized = False
        # Note: epoch continues to advance

    # =========================================================================
    # Invariants
    # =========================================================================

    @invariant()
    def history_bounded(self):
        """History never exceeds window size."""
        assert len(self.tracker._loss_history) <= self.tracker.history_window, \
            f"Loss history {len(self.tracker._loss_history)} > window {self.tracker.history_window}"
        assert len(self.tracker._accuracy_history) <= self.tracker.history_window, \
            f"Accuracy history {len(self.tracker._accuracy_history)} > window {self.tracker.history_window}"

    @invariant()
    def plateau_non_negative(self):
        """Plateau count is always non-negative."""
        assert self.tracker._plateau_count >= 0, \
            f"Plateau count {self.tracker._plateau_count} < 0"

    @invariant()
    def stable_count_non_negative(self):
        """Stable count is always non-negative."""
        assert self.tracker._stable_count >= 0, \
            f"Stable count {self.tracker._stable_count} < 0"

    @invariant()
    def best_accuracy_non_negative(self):
        """Best accuracy is always non-negative."""
        assert self.tracker._best_accuracy >= 0, \
            f"Best accuracy {self.tracker._best_accuracy} < 0"

    @invariant()
    def stabilization_latch_monotonic(self):
        """Once stabilized, stays stabilized (until reset)."""
        if self.was_stabilized:
            assert self.tracker.is_stabilized, \
                "Stabilization latch violated: was True, now False"


# Create test class for pytest discovery
TestSignalTrackerStates = SignalTrackerStateMachine.TestCase


# =============================================================================
# State Machine: Combined Policy + Tracker
# =============================================================================

class CombinedTamiyoStateMachine(RuleBasedStateMachine):
    """Stateful test for Policy and Tracker working together.

    This tests the realistic usage pattern where tracker feeds signals
    to the policy for decision making.
    """

    def __init__(self):
        super().__init__()
        self.policy = None
        self.tracker = None
        self.epoch = 0
        self.decisions = []

    @initialize()
    def setup(self):
        """Initialize fresh policy and tracker."""
        self.policy = HeuristicTamiyo(topology="cnn")
        self.tracker = SignalTracker()
        self.epoch = 0
        self.decisions = []

    @rule(
        loss=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        accuracy=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def training_step(self, loss, accuracy):
        """Simulate one training step: update tracker, make decision."""
        # Update tracker
        signals = self.tracker.update(
            epoch=self.epoch,
            global_step=self.epoch * 100,
            train_loss=loss,
            train_accuracy=accuracy,
            val_loss=loss,
            val_accuracy=accuracy,
            active_seeds=[],
        )

        # Make decision
        decision = self.policy.decide(signals, active_seeds=[])
        self.decisions.append(decision)
        self.epoch += 1

    @rule()
    def reset_both(self):
        """Reset both policy and tracker."""
        self.policy.reset()
        self.tracker.reset()
        self.decisions = []

    @rule()
    def reset_policy_only(self):
        """Reset only policy (tracker continues)."""
        self.policy.reset()
        # Decisions tracked separately
        self.decisions = []

    # =========================================================================
    # Invariants
    # =========================================================================

    @invariant()
    def tracker_signals_valid(self):
        """Tracker always has valid signal state."""
        assert self.tracker._plateau_count >= 0
        assert self.tracker._stable_count >= 0
        assert len(self.tracker._loss_history) <= self.tracker.history_window

    @invariant()
    def policy_state_valid(self):
        """Policy always has valid internal state."""
        assert self.policy._blueprint_index >= 0
        assert self.policy._germination_count >= 0
        for p in self.policy._blueprint_penalties.values():
            assert p >= 0

    @invariant()
    def all_decisions_valid(self):
        """All decisions in history are valid."""
        for d in self.decisions:
            assert isinstance(d, TamiyoDecision)
            assert d.action is not None
            assert d.reason  # Non-empty


# Create test class for pytest discovery
TestCombinedTamiyoStates = CombinedTamiyoStateMachine.TestCase


# =============================================================================
# Standard Property Tests for Reset Completeness
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestResetCompleteness:
    """Verify reset() fully clears all state."""

    def test_policy_reset_clears_all(self):
        """Policy.reset() clears all internal state."""
        policy = HeuristicTamiyo(topology="cnn")

        # Build up some state
        class MockSignals:
            class metrics:
                epoch = 10
                plateau_epochs = 5
                host_stabilized = 1
                accuracy_delta = 0.0

        for _ in range(5):
            policy.decide(MockSignals(), active_seeds=[])

        # Add penalties
        policy._blueprint_penalties["conv_light"] = 5.0
        policy._last_prune_epoch = 10

        # Reset
        policy.reset()

        # Verify all state cleared
        assert policy._blueprint_index == 0
        assert policy._germination_count == 0
        assert len(policy._decisions_made) == 0
        assert policy._last_prune_epoch == -100
        assert len(policy._blueprint_penalties) == 0
        assert policy._last_decay_epoch == -1

    def test_tracker_reset_clears_all(self):
        """Tracker.reset() clears all internal state."""
        tracker = SignalTracker()

        # Build up some state
        for i in range(10):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=1.0 - i * 0.05,
                train_accuracy=50.0 + i,
                val_loss=1.0 - i * 0.05,
                val_accuracy=50.0 + i,
                active_seeds=[],
            )

        # Reset
        tracker.reset()

        # Verify all state cleared
        assert len(tracker._loss_history) == 0
        assert len(tracker._accuracy_history) == 0
        assert tracker._best_accuracy == 0.0
        assert tracker._plateau_count == 0
        assert tracker._prev_accuracy == 0.0
        assert tracker._prev_loss == float('inf')
        assert tracker._is_stabilized is False
        assert tracker._stable_count == 0
