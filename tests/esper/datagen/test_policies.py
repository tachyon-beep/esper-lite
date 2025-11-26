"""Tests for behavior policy wrapper."""

import pytest
from unittest.mock import MagicMock, patch
from esper.datagen.policies import BehaviorPolicy, create_policy
from esper.datagen.configs import BehaviorPolicyConfig


class TestBehaviorPolicy:
    def test_greedy_decision(self):
        """Test that greedy policy returns highest prob action."""
        config = BehaviorPolicyConfig(policy_id="test", epsilon=0.0)
        policy = BehaviorPolicy(config)

        # Mock signals where WAIT is clearly best
        signals = self._make_signals(plateau_epochs=0, epoch=3)

        action, probs = policy.decide(signals)

        assert probs.greedy_action == "WAIT"
        assert probs.sampled_action == "WAIT"
        assert probs.was_exploratory == False
        assert probs.epsilon == 0.0

    def test_epsilon_greedy_exploration(self):
        """Test that epsilon > 0 can select non-greedy actions."""
        config = BehaviorPolicyConfig(policy_id="test", epsilon=1.0)  # Always explore
        policy = BehaviorPolicy(config)

        signals = self._make_signals(plateau_epochs=0, epoch=3)

        # With epsilon=1.0, uniform random, should eventually get non-WAIT
        actions_seen = set()
        for _ in range(100):
            action, probs = policy.decide(signals)
            actions_seen.add(action)
            assert probs.epsilon == 1.0

        # Should see multiple actions with full exploration
        assert len(actions_seen) > 1

    def test_behavior_prob_logged(self):
        """Test that behavior probability is correctly computed."""
        config = BehaviorPolicyConfig(policy_id="test", epsilon=0.2)
        policy = BehaviorPolicy(config)

        signals = self._make_signals(plateau_epochs=0, epoch=3)
        action, probs = policy.decide(signals)

        # Behavior prob should account for epsilon
        assert probs.behavior_prob > 0
        assert probs.behavior_prob <= 1.0
        # Sum of behavior probs should be 1
        total = sum(probs.behavior_probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_random_policy(self):
        """Test random policy (epsilon=1.0)."""
        config = BehaviorPolicyConfig.from_preset("random")
        policy = BehaviorPolicy(config)

        signals = self._make_signals(plateau_epochs=5, epoch=10)

        # All actions should have equal probability
        _, probs = policy.decide(signals)
        for p in probs.behavior_probs.values():
            assert abs(p - 0.25) < 1e-6

    def _make_signals(self, plateau_epochs: int, epoch: int):
        """Create mock signals for testing."""
        signals = MagicMock()
        signals.epoch = epoch
        signals.plateau_epochs = plateau_epochs
        signals.val_accuracy = 70.0
        signals.best_val_accuracy = 70.0
        signals.available_slots = 1
        signals.active_seeds = []
        return signals


class TestCreatePolicy:
    def test_create_from_preset(self):
        policy = create_policy("baseline")
        assert policy.config.policy_id == "baseline"

    def test_create_with_epsilon(self):
        policy = create_policy("baseline", epsilon=0.15)
        assert policy.config.epsilon == 0.15
