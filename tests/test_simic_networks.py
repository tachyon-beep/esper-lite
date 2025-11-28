"""Tests for RL network architectures."""

import pytest
import torch

from esper.simic.networks import ActorCritic, QNetwork, VNetwork


class TestActorCritic:
    """Tests for PPO ActorCritic network."""

    def test_forward_shapes(self):
        """Test forward pass returns correct shapes."""
        net = ActorCritic(state_dim=27, action_dim=7, hidden_dim=64)
        state = torch.randn(4, 27)  # batch of 4

        dist, value = net(state)

        assert value.shape == (4,)
        assert dist.probs.shape == (4, 7)

    def test_get_action(self):
        """Test single action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        action, log_prob, value = net.get_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 7
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        # Deterministic should give same action each time
        actions = [net.get_action(state, deterministic=True)[0] for _ in range(5)]
        assert len(set(actions)) == 1

    def test_get_action_batch(self):
        """Test batched action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(8, 27)

        actions, log_probs, values = net.get_action_batch(states)

        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_evaluate_actions(self):
        """Test action evaluation for PPO update."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(16, 27)
        actions = torch.randint(0, 7, (16,))

        log_probs, values, entropy = net.evaluate_actions(states, actions)

        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)


class TestQNetwork:
    """Tests for IQL Q-network."""

    def test_forward_shape(self):
        """Test Q-network outputs Q-values for all actions."""
        net = QNetwork(state_dim=27, action_dim=7)
        state = torch.randn(4, 27)

        q_values = net(state)

        assert q_values.shape == (4, 7)


class TestVNetwork:
    """Tests for IQL V-network."""

    def test_forward_shape(self):
        """Test V-network outputs scalar value."""
        net = VNetwork(state_dim=27)
        state = torch.randn(4, 27)

        values = net(state)

        assert values.shape == (4, 1)
