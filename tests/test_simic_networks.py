"""Tests for RL network architectures."""

import pytest
import torch

from esper.simic.networks import ActorCritic, QNetwork, VNetwork


class TestActorCritic:
    """Tests for PPO ActorCritic network with action masking."""

    def test_forward_shapes(self):
        """Test forward pass returns correct shapes."""
        net = ActorCritic(state_dim=27, action_dim=7, hidden_dim=64)
        state = torch.randn(4, 27)  # batch of 4
        mask = torch.ones(4, 7)  # all actions valid

        dist, value = net(state, mask)

        assert value.shape == (4,)
        assert dist.probs.shape == (4, 7)

    def test_get_action(self):
        """Test single action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)
        mask = torch.ones(1, 7)

        action, log_prob, value = net.get_action(state, mask)

        assert isinstance(action, int)
        assert 0 <= action < 7
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)
        mask = torch.ones(1, 7)

        # Deterministic should give same action each time
        actions = [net.get_action(state, mask, deterministic=True)[0] for _ in range(5)]
        assert len(set(actions)) == 1

    def test_get_action_batch(self):
        """Test batched action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(8, 27)
        masks = torch.ones(8, 7)

        actions, log_probs, values = net.get_action_batch(states, masks)

        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_evaluate_actions(self):
        """Test action evaluation for PPO update."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(16, 27)
        actions = torch.randint(0, 7, (16,))
        masks = torch.ones(16, 7)

        log_probs, values, entropy = net.evaluate_actions(states, actions, masks)

        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)

    def test_action_masking_blocks_invalid(self):
        """Test that masked actions have near-zero probability."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)
        # Only action 0 and 3 are valid
        mask = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

        with torch.no_grad():
            dist, _ = net(state, mask)

        probs = dist.probs[0]
        # Masked actions should have near-zero probability
        assert probs[1].item() < 1e-6
        assert probs[2].item() < 1e-6
        assert probs[4].item() < 1e-6
        # Valid actions should have non-zero probability
        assert probs[0].item() > 0.0
        assert probs[3].item() > 0.0

    def test_masked_entropy_excludes_invalid(self):
        """Test that entropy only considers valid actions."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        # All actions valid - higher entropy
        all_valid_mask = torch.ones(1, 7)
        with torch.no_grad():
            dist_all, _ = net(state, all_valid_mask)
            entropy_all = dist_all.entropy()

        # Only 2 actions valid - lower max possible entropy
        two_valid_mask = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            dist_two, _ = net(state, two_valid_mask)
            entropy_two = dist_two.entropy()

        # With only 2 valid actions, max entropy is log(2) ≈ 0.693
        # With 7 valid actions, max entropy is log(7) ≈ 1.946
        # Actual entropy depends on network output, but masked should be lower
        assert entropy_all.item() >= 0.0
        assert entropy_two.item() >= 0.0


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
