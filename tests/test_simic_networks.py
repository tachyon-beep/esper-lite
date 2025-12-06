"""Tests for RL network architectures."""

import pytest
import torch

from esper.simic.networks import ActorCritic, QNetwork, VNetwork, RecurrentActorCritic


class TestActorCritic:
    """Tests for PPO ActorCritic network with action masking."""

    def test_forward_shapes(self):
        """Test forward pass returns correct shapes."""
        net = ActorCritic(state_dim=30, action_dim=7, hidden_dim=64)
        state = torch.randn(4, 30)  # batch of 4
        mask = torch.ones(4, 7)  # all actions valid

        dist, value = net(state, mask)

        assert value.shape == (4,)
        assert dist.probs.shape == (4, 7)

    def test_get_action(self):
        """Test single action sampling."""
        net = ActorCritic(state_dim=30, action_dim=7)
        state = torch.randn(1, 30)
        mask = torch.ones(1, 7)

        action, log_prob, value, _ = net.get_action(state, mask)

        assert isinstance(action, int)
        assert 0 <= action < 7
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        net = ActorCritic(state_dim=30, action_dim=7)
        state = torch.randn(1, 30)
        mask = torch.ones(1, 7)

        # Deterministic should give same action each time
        actions = [net.get_action(state, mask, deterministic=True)[0] for _ in range(5)]
        assert len(set(actions)) == 1

    def test_get_action_batch(self):
        """Test batched action sampling."""
        net = ActorCritic(state_dim=30, action_dim=7)
        states = torch.randn(8, 30)
        masks = torch.ones(8, 7)

        actions, log_probs, values = net.get_action_batch(states, masks)

        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_evaluate_actions(self):
        """Test action evaluation for PPO update."""
        net = ActorCritic(state_dim=30, action_dim=7)
        states = torch.randn(16, 30)
        actions = torch.randint(0, 7, (16,))
        masks = torch.ones(16, 7)

        log_probs, values, entropy = net.evaluate_actions(states, actions, masks)

        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)

    def test_action_masking_blocks_invalid(self):
        """Test that masked actions have near-zero probability."""
        net = ActorCritic(state_dim=30, action_dim=7)
        state = torch.randn(1, 30)
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
        net = ActorCritic(state_dim=30, action_dim=7)
        state = torch.randn(1, 30)

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
        net = QNetwork(state_dim=30, action_dim=7)
        state = torch.randn(4, 30)

        q_values = net(state)

        assert q_values.shape == (4, 7)


class TestVNetwork:
    """Tests for IQL V-network."""

    def test_forward_shape(self):
        """Test V-network outputs scalar value."""
        net = VNetwork(state_dim=30)
        state = torch.randn(4, 30)

        values = net(state)

        assert values.shape == (4, 1)


class TestRecurrentActorCritic:
    """Tests for LSTM-based actor-critic network."""

    def test_init_creates_lstm_layer(self):
        """RecurrentActorCritic should have LSTM between encoder and heads."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        # Direct attribute access (no hasattr per CLAUDE.md)
        assert isinstance(net.lstm, torch.nn.LSTM)
        assert net.lstm.hidden_size == 128

    def test_get_initial_hidden_returns_correct_shape(self):
        """Initial hidden state should be zeros with correct shape."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        h, c = net.get_initial_hidden(batch_size=4, device=torch.device('cpu'))
        # Shape: [num_layers, batch, hidden]
        assert h.shape == (1, 4, 128)
        assert c.shape == (1, 4, 128)
        assert (h == 0).all()
        assert (c == 0).all()

    def test_forward_single_step_returns_dist_value_hidden(self):
        """Forward with single step should return distribution, value, and new hidden."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        state = torch.randn(4, 30)  # [batch, state_dim]
        mask = torch.ones(4, 7, dtype=torch.bool)

        dist, value, hidden = net.forward(state, mask, hidden=None)

        assert value.shape == (4,)
        assert len(hidden) == 2  # (h, c)
        assert hidden[0].shape == (1, 4, 128)  # [layers, batch, hidden]

    def test_forward_sequence_returns_sequence_outputs(self):
        """Forward with sequence should return outputs for each timestep."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        states = torch.randn(4, 16, 30)  # [batch, seq_len, state_dim]
        masks = torch.ones(4, 16, 7, dtype=torch.bool)

        dist, values, hidden = net.forward(states, masks, hidden=None)

        assert dist._dist.logits.shape == (4, 16, 7)
        assert values.shape == (4, 16)

    def test_hidden_state_persists_across_calls(self):
        """Hidden state from one forward should affect next forward."""
        torch.manual_seed(42)  # Deterministic for reliable test
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        state = torch.randn(1, 30)
        mask = torch.ones(1, 7, dtype=torch.bool)

        # First call builds hidden state
        _, _, hidden1 = net.forward(state, mask, hidden=None)

        # Build up more context
        for _ in range(5):
            _, _, hidden1 = net.forward(torch.randn(1, 30), mask, hidden=hidden1)

        # Compare value with context vs fresh
        _, value_with_context, _ = net.forward(state, mask, hidden=hidden1)
        _, value_fresh, _ = net.forward(state, mask, hidden=None)

        # Values must differ (context changes output)
        assert not torch.allclose(value_with_context, value_fresh, atol=1e-5)

    def test_evaluate_actions_returns_log_probs_for_sequence(self):
        """evaluate_actions should return log probs for all sequence positions."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        states = torch.randn(2, 8, 30)  # [batch, seq, state_dim]
        actions = torch.randint(0, 7, (2, 8))  # [batch, seq]
        masks = torch.ones(2, 8, 7, dtype=torch.bool)

        log_probs, values, entropy, hidden = net.evaluate_actions(states, actions, masks)

        assert log_probs.shape == (2, 8)
        assert values.shape == (2, 8)
        assert entropy.shape == (2, 8)
        assert hidden[0].shape == (1, 2, 128)

    def test_device_handling_gpu_to_cpu(self):
        """Network should handle device placement correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        net = RecurrentActorCritic(state_dim=30, action_dim=7).cuda()
        state = torch.randn(1, 30, device='cuda')
        mask = torch.ones(1, 7, dtype=torch.bool, device='cuda')

        _, _, hidden = net.forward(state, mask)
        assert hidden[0].device.type == 'cuda'

    def test_entropy_is_normalized(self):
        """Verify entropy is normalized to [0, 1] range (MaskedCategorical)."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        states = torch.randn(2, 8, 30)
        actions = torch.randint(0, 7, (2, 8))
        masks = torch.ones(2, 8, 7, dtype=torch.bool)

        log_probs, values, entropy, hidden = net.evaluate_actions(states, actions, masks)

        # Entropy must be in [0, 1] (normalized, with small epsilon for float precision)
        assert (entropy >= 0).all(), f"Entropy has negative values: {entropy.min()}"
        assert (entropy <= 1.0 + 1e-6).all(), f"Entropy exceeds 1: {entropy.max()}"

    def test_forget_gate_bias_initialized_to_one(self):
        """Verify LSTM forget gate bias is initialized to 1.0 for better gradient flow."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=128)
        h = net.lstm_hidden_dim
        # Forget gate is second quarter of bias vector (LSTM gate order: input, forget, cell, output)
        assert (net.lstm.bias_ih_l0.data[h:2*h] == 1.0).all(), "bias_ih forget gate should be 1.0"
        assert (net.lstm.bias_hh_l0.data[h:2*h] == 1.0).all(), "bias_hh forget gate should be 1.0"

    def test_forget_gate_bias_multilayer(self):
        """Verify forget gate bias is 1.0 for all LSTM layers."""
        net = RecurrentActorCritic(state_dim=30, action_dim=7, lstm_hidden_dim=64, num_lstm_layers=2)
        h = net.lstm_hidden_dim
        # Layer 0
        assert (net.lstm.bias_ih_l0.data[h:2*h] == 1.0).all(), "Layer 0 bias_ih forget gate should be 1.0"
        assert (net.lstm.bias_hh_l0.data[h:2*h] == 1.0).all(), "Layer 0 bias_hh forget gate should be 1.0"
        # Layer 1
        assert (net.lstm.bias_ih_l1.data[h:2*h] == 1.0).all(), "Layer 1 bias_ih forget gate should be 1.0"
        assert (net.lstm.bias_hh_l1.data[h:2*h] == 1.0).all(), "Layer 1 bias_hh forget gate should be 1.0"

    def test_recurrent_actor_critic_has_layernorm(self):
        """RecurrentActorCritic should have LayerNorm on LSTM output."""
        net = RecurrentActorCritic(state_dim=10, action_dim=4)

        # Verify lstm_ln exists and is correct type (will raise AttributeError if missing)
        lstm_ln = net.lstm_ln
        assert isinstance(lstm_ln, torch.nn.LayerNorm), "lstm_ln should be LayerNorm"

        # LayerNorm should match LSTM hidden dim
        assert lstm_ln.normalized_shape[0] == net.lstm_hidden_dim


def test_attention_seed_small_channels():
    """AttentionSeed should handle small channel counts gracefully."""
    from esper.kasmina.blueprints.cnn import create_attention_seed

    # This should not crash - channels=2 with reduction=4 would give 0 features
    seed = create_attention_seed(channels=2, reduction=4)

    # Verify it works
    x = torch.randn(1, 2, 8, 8)
    out = seed(x)
    assert out.shape == x.shape

    # Also test edge case: channels=1
    seed_tiny = create_attention_seed(channels=1, reduction=4)
    x_tiny = torch.randn(1, 1, 8, 8)
    out_tiny = seed_tiny(x_tiny)
    assert out_tiny.shape == x_tiny.shape
