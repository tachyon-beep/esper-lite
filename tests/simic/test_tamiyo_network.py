"""Tests for FactoredRecurrentActorCritic - Tamiyo's neural network."""

import pytest
import torch

from esper.simic.tamiyo_network import FactoredRecurrentActorCritic


class TestFactoredRecurrentActorCritic:
    """Tests for the factored recurrent network."""

    def test_forward_returns_all_heads(self):
        """Forward pass must return logits for all 4 factored heads."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 1, 50)  # [batch, seq, state_dim]

        output = net(state)

        assert "slot_logits" in output
        assert "blueprint_logits" in output
        assert "blend_logits" in output
        assert "op_logits" in output
        assert "value" in output
        assert "hidden" in output

        # Check shapes
        assert output["slot_logits"].shape == (2, 1, 3)  # NUM_SLOTS=3
        assert output["blueprint_logits"].shape == (2, 1, 5)  # NUM_BLUEPRINTS=5
        assert output["blend_logits"].shape == (2, 1, 3)  # NUM_BLENDS=3
        assert output["op_logits"].shape == (2, 1, 4)  # NUM_OPS=4
        assert output["value"].shape == (2, 1)

    def test_hidden_state_propagates(self):
        """LSTM hidden state must propagate across time steps."""
        net = FactoredRecurrentActorCritic(state_dim=50, lstm_hidden_dim=64)

        batch_size = 2
        hidden = net.get_initial_hidden(batch_size, torch.device("cpu"))

        # First step
        state1 = torch.randn(batch_size, 1, 50)
        output1 = net(state1, hidden=hidden)

        # Second step with updated hidden
        state2 = torch.randn(batch_size, 1, 50)
        output2 = net(state2, hidden=output1["hidden"])

        # Hidden states should be different after processing different inputs
        h1, c1 = output1["hidden"]
        h2, c2 = output2["hidden"]
        assert not torch.allclose(h1, h2), "Hidden states should change between steps"

    def test_masks_applied_correctly(self):
        """Invalid actions must have -inf logits after masking."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(1, 1, 50)

        # Mask out slot 0 and 2, only slot 1 valid
        slot_mask = torch.tensor([[[False, True, False]]])
        op_mask = torch.tensor([[[True, True, True, True]]])

        output = net(
            state,
            slot_mask=slot_mask,
            blueprint_mask=None,
            blend_mask=None,
            op_mask=op_mask,
        )

        slot_logits = output["slot_logits"][0, 0]
        assert slot_logits[0] == float("-inf")
        assert slot_logits[1] != float("-inf")
        assert slot_logits[2] == float("-inf")

    def test_per_head_log_probs(self):
        """evaluate_actions must return per-head log probs."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 5, 50)  # [batch, seq, state_dim]
        actions = {
            "slot": torch.zeros(2, 5, dtype=torch.long),
            "blueprint": torch.zeros(2, 5, dtype=torch.long),
            "blend": torch.zeros(2, 5, dtype=torch.long),
            "op": torch.zeros(2, 5, dtype=torch.long),
        }

        log_probs, values, entropy, hidden = net.evaluate_actions(state, actions)

        # Per-head log probs
        assert "slot" in log_probs
        assert "blueprint" in log_probs
        assert "blend" in log_probs
        assert "op" in log_probs

        # All should be [batch, seq]
        assert log_probs["slot"].shape == (2, 5)
        assert log_probs["blueprint"].shape == (2, 5)

    def test_get_action_returns_per_head_log_probs(self):
        """get_action must return per-head log probs for buffer storage."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 50)  # [batch, state_dim]

        actions, log_probs, values, hidden = net.get_action(state)

        # Should have per-head actions and log_probs
        assert "slot" in actions
        assert "blueprint" in actions
        assert "blend" in actions
        assert "op" in actions

        assert "slot" in log_probs
        assert "blueprint" in log_probs
        assert "blend" in log_probs
        assert "op" in log_probs

        # Actions should be batch-sized
        assert actions["slot"].shape == (2,)
        assert log_probs["slot"].shape == (2,)

    def test_entropy_normalized_per_head(self):
        """Entropy should be normalized by max entropy for each head."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 5, 50)
        actions = {
            "slot": torch.zeros(2, 5, dtype=torch.long),
            "blueprint": torch.zeros(2, 5, dtype=torch.long),
            "blend": torch.zeros(2, 5, dtype=torch.long),
            "op": torch.zeros(2, 5, dtype=torch.long),
        }

        _, _, entropy, _ = net.evaluate_actions(state, actions)

        # Normalized entropy should be between 0 and 1
        for key in ["slot", "blueprint", "blend", "op"]:
            assert (entropy[key] >= 0).all(), f"{key} entropy has negative values"
            assert (entropy[key] <= 1.01).all(), f"{key} entropy exceeds 1 (not normalized)"

    def test_deterministic_action(self):
        """Deterministic mode should always return argmax."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(1, 50)

        # Run multiple times - deterministic should be consistent
        actions1, _, _, _ = net.get_action(state, deterministic=True)
        actions2, _, _, _ = net.get_action(state, deterministic=True)

        for key in ["slot", "blueprint", "blend", "op"]:
            assert actions1[key] == actions2[key], f"{key} action not deterministic"
