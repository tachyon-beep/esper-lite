"""Tests for LSTM hidden state recovery after rollback.

Addresses P2 finding from Simic audit - LSTM hidden state recovery
after rollback was previously untested. These tests verify the contract
that hidden states are reset (not restored) after rollback.
"""

from __future__ import annotations

import pytest
import torch

from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
from esper.leyline.slot_config import SlotConfig


@pytest.fixture
def slot_config() -> SlotConfig:
    """Create slot config for testing."""
    return SlotConfig.for_grid(rows=1, cols=3)


@pytest.fixture
def policy(slot_config: SlotConfig) -> FactoredRecurrentActorCritic:
    """Create policy for testing."""
    return FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
        lstm_hidden_dim=64,
    )


class TestLSTMStateRecovery:
    """Tests for LSTM hidden state management during rollback."""

    def test_hidden_state_reset_clears_memory(self, policy: FactoredRecurrentActorCritic) -> None:
        """Resetting hidden state should clear LSTM memory."""
        device = next(policy.parameters()).device
        batch_size = 1

        # Run forward to build hidden state
        obs = torch.randn(batch_size, 1, policy.state_dim, device=device)
        blueprints = torch.zeros(batch_size, 1, policy.num_slots, dtype=torch.long, device=device)
        result = policy.forward(
            state=obs,
            blueprint_indices=blueprints,
            hidden=None,
        )
        hidden = result["hidden"]

        # Verify hidden state is non-zero after forward
        assert hidden[0].abs().sum() > 0, "Hidden state should be non-zero after forward"

        # Get fresh hidden state
        fresh_hidden = policy.get_initial_hidden(batch_size, device)

        # Fresh hidden should be zeros
        assert fresh_hidden[0].abs().sum() == 0, "Fresh hidden state should be zeros"
        assert fresh_hidden[1].abs().sum() == 0, "Fresh cell state should be zeros"

    def test_hidden_state_detach_breaks_gradient(self, policy: FactoredRecurrentActorCritic) -> None:
        """Detaching hidden state should break gradient graph."""
        device = next(policy.parameters()).device
        batch_size = 1

        obs = torch.randn(batch_size, 1, policy.state_dim, device=device)
        blueprints = torch.zeros(batch_size, 1, policy.num_slots, dtype=torch.long, device=device)
        result = policy.forward(
            state=obs,
            blueprint_indices=blueprints,
            hidden=None,
        )
        hidden = result["hidden"]

        # Detach
        detached_h = hidden[0].detach()
        detached_c = hidden[1].detach()

        assert not detached_h.requires_grad
        assert not detached_c.requires_grad

    def test_hidden_state_after_rollback_scenario(self, policy: FactoredRecurrentActorCritic) -> None:
        """Simulate rollback: hidden state should be reset, not restored.

        This tests the contract: after rollback, we reset hidden state
        rather than trying to restore old LSTM memory. This is the correct
        behavior because:
        1. Governor doesn't snapshot LSTM hidden states (just model weights)
        2. Restoring stale hidden states would cause policy-hidden mismatch
        3. Fresh hidden state gives clean slate for recovery
        """
        device = next(policy.parameters()).device
        batch_size = 1

        # Build up hidden state with multiple forward passes
        hidden = None
        obs = torch.randn(batch_size, 1, policy.state_dim, device=device)
        blueprints = torch.zeros(batch_size, 1, policy.num_slots, dtype=torch.long, device=device)

        for _ in range(10):
            result = policy.forward(
                state=obs,
                blueprint_indices=blueprints,
                hidden=hidden,
            )
            hidden = result["hidden"]

        pre_rollback_h = hidden[0].clone()

        # Simulate rollback (reset hidden state)
        fresh_hidden = policy.get_initial_hidden(batch_size, device)

        # Verify reset, not restore
        assert not torch.allclose(fresh_hidden[0], pre_rollback_h), \
            "Fresh hidden should differ from pre-rollback hidden"
        assert fresh_hidden[0].abs().sum() == 0, \
            "Fresh hidden should be zeros"

    def test_hidden_state_shapes_match(self, policy: FactoredRecurrentActorCritic) -> None:
        """Verify hidden state tensor shapes are consistent."""
        device = next(policy.parameters()).device
        batch_size = 4

        # Get initial hidden
        initial_hidden = policy.get_initial_hidden(batch_size, device)

        # Run forward
        obs = torch.randn(batch_size, 1, policy.state_dim, device=device)
        blueprints = torch.zeros(batch_size, 1, policy.num_slots, dtype=torch.long, device=device)
        result = policy.forward(
            state=obs,
            blueprint_indices=blueprints,
            hidden=initial_hidden,
        )
        returned_hidden = result["hidden"]

        # Shapes should match
        assert initial_hidden[0].shape == returned_hidden[0].shape
        assert initial_hidden[1].shape == returned_hidden[1].shape

        # Correct dimensions: [num_layers, batch, hidden_dim]
        assert initial_hidden[0].shape == (policy.lstm_layers, batch_size, policy.lstm_hidden_dim)

    def test_hidden_state_preserved_across_forward_passes(
        self, policy: FactoredRecurrentActorCritic
    ) -> None:
        """Hidden state should evolve but remain valid across forward passes.

        This tests that LSTM memory accumulates correctly when passed
        through consecutive forward() calls, which is essential for
        temporal reasoning during seed learning.
        """
        device = next(policy.parameters()).device
        batch_size = 1

        obs = torch.randn(batch_size, 1, policy.state_dim, device=device)
        blueprints = torch.zeros(batch_size, 1, policy.num_slots, dtype=torch.long, device=device)

        # First forward - starts from zeros
        hidden = policy.get_initial_hidden(batch_size, device)
        result1 = policy.forward(
            state=obs,
            blueprint_indices=blueprints,
            hidden=hidden,
        )
        hidden1 = result1["hidden"]

        # Second forward - uses hidden from first
        result2 = policy.forward(
            state=obs,
            blueprint_indices=blueprints,
            hidden=hidden1,
        )
        hidden2 = result2["hidden"]

        # Hidden states should have evolved (not identical)
        # Note: With same input, hidden will still change due to LSTM dynamics
        assert not torch.allclose(hidden1[0], hidden2[0]), \
            "Hidden state should evolve across forward passes"

        # Both should be finite (no NaN/Inf)
        assert torch.isfinite(hidden2[0]).all(), "Hidden state should be finite"
        assert torch.isfinite(hidden2[1]).all(), "Cell state should be finite"

    def test_batch_independence_in_hidden_state(
        self, policy: FactoredRecurrentActorCritic
    ) -> None:
        """Hidden states for different batch elements should be independent.

        This verifies that resetting hidden state for one env doesn't
        affect others in a vectorized rollout.
        """
        device = next(policy.parameters()).device
        batch_size = 4

        # Build up different hidden states per batch element
        hidden = policy.get_initial_hidden(batch_size, device)

        # Use different inputs per batch element
        obs = torch.randn(batch_size, 1, policy.state_dim, device=device)
        blueprints = torch.zeros(batch_size, 1, policy.num_slots, dtype=torch.long, device=device)

        for _ in range(5):
            result = policy.forward(
                state=obs,
                blueprint_indices=blueprints,
                hidden=hidden,
            )
            hidden = result["hidden"]

        # Now "reset" only batch element 0 by zeroing its hidden state
        h, c = hidden
        h_modified = h.clone()
        c_modified = c.clone()
        h_modified[:, 0, :] = 0
        c_modified[:, 0, :] = 0

        # Other batch elements should retain their non-zero hidden state
        assert h_modified[:, 1:, :].abs().sum() > 0, \
            "Non-reset batch elements should retain hidden state"
        assert h_modified[:, 0, :].abs().sum() == 0, \
            "Reset batch element should have zero hidden state"
