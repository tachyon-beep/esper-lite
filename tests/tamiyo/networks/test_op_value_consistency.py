"""Unit tests for op/value consistency in get_action().

CRITICAL BUG FIX: Verify that get_action() uses the same op for both
value computation and action selection. This prevents biased advantages
and bootstrap corruption.

Related bug report: docs/bugs/CRITICAL-op-value-mismatch.md
"""

import torch
import pytest
from esper.tamiyo.networks import FactoredRecurrentActorCritic
from esper.leyline import HEAD_NAMES


class TestOpValueConsistency:
    """Verify get_action() uses same op for value and action."""

    @pytest.fixture
    def network(self):
        return FactoredRecurrentActorCritic(state_dim=126)

    @pytest.fixture
    def inputs(self, network):
        batch_size = 4
        state = torch.randn(batch_size, 126)
        bp_idx = torch.randint(0, 13, (batch_size, 3))
        masks = {k: None for k in HEAD_NAMES}
        return state, bp_idx, masks

    def test_deterministic_op_value_match(self, network, inputs):
        """In deterministic mode, value must use argmax op."""
        state, bp_idx, masks = inputs

        result = network.get_action(state, bp_idx, hidden=None, deterministic=True)

        # Get expected value by recomputing with argmax op
        with torch.inference_mode():
            output = network.forward(
                state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None
            )
            op_logits = output["op_logits"][:, 0, :]
            argmax_op = op_logits.argmax(dim=-1)
            expected_value = network._compute_value(
                output["lstm_out"][:, 0:1, :], argmax_op.unsqueeze(1)
            ).squeeze(1)

        # Verify returned action is argmax
        torch.testing.assert_close(result.actions["op"], argmax_op)
        # Verify value corresponds to argmax op
        torch.testing.assert_close(result.values, expected_value, rtol=1e-5, atol=1e-5)

    def test_stochastic_op_value_match(self, network, inputs):
        """In stochastic mode, value must use same op as action."""
        state, bp_idx, masks = inputs

        # Run multiple times to ensure it's not just luck
        for _ in range(10):
            result = network.get_action(state, bp_idx, hidden=None, deterministic=False)

            # Verify sampled_op matches action["op"]
            torch.testing.assert_close(result.sampled_op, result.actions["op"])

            # Verify value was computed with the returned op
            # We can't directly compare to forward() output because forward() samples differently,
            # but we verify the contract: sampled_op == action["op"]
            assert torch.equal(result.sampled_op, result.actions["op"]), \
                "sampled_op must equal action['op'] in stochastic mode"

    def test_single_valid_op_consistency(self, network, inputs):
        """When only one op is valid, both modes should behave the same."""
        state, bp_idx, _ = inputs

        # Create mask with single valid op
        single_op_mask = torch.zeros(state.shape[0], network.num_ops, dtype=torch.bool)
        single_op_mask[:, 2] = True  # Only op 2 is valid

        result_det = network.get_action(state, bp_idx, hidden=None, op_mask=single_op_mask, deterministic=True)
        result_stoch = network.get_action(state, bp_idx, hidden=None, op_mask=single_op_mask, deterministic=False)

        # Both should select op 2
        assert (result_det.actions["op"] == 2).all(), "Deterministic should select the only valid op"
        assert (result_stoch.actions["op"] == 2).all(), "Stochastic should select the only valid op"

        # Values should be identical (same op used)
        torch.testing.assert_close(result_det.values, result_stoch.values, rtol=1e-5, atol=1e-5)

    def test_op_action_consistency_across_batch(self, network, inputs):
        """Verify op/value consistency holds for every batch element."""
        state, bp_idx, masks = inputs
        batch_size = state.shape[0]

        # Deterministic mode
        result = network.get_action(state, bp_idx, hidden=None, deterministic=True)

        # Check each batch element
        for i in range(batch_size):
            with torch.inference_mode():
                # Get single-element batch
                single_state = state[i:i+1]
                single_bp = bp_idx[i:i+1]

                output = network.forward(
                    single_state.unsqueeze(1), single_bp.unsqueeze(1), hidden=None
                )
                argmax_op = output["op_logits"][:, 0, :].argmax(dim=-1)

                # Verify this batch element used argmax
                assert result.actions["op"][i] == argmax_op[0], \
                    f"Batch element {i} action should be argmax"

    def test_value_recomputation_in_deterministic_mode(self, network, inputs):
        """Verify value is actually recomputed (not reused from forward())."""
        state, bp_idx, masks = inputs

        # Get forward() output
        with torch.inference_mode():
            output_forward = network.forward(
                state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None
            )
            value_from_forward = output_forward["value"][:, 0]
            sampled_op_from_forward = output_forward["sampled_op"][:, 0]

        # Get deterministic action
        result = network.get_action(state, bp_idx, hidden=None, deterministic=True)

        # If argmax differs from sample, value must differ
        op_differs = (result.actions["op"] != sampled_op_from_forward).any()
        value_differs = not torch.allclose(result.values, value_from_forward, rtol=1e-5, atol=1e-5)

        # If ops differ, values MUST differ (otherwise value wasn't recomputed)
        if op_differs:
            assert value_differs, \
                "When argmax_op != sampled_op, value must be recomputed (not reused from forward())"
