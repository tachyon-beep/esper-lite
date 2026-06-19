"""Op/value semantics for get_action() under the op-INDEPENDENT V(s) baseline (P0-1).

The PPO baseline is now V(s) (state_value), which does NOT depend on the op. So
get_action().values is op-independent and identical for deterministic and stochastic
legs. What MUST still hold: the returned op action equals sampled_op (consistency
between the action taken and the Q(s, op) telemetry head), and V(s) is unaffected by
which op is selected.
"""

import torch
from torch import nn
import pytest
from esper.tamiyo.networks import FactoredRecurrentActorCritic
from esper.leyline import HEAD_NAMES


class TestOpIndependentBaseline:
    """get_action() returns the op-INDEPENDENT V(s) baseline."""

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

    def test_value_equals_state_value_head(self, network, inputs):
        """get_action().values == V(s) from state_value_head (op-independent)."""
        state, bp_idx, _ = inputs
        result = network.get_action(state, bp_idx, hidden=None, deterministic=True)
        with torch.inference_mode():
            output = network.forward(state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None)
            expected = network._compute_state_value(output["lstm_out"])[:, 0]
        torch.testing.assert_close(result.values, expected, rtol=1e-5, atol=1e-5)

    def test_value_is_op_independent(self, monkeypatch, network, inputs):
        """If only the op differs between two get_action calls, V(s) is unchanged."""
        state, bp_idx, _ = inputs

        # Make the op head uniform so deterministic/stochastic ops vary, then confirm
        # the value does NOT depend on which op is selected.
        class UniformOpHead(nn.Module):
            def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
                return lstm_out.new_zeros((*lstm_out.shape[:-1], network.num_ops))

        network.op_head = UniformOpHead()
        network.eval()

        det = network.get_action(state, bp_idx, hidden=None, deterministic=True)
        torch.manual_seed(7)
        stoch = network.get_action(state, bp_idx, hidden=None, deterministic=False)

        # The deterministic and stochastic legs select (generally) different ops, but
        # the V(s) baseline is identical because it never sees the op.
        torch.testing.assert_close(
            det.values, stoch.values, rtol=1e-5, atol=1e-5,
            msg="V(s) must be op-independent (same value regardless of selected op)",
        )

    def test_sampled_op_matches_action_stochastic(self, network, inputs):
        """The returned op action equals sampled_op (action / Q-telemetry consistency)."""
        state, bp_idx, _ = inputs
        for _ in range(10):
            result = network.get_action(state, bp_idx, hidden=None, deterministic=False)
            torch.testing.assert_close(result.sampled_op, result.actions["op"])

    def test_deterministic_op_is_argmax(self, network, inputs):
        """Deterministic mode selects the argmax op for the ACTION."""
        state, bp_idx, _ = inputs
        result = network.get_action(state, bp_idx, hidden=None, deterministic=True)
        with torch.inference_mode():
            output = network.forward(state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None)
            argmax_op = output["op_logits"][:, 0, :].argmax(dim=-1)
        torch.testing.assert_close(result.actions["op"], argmax_op)

    def test_value_identical_det_vs_stoch(self, network, inputs):
        """Even with a real (non-uniform) op head, V(s) is the same for both legs."""
        state, bp_idx, _ = inputs
        det = network.get_action(state, bp_idx, hidden=None, deterministic=True)
        stoch = network.get_action(state, bp_idx, hidden=None, deterministic=False)
        torch.testing.assert_close(det.values, stoch.values, rtol=1e-5, atol=1e-5)

    def test_single_valid_op_consistency(self, network, inputs):
        """When only one op is valid, both modes select it and V(s) matches."""
        state, bp_idx, _ = inputs
        single_op_mask = torch.zeros(state.shape[0], network.num_ops, dtype=torch.bool)
        single_op_mask[:, 2] = True

        result_det = network.get_action(state, bp_idx, hidden=None, op_mask=single_op_mask, deterministic=True)
        result_stoch = network.get_action(state, bp_idx, hidden=None, op_mask=single_op_mask, deterministic=False)

        assert (result_det.actions["op"] == 2).all()
        assert (result_stoch.actions["op"] == 2).all()
        torch.testing.assert_close(result_det.values, result_stoch.values, rtol=1e-5, atol=1e-5)
