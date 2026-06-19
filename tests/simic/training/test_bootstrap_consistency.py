"""Bootstrap value consistency under the op-INDEPENDENT V(s) baseline (P0-1).

The truncation bootstrap is V(s'), produced by get_action(deterministic=True).value.
Because V(s) is op-independent, the bootstrap:
  - equals state_value_head(lstm_out) (NOT Q(s, argmax_op)),
  - is deterministic across repeated calls,
  - is identical between deterministic and stochastic legs (V never sees the op).
"""

import torch
from esper.tamiyo.policy import create_policy
from esper.leyline import HEAD_NAMES, OBS_V3_NON_BLUEPRINT_DIM, SlotConfig


class TestBootstrapConsistency:
    """Verify bootstrap values are the op-INDEPENDENT V(s)."""

    def _policy(self, slot_config):
        return create_policy(
            policy_type="lstm",
            state_dim=OBS_V3_NON_BLUEPRINT_DIM,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )

    def test_bootstrap_is_state_value(self):
        """Bootstrap (deterministic=True).value == V(s) = state_value_head(lstm_out)."""
        slot_config = SlotConfig.default()
        policy = self._policy(slot_config)

        state = torch.randn(2, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (2, slot_config.num_slots))

        result = policy.get_action(
            features=state,
            blueprint_indices=bp_idx,
            masks={k: None for k in HEAD_NAMES} | {"slot_by_op": None},
            hidden=None,
            deterministic=True,
        )

        with torch.inference_mode():
            output = policy.network.forward(
                state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None
            )
            expected_v = policy.network._compute_state_value(output["lstm_out"])[:, 0]

        torch.testing.assert_close(result.value, expected_v, rtol=1e-5, atol=1e-5)

    def test_bootstrap_not_q_argmax_op(self):
        """Bootstrap is V(s), NOT Q(s, argmax_op): they differ when the heads differ."""
        slot_config = SlotConfig.default()
        policy = self._policy(slot_config)

        state = torch.randn(4, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (4, slot_config.num_slots))

        result = policy.get_action(
            features=state,
            blueprint_indices=bp_idx,
            masks={k: None for k in HEAD_NAMES} | {"slot_by_op": None},
            hidden=None,
            deterministic=True,
        )

        with torch.inference_mode():
            output = policy.network.forward(
                state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None
            )
            argmax_op = output["op_logits"][:, 0, :].argmax(dim=-1)
            q_argmax = policy.network._compute_q(
                output["lstm_out"], argmax_op.unsqueeze(1)
            )[:, 0]

        # The op-independent V(s) generically differs from the op-conditioned Q(s, op).
        assert not torch.allclose(result.value, q_argmax, rtol=1e-4, atol=1e-4)

    def test_bootstrap_deterministic_across_calls(self):
        """Bootstrap is deterministic across multiple calls."""
        slot_config = SlotConfig.default()
        policy = self._policy(slot_config)

        state = torch.randn(3, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (3, slot_config.num_slots))
        masks = {k: None for k in HEAD_NAMES} | {"slot_by_op": None}

        results = []
        for _ in range(5):
            result = policy.get_action(
                features=state,
                blueprint_indices=bp_idx,
                masks=masks,
                hidden=None,
                deterministic=True,
            )
            results.append((result.action["op"].clone(), result.value.clone()))

        for i in range(1, 5):
            torch.testing.assert_close(results[i][0], results[0][0], msg="Op actions should be identical")
            torch.testing.assert_close(results[i][1], results[0][1], msg="Values should be identical")

    def test_bootstrap_value_op_independent(self):
        """V(s) bootstrap is identical for deterministic and stochastic legs."""
        slot_config = SlotConfig.default()
        policy = self._policy(slot_config)

        state = torch.randn(3, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (3, slot_config.num_slots))
        masks = {k: None for k in HEAD_NAMES} | {"slot_by_op": None}

        det = policy.get_action(features=state, blueprint_indices=bp_idx, masks=masks,
                                hidden=None, deterministic=True)
        stoch = policy.get_action(features=state, blueprint_indices=bp_idx, masks=masks,
                                  hidden=None, deterministic=False)
        torch.testing.assert_close(det.value, stoch.value, rtol=1e-5, atol=1e-5)
