"""Integration tests for bootstrap value consistency.

CRITICAL BUG FIX: Verify bootstrap values (used for truncated episodes)
use the correct op for value computation. Bootstrap should use argmax op
when deterministic=True, not a random sample.

Related bug report: docs/bugs/CRITICAL-op-value-mismatch.md
"""

import torch
from esper.tamiyo.policy import create_policy
from esper.leyline import HEAD_NAMES, OBS_V3_NON_BLUEPRINT_DIM


class TestBootstrapConsistency:
    """Verify bootstrap values use correct op."""

    def test_bootstrap_uses_deterministic_op(self):
        """Bootstrap (deterministic=True) should use argmax op for value."""
        policy = create_policy(
            policy_type="lstm",
            state_dim=OBS_V3_NON_BLUEPRINT_DIM,
            num_slots=3,
            device="cpu",
            compile_mode="off",
        )

        state = torch.randn(2, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (2, 3))

        # Bootstrap call (as done in vectorized.py line 3236)
        result = policy.get_action(
            features=state,
            blueprint_indices=bp_idx,
            masks={k: None for k in HEAD_NAMES},
            hidden=None,
            deterministic=True
        )

        # Verify action is argmax
        with torch.inference_mode():
            output = policy.network.forward(
                state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None
            )
            expected_argmax = output["op_logits"][:, 0, :].argmax(dim=-1)

        torch.testing.assert_close(result.action["op"], expected_argmax)

        # Verify value is Q(s, argmax_op)
        with torch.inference_mode():
            expected_value = policy.network._compute_value(
                output["lstm_out"][:, 0:1, :],
                expected_argmax.unsqueeze(1)
            ).squeeze(1)

        torch.testing.assert_close(result.value, expected_value, rtol=1e-5, atol=1e-5)

    def test_bootstrap_value_not_from_forward_sample(self):
        """Verify bootstrap doesn't reuse value from forward()'s random sample."""
        policy = create_policy(
            policy_type="lstm",
            state_dim=OBS_V3_NON_BLUEPRINT_DIM,
            num_slots=3,
            device="cpu",
            compile_mode="off",
        )

        state = torch.randn(4, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (4, 3))

        # Get forward() output (uses sample)
        with torch.inference_mode():
            output_forward = policy.network.forward(
                state.unsqueeze(1), bp_idx.unsqueeze(1), hidden=None
            )
            value_from_forward = output_forward["value"][:, 0]
            sampled_op_from_forward = output_forward["sampled_op"][:, 0]

        # Get bootstrap value (uses argmax)
        result = policy.get_action(
            features=state,
            blueprint_indices=bp_idx,
            masks={k: None for k in HEAD_NAMES},
            hidden=None,
            deterministic=True
        )

        # If argmax differs from sample, bootstrap value MUST differ
        op_differs = (result.action["op"] != sampled_op_from_forward).any()

        if op_differs:
            # Values must differ because they're conditioned on different ops
            value_differs = not torch.allclose(result.value, value_from_forward, rtol=1e-5, atol=1e-5)
            assert value_differs, \
                "Bootstrap value must be Q(s, argmax_op), not Q(s, sampled_op) from forward()"

    def test_bootstrap_consistency_across_calls(self):
        """Verify bootstrap is deterministic across multiple calls."""
        policy = create_policy(
            policy_type="lstm",
            state_dim=OBS_V3_NON_BLUEPRINT_DIM,
            num_slots=3,
            device="cpu",
            compile_mode="off",
        )

        state = torch.randn(3, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, 13, (3, 3))
        masks = {k: None for k in HEAD_NAMES}

        # Call bootstrap multiple times (should be deterministic)
        results = []
        for _ in range(5):
            result = policy.get_action(
                features=state,
                blueprint_indices=bp_idx,
                masks=masks,
                hidden=None,
                deterministic=True
            )
            results.append((result.action["op"].clone(), result.value.clone()))

        # All calls should produce identical results
        for i in range(1, 5):
            torch.testing.assert_close(results[i][0], results[0][0], msg="Op actions should be identical")
            torch.testing.assert_close(results[i][1], results[0][1], msg="Values should be identical")
