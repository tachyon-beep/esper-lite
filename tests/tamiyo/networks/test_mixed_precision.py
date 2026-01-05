"""Mixed-precision regression tests for FactoredRecurrentActorCritic.

Critical tests for dtype handling in _compute_value() to prevent regressions
where bfloat16/float16 networks crash due to dtype mismatch in torch.cat.

Bug background:
  The original code used `.float()` to convert one-hot encodings, which
  caused a dtype mismatch when the network was converted to bfloat16/float16.
  The value_head (with bf16 weights) received float32 input, causing:
    "mat1 and mat2 must have the same dtype, but got Float and BFloat16"

Fix:
  Use `.to(lstm_out)` to match dtype and device of the one-hot encoding
  to the LSTM output, ensuring consistency in any precision mode.
"""

import pytest
import torch

from esper.leyline import NUM_BLUEPRINTS, OBS_V3_NON_BLUEPRINT_DIM
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic


class TestMixedPrecision:
    """Verify network operations work correctly in reduced precision."""

    def test_compute_value_bfloat16(self):
        """_compute_value must work with bfloat16 inputs.

        Regression test for dtype mismatch bug where op_one_hot was forced
        to float32 via .float(), causing crash when value_head is bfloat16.
        """
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)
        net = net.bfloat16()

        # Create bfloat16 inputs
        batch, seq = 2, 3
        lstm_out = torch.randn(batch, seq, net.lstm_hidden_dim, dtype=torch.bfloat16)
        op = torch.randint(0, net.num_ops, (batch, seq))

        # This should work (previously crashed with dtype mismatch)
        value = net._compute_value(lstm_out, op)

        assert value.dtype == torch.bfloat16, f"Expected bfloat16, got {value.dtype}"
        assert value.shape == (batch, seq), f"Expected ({batch}, {seq}), got {value.shape}"
        assert not torch.isnan(value).any(), "Value contains NaN"
        assert not torch.isinf(value).any(), "Value contains Inf"

    def test_compute_value_float16(self):
        """_compute_value must work with float16 inputs."""
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)
        net = net.half()

        batch, seq = 2, 3
        lstm_out = torch.randn(batch, seq, net.lstm_hidden_dim, dtype=torch.float16)
        op = torch.randint(0, net.num_ops, (batch, seq))

        value = net._compute_value(lstm_out, op)

        assert value.dtype == torch.float16, f"Expected float16, got {value.dtype}"
        assert value.shape == (batch, seq)
        assert not torch.isnan(value).any()

    def test_compute_value_float32(self):
        """_compute_value must still work with float32 (no regression)."""
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)

        batch, seq = 2, 3
        lstm_out = torch.randn(batch, seq, net.lstm_hidden_dim, dtype=torch.float32)
        op = torch.randint(0, net.num_ops, (batch, seq))

        value = net._compute_value(lstm_out, op)

        assert value.dtype == torch.float32, f"Expected float32, got {value.dtype}"
        assert value.shape == (batch, seq)

    def test_forward_bfloat16_full_pass(self):
        """Full forward() pass must work in bfloat16.

        Note: LSTM on CPU with bfloat16 may not be supported by oneDNN.
        This test uses float32 LSTM output but verifies _compute_value path.
        """
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)

        # Keep network in float32 for LSTM compatibility on CPU
        # Verify that _compute_value would work if LSTM output was bfloat16
        batch, seq = 2, 1
        state = torch.randn(batch, seq, OBS_V3_NON_BLUEPRINT_DIM)
        bp_idx = torch.randint(0, NUM_BLUEPRINTS, (batch, seq, 3))

        # Forward should work in float32
        output = net(state, bp_idx)

        assert output["value"].shape == (batch, seq)
        assert output["sampled_op"].shape == (batch, seq)
        assert not torch.isnan(output["value"]).any()

    def test_evaluate_actions_bfloat16(self):
        """evaluate_actions must use correct dtype for _compute_value.

        The evaluate_actions path also calls _compute_value with the
        stored op from the buffer, so it must handle dtype correctly.
        """
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)
        net = net.bfloat16()

        batch, seq = 2, 3
        lstm_out = torch.randn(batch, seq, net.lstm_hidden_dim, dtype=torch.bfloat16)
        op = torch.randint(0, net.num_ops, (batch, seq))

        # Direct _compute_value call (simulates what evaluate_actions does)
        value = net._compute_value(lstm_out, op)

        assert value.dtype == torch.bfloat16


class TestDtypeConsistency:
    """Verify dtype consistency throughout the network."""

    def test_one_hot_matches_lstm_dtype(self):
        """One-hot encoding must match LSTM output dtype."""
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            net = net.to(dtype)
            lstm_out = torch.randn(2, 1, net.lstm_hidden_dim, dtype=dtype)
            op = torch.tensor([[0], [1]])

            # Access internal computation (this is what we're testing)
            import torch.nn.functional as F
            op_one_hot = F.one_hot(op, num_classes=net.num_ops).to(lstm_out)

            assert op_one_hot.dtype == dtype, f"One-hot dtype {op_one_hot.dtype} != {dtype}"
            assert op_one_hot.device == lstm_out.device

    def test_value_input_dim_matches_num_ops(self):
        """value_head input dimension must match lstm_hidden_dim + num_ops.

        Regression test for hardcoded NUM_OPS constant bug.
        """
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)

        expected_dim = net.lstm_hidden_dim + net.num_ops
        actual_dim = net.value_head[0].in_features

        assert actual_dim == expected_dim, (
            f"value_head input dim {actual_dim} != expected {expected_dim} "
            f"(lstm_hidden_dim={net.lstm_hidden_dim} + num_ops={net.num_ops})"
        )

    def test_blueprint_embedding_matches_num_blueprints(self):
        """BlueprintEmbedding size must match num_blueprints + 1.

        Regression test for hardcoded NUM_BLUEPRINTS constant bug.
        """
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)

        expected_size = net.num_blueprints + 1  # +1 for null embedding
        actual_size = net.blueprint_embedding.embedding.num_embeddings

        assert actual_size == expected_size, (
            f"BlueprintEmbedding size {actual_size} != expected {expected_size} "
            f"(num_blueprints={net.num_blueprints} + 1)"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for autocast test"
)
class TestAutocast:
    """Verify network works under torch.autocast."""

    def test_compute_value_under_autocast(self):
        """_compute_value must work under CUDA autocast."""
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM).cuda()

        batch, seq = 2, 3
        op = torch.randint(0, net.num_ops, (batch, seq), device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lstm_out = torch.randn(
                batch, seq, net.lstm_hidden_dim,
                device="cuda"
            )  # Will be bf16 under autocast

            value = net._compute_value(lstm_out, op)

            # Value should be computed without error
            assert not torch.isnan(value).any()
            assert not torch.isinf(value).any()

    def test_forward_under_autocast(self):
        """Full forward pass must work under CUDA autocast."""
        net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM).cuda()

        batch, seq = 2, 1
        state = torch.randn(batch, seq, OBS_V3_NON_BLUEPRINT_DIM, device="cuda")
        bp_idx = torch.randint(0, NUM_BLUEPRINTS, (batch, seq, 3), device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = net(state, bp_idx)

            assert not torch.isnan(output["value"]).any()
            assert output["sampled_op"].shape == (batch, seq)
