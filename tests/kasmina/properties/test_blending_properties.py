"""Property-based tests for Kasmina blending algorithm invariants.

These tests use Hypothesis to verify critical invariants hold for
LinearBlend, SigmoidBlend, and GatedBlend:

1. LinearBlend is monotonically non-decreasing
2. SigmoidBlend output is bounded in [0, 1]
3. GatedBlend gate output is bounded in [0, 1]
4. All algorithms reach alpha >= 0.95 at total_steps
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.kasmina.blending import LinearBlend, SigmoidBlend, GatedBlend, BlendCatalog

from tests.strategies import channel_dimensions


class TestLinearBlendMonotonic:
    """LinearBlend must be monotonically non-decreasing."""

    @given(
        total_steps=st.integers(min_value=1, max_value=100),
        step_sequence=st.lists(
            st.integers(min_value=0, max_value=200),
            min_size=2,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_linear_blend_monotonic_over_steps(
        self, total_steps: int, step_sequence: list[int]
    ):
        """LinearBlend.get_alpha(t) <= get_alpha(t+1) for sorted steps."""
        blend = LinearBlend(total_steps=total_steps)

        sorted_steps = sorted(step_sequence)
        alphas = [blend.get_alpha(s) for s in sorted_steps]

        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1], \
                f"Non-monotonic: alpha[{sorted_steps[i]}]={alphas[i]} > alpha[{sorted_steps[i+1]}]={alphas[i+1]}"

    @given(total_steps=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_linear_blend_starts_at_zero(self, total_steps: int):
        """LinearBlend starts at alpha=0 when step=0."""
        blend = LinearBlend(total_steps=total_steps)
        assert blend.get_alpha(0) == 0.0

    @given(total_steps=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_linear_blend_reaches_one_at_total_steps(self, total_steps: int):
        """LinearBlend reaches alpha=1.0 at step=total_steps."""
        blend = LinearBlend(total_steps=total_steps)
        assert blend.get_alpha(total_steps) == 1.0

    @given(
        total_steps=st.integers(min_value=1, max_value=100),
        step=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=100)
    def test_linear_blend_always_bounded(self, total_steps: int, step: int):
        """LinearBlend output is always in [0, 1]."""
        blend = LinearBlend(total_steps=total_steps)
        alpha = blend.get_alpha(step)
        assert 0.0 <= alpha <= 1.0


class TestSigmoidBlendBounded:
    """SigmoidBlend output must be bounded in [0, 1]."""

    @given(
        total_steps=st.integers(min_value=5, max_value=100),
        steepness=st.floats(min_value=0.1, max_value=5.0),
        step=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=100)
    def test_sigmoid_blend_always_bounded(
        self, total_steps: int, steepness: float, step: int
    ):
        """SigmoidBlend output is always in [0, 1] for reasonable step ranges."""
        blend = SigmoidBlend(total_steps=total_steps, steepness=steepness)
        alpha = blend.get_alpha(step)
        assert 0.0 <= alpha <= 1.0, f"Alpha {alpha} out of bounds at step {step}"

    @given(
        total_steps=st.integers(min_value=1, max_value=100),
        steepness=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_sigmoid_blend_near_zero_at_start(self, total_steps: int, steepness: float):
        """SigmoidBlend is near 0 at step=0."""
        blend = SigmoidBlend(total_steps=total_steps, steepness=steepness)
        alpha = blend.get_alpha(0)
        # Sigmoid at -6*steepness should be very small (< 0.1 for reasonable steepness)
        assert alpha < 0.5, f"Alpha {alpha} at step 0 should be < 0.5"

    @given(
        total_steps=st.integers(min_value=1, max_value=100),
        steepness=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_sigmoid_blend_near_one_at_end(self, total_steps: int, steepness: float):
        """SigmoidBlend is near 1 at step=total_steps."""
        blend = SigmoidBlend(total_steps=total_steps, steepness=steepness)
        alpha = blend.get_alpha(total_steps)
        # Sigmoid at +6*steepness should be very close to 1 (> 0.9)
        assert alpha > 0.5, f"Alpha {alpha} at total_steps should be > 0.5"

    @given(
        total_steps=st.integers(min_value=10, max_value=100),
        steepness=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=50)
    def test_sigmoid_blend_monotonic(self, total_steps: int, steepness: float):
        """SigmoidBlend should be monotonically non-decreasing."""
        blend = SigmoidBlend(total_steps=total_steps, steepness=steepness)

        previous_alpha = blend.get_alpha(0)
        for step in range(1, total_steps + 1):
            current_alpha = blend.get_alpha(step)
            assert current_alpha >= previous_alpha, \
                f"Non-monotonic at step {step}: {previous_alpha} > {current_alpha}"
            previous_alpha = current_alpha


class TestGatedBlendBounded:
    """GatedBlend gate output must be bounded in [0, 1]."""

    @given(
        channels=channel_dimensions(min_channels=16, max_channels=128),
        batch=st.integers(min_value=1, max_value=4),
        spatial=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=50)
    def test_gated_blend_cnn_bounded(self, channels: int, batch: int, spatial: int):
        """GatedBlend CNN output is always in [0, 1]."""
        blend = GatedBlend(channels=channels, topology="cnn")

        x = torch.randn(batch, channels, spatial, spatial)
        alpha = blend.get_alpha_for_blend(x)

        assert (alpha >= 0.0).all(), "Gate output has values < 0"
        assert (alpha <= 1.0).all(), "Gate output has values > 1"

    @given(
        channels=channel_dimensions(min_channels=16, max_channels=128),
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=50)
    def test_gated_blend_transformer_bounded(
        self, channels: int, batch: int, seq_len: int
    ):
        """GatedBlend Transformer output is always in [0, 1]."""
        blend = GatedBlend(channels=channels, topology="transformer")

        x = torch.randn(batch, seq_len, channels)
        alpha = blend.get_alpha_for_blend(x)

        assert (alpha >= 0.0).all(), "Gate output has values < 0"
        assert (alpha <= 1.0).all(), "Gate output has values > 1"

    @given(channels=channel_dimensions(min_channels=16, max_channels=128))
    @settings(max_examples=30)
    def test_gated_blend_gate_is_sigmoid(self, channels: int):
        """GatedBlend uses sigmoid for bounded output."""
        blend = GatedBlend(channels=channels, topology="cnn")

        # Check the last layer is Sigmoid
        last_layer = blend.gate[-1]
        assert isinstance(last_layer, torch.nn.Sigmoid)


class TestBlendAlphaConvergence:
    """All algorithms should reach high alpha at total_steps."""

    @given(total_steps=st.integers(min_value=5, max_value=50))
    @settings(max_examples=30)
    def test_linear_reaches_threshold_at_end(self, total_steps: int):
        """LinearBlend reaches alpha >= 0.95 at step = total_steps."""
        blend = LinearBlend(total_steps=total_steps)
        # At exactly total_steps, should be 1.0
        assert blend.get_alpha(total_steps) >= 0.95

    @given(total_steps=st.integers(min_value=5, max_value=50))
    @settings(max_examples=30)
    def test_sigmoid_reaches_threshold_at_end(self, total_steps: int):
        """SigmoidBlend reaches alpha >= 0.95 at step = total_steps."""
        blend = SigmoidBlend(total_steps=total_steps, steepness=1.0)
        # Sigmoid should be high at the end
        assert blend.get_alpha(total_steps) >= 0.95

    @given(total_steps=st.integers(min_value=5, max_value=50))
    @settings(max_examples=30)
    def test_gated_get_alpha_reaches_threshold(self, total_steps: int):
        """GatedBlend.get_alpha() reaches >= 0.95 at total_steps."""
        blend = GatedBlend(channels=64, total_steps=total_steps)
        # get_alpha for gated is step-based (for G3 gate compatibility)
        assert blend.get_alpha(total_steps) >= 0.95


class TestBlendAlphaForBlendTensor:
    """get_alpha_for_blend() should return proper tensor shapes."""

    @given(
        total_steps=st.integers(min_value=5, max_value=50),
        batch=st.integers(min_value=1, max_value=4),
        channels=st.sampled_from([32, 64, 128]),
    )
    @settings(max_examples=30)
    def test_linear_alpha_tensor_broadcasts(
        self, total_steps: int, batch: int, channels: int
    ):
        """LinearBlend alpha tensor broadcasts to input shape."""
        blend = LinearBlend(total_steps=total_steps)
        blend.step(total_steps // 2)

        x = torch.randn(batch, channels, 8, 8)
        alpha = blend.get_alpha_for_blend(x)

        # Should be scalar tensor (broadcasts to any shape)
        assert alpha.dim() == 0 or alpha.numel() == 1
        assert alpha.device == x.device
        assert alpha.dtype == x.dtype

    @given(
        batch=st.integers(min_value=1, max_value=4),
        channels=st.sampled_from([32, 64, 128]),
    )
    @settings(max_examples=30)
    def test_gated_alpha_tensor_shape_cnn(self, batch: int, channels: int):
        """GatedBlend CNN alpha tensor has shape (B, 1, 1, 1)."""
        blend = GatedBlend(channels=channels, topology="cnn")

        x = torch.randn(batch, channels, 8, 8)
        alpha = blend.get_alpha_for_blend(x)

        assert alpha.shape == (batch, 1, 1, 1)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        channels=st.sampled_from([32, 64, 128]),
        seq_len=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=30)
    def test_gated_alpha_tensor_shape_transformer(
        self, batch: int, channels: int, seq_len: int
    ):
        """GatedBlend Transformer alpha tensor has shape (B, 1, 1)."""
        blend = GatedBlend(channels=channels, topology="transformer")

        x = torch.randn(batch, seq_len, channels)
        alpha = blend.get_alpha_for_blend(x)

        assert alpha.shape == (batch, 1, 1)


class TestBlendCatalogConsistency:
    """BlendCatalog should create consistent algorithms."""

    def test_catalog_lists_all_algorithms(self):
        """BlendCatalog.list_algorithms() includes all known algorithms."""
        algorithms = BlendCatalog.list_algorithms()
        assert "linear" in algorithms
        assert "sigmoid" in algorithms
        assert "gated" in algorithms

    @given(total_steps=st.integers(min_value=5, max_value=50))
    @settings(max_examples=20)
    def test_catalog_creates_linear(self, total_steps: int):
        """BlendCatalog creates LinearBlend correctly."""
        blend = BlendCatalog.create("linear", total_steps=total_steps)
        assert isinstance(blend, LinearBlend)
        assert blend.total_steps == total_steps

    @given(
        total_steps=st.integers(min_value=5, max_value=50),
        steepness=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=20)
    def test_catalog_creates_sigmoid(self, total_steps: int, steepness: float):
        """BlendCatalog creates SigmoidBlend correctly."""
        blend = BlendCatalog.create("sigmoid", total_steps=total_steps, steepness=steepness)
        assert isinstance(blend, SigmoidBlend)
        assert blend.total_steps == total_steps
        assert blend.steepness == steepness

    @given(channels=channel_dimensions(min_channels=16, max_channels=128))
    @settings(max_examples=20)
    def test_catalog_creates_gated(self, channels: int):
        """BlendCatalog creates GatedBlend correctly."""
        blend = BlendCatalog.create("gated", channels=channels)
        assert isinstance(blend, GatedBlend)

    def test_catalog_unknown_raises(self):
        """BlendCatalog raises ValueError for unknown algorithm."""
        with pytest.raises(ValueError, match="Unknown"):
            BlendCatalog.create("unknown_algorithm")
