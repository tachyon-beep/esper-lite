"""Property-based tests for Kasmina blending algorithm invariants.

These tests use Hypothesis to verify critical invariants hold for GatedBlend:

1. GatedBlend gate output is bounded in [0, 1]
2. GatedBlend reaches alpha >= 0.95 at total_steps
3. GatedBlend produces correct tensor shapes for CNN and Transformer topologies
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.kasmina.blending import GatedBlend, BlendCatalog

from tests.strategies import channel_dimensions


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


class TestBlendAlphaForBlendTensor:
    """get_alpha_for_blend() should return proper tensor shapes."""

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
        assert "gated" in algorithms

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
