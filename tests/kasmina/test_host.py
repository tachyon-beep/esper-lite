"""Tests for Host networks (CNNHost and TransformerHost).

This module tests:
- Segment boundaries and channel mappings
- Forward pass through segments (forward_to_segment, forward_from_segment)
- Multi-slot model creation and forward passes
- Round-trip consistency (full forward vs segmented forward)
"""

import torch
import torch.nn as nn

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel


# =============================================================================
# CNNHost Segment Tests
# =============================================================================


def test_host_segment_channels():
    """CNNHost should expose channel counts at each segment boundary."""
    host = CNNHost()

    # Should expose channels at each injection point (access directly - AttributeError if missing)
    assert host.segment_channels == {
        "r0c0": 32,   # After block1
        "r0c1": 64,     # After block2
        "r0c2": 128,   # After block3
    }


def test_host_forward_segments():
    """CNNHost should support segmented forward pass."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to each segment
    x_early = host.forward_to_segment("r0c0", x)
    assert x_early.shape == (2, 32, 16, 16)  # After block1 + pool

    x_mid = host.forward_to_segment("r0c1", x)
    assert x_mid.shape == (2, 64, 8, 8)  # After block2 + pool

    x_late = host.forward_to_segment("r0c2", x)
    assert x_late.shape == (2, 128, 4, 4)  # After block3 + pool

    # Forward from segment to output
    out = host.forward_from_segment("r0c2", x_late)
    assert out.shape == (2, 10)


def test_forward_from_early_segment():
    """Should be able to forward from early segment through rest of network."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to early, then from early to output
    x_early = host.forward_to_segment("r0c0", x)
    out = host.forward_from_segment("r0c0", x_early)
    assert out.shape == (2, 10)


def test_forward_from_mid_segment():
    """Should be able to forward from mid segment through rest of network."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to mid, then from mid to output
    x_mid = host.forward_to_segment("r0c1", x)
    out = host.forward_from_segment("r0c1", x_mid)
    assert out.shape == (2, 10)


def test_segment_channels_match_injection_points():
    """segment_channels should match the injection_points property."""
    host = CNNHost()

    # Both should expose the same channel information
    assert "block2_post" in host.injection_points
    assert host.injection_points["block2_post"] == host.segment_channels["r0c1"]


# =============================================================================
# Multi-Slot Model Tests
# =============================================================================


def test_multislot_model_creation():
    """MorphogeneticModel should support multiple slots."""
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    assert len(model.seed_slots) == 3
    assert "r0c0" in model.seed_slots
    assert "r0c1" in model.seed_slots
    assert "r0c2" in model.seed_slots

    # Each slot should have correct channels
    assert model.seed_slots["r0c0"].channels == 32
    assert model.seed_slots["r0c1"].channels == 64
    assert model.seed_slots["r0c2"].channels == 128


def test_multislot_forward_pass():
    """Multi-slot model forward should pass through all slots."""
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_multislot_germinate_specific_slot():
    """Should germinate seed in specific slot."""
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    # Germinate in mid slot (use actual blueprint name)
    model.germinate_seed("conv_light", "test_seed", slot="r0c1")

    assert model.seed_slots["r0c1"].is_active
    assert not model.seed_slots["r0c0"].is_active
    assert not model.seed_slots["r0c2"].is_active


# =============================================================================
# TransformerHost Segment Tests
# =============================================================================


class TestTransformerHostSegments:
    """Test TransformerHost segment_channels and segment methods."""

    def test_segment_channels_exists(self):
        """TransformerHost must expose segment_channels attribute."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        assert hasattr(host, "segment_channels")
        assert isinstance(host.segment_channels, dict)
        assert set(host.segment_channels.keys()) == {"r0c0", "r0c1", "r0c2"}

    def test_segment_channels_values(self):
        """All segments should map to n_embd dimension."""
        n_embd = 128
        host = TransformerHost(vocab_size=100, n_embd=n_embd, n_head=4, n_layer=6, block_size=32)
        for segment, dim in host.segment_channels.items():
            assert dim == n_embd, f"Segment {segment} should have dim {n_embd}, got {dim}"

    def test_forward_to_segment_returns_embeddings(self):
        """forward_to_segment should return hidden states at segment boundary."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        x = torch.randint(0, 100, (2, 16))  # batch=2, seq=16

        h = host.forward_to_segment("r0c1", x)
        assert h.shape == (2, 16, 64)  # (batch, seq, n_embd)

    def test_forward_from_segment_returns_logits(self):
        """forward_from_segment should return logits from hidden states."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        h = torch.randn(2, 16, 64)  # (batch, seq, n_embd)

        logits = host.forward_from_segment("r0c1", h)
        assert logits.shape == (2, 16, 100)  # (batch, seq, vocab_size)

    def test_segment_round_trip_matches_forward(self):
        """forward_to_segment + forward_from_segment should match full forward."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        host.eval()  # Disable dropout for deterministic comparison
        x = torch.randint(0, 100, (2, 16))

        # Full forward
        with torch.no_grad():
            full_out = host(x)

        # Segment round-trip through "r0c1"
        with torch.no_grad():
            h = host.forward_to_segment("r0c1", x)
            segment_out = host.forward_from_segment("r0c1", h)

        # Should be identical (deterministic with eval mode)
        torch.testing.assert_close(full_out, segment_out, rtol=1e-5, atol=1e-5)


class TestCNNHostSegments:
    """Test CNNHost segment round-trips respect registered slots."""

    def test_cnn_segment_consistency(self):
        host = CNNHost(num_classes=10, base_channels=8)

        class AddConstant(nn.Module):
            def __init__(self, value: float):
                super().__init__()
                self.value = value

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.as_tensor(self.value, device=x.device, dtype=x.dtype)

        host.register_slot("block2_post", AddConstant(0.1))
        host.register_slot("block3_post", AddConstant(0.2))
        host.eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            full_out = host(x)
            mid_h = host.forward_to_segment("r0c1", x)
            segment_out = host.forward_from_segment("r0c1", mid_h)

            early_h = host.forward_to_segment("r0c0", x)
            late_h_direct = host.forward_to_segment("r0c2", x)
            late_h_from_early = host.forward_to_segment("r0c2", early_h, from_segment="r0c0")
            late_out_from_early = host.forward_from_segment("r0c2", late_h_from_early)

        torch.testing.assert_close(segment_out, full_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(late_h_from_early, late_h_direct, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(late_out_from_early, full_out, rtol=1e-5, atol=1e-5)


# =============================================================================
# TransformerHost with MorphogeneticModel Tests
# =============================================================================


def test_transformer_forward_matches_host():
    """MorphogeneticModel with TransformerHost should match host output when no seeds active."""
    # Create host and model with multiple slots
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, block_size=32, dropout=0.0)
    host.eval()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])
    model.eval()

    # Test input
    x = torch.randint(0, 1000, (2, 16))

    # Forward through both
    with torch.no_grad():
        host_out = host(x)
        model_out = model(x)

    # Should match exactly when no seeds are active
    assert torch.allclose(host_out, model_out, atol=1e-6), \
        f"Outputs differ: max diff = {(host_out - model_out).abs().max().item()}"


def test_transformer_single_slot_forward():
    """MorphogeneticModel with single slot should process all layers."""
    # Create host and model with only mid slot
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, block_size=32, dropout=0.0)
    host.eval()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    model.eval()

    # Test input
    x = torch.randint(0, 1000, (2, 16))

    # Forward through both
    with torch.no_grad():
        host_out = host(x)
        model_out = model(x)

    # Should match exactly - all layers must be processed
    assert torch.allclose(host_out, model_out, atol=1e-6), \
        f"Outputs differ: max diff = {(host_out - model_out).abs().max().item()}"
