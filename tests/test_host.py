"""Host network tests."""

import torch
import torch.nn as nn

from esper.kasmina.host import CNNHost, TransformerHost


class TestTransformerHostSegments:
    """Test TransformerHost segment_channels and segment methods."""

    def test_segment_channels_exists(self):
        """TransformerHost must expose segment_channels attribute."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        assert hasattr(host, "segment_channels")
        assert isinstance(host.segment_channels, dict)
        assert set(host.segment_channels.keys()) == {"early", "mid", "late"}

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

        h = host.forward_to_segment("mid", x)
        assert h.shape == (2, 16, 64)  # (batch, seq, n_embd)

    def test_forward_from_segment_returns_logits(self):
        """forward_from_segment should return logits from hidden states."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        h = torch.randn(2, 16, 64)  # (batch, seq, n_embd)

        logits = host.forward_from_segment("mid", h)
        assert logits.shape == (2, 16, 100)  # (batch, seq, vocab_size)

    def test_segment_round_trip_matches_forward(self):
        """forward_to_segment + forward_from_segment should match full forward."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        host.eval()  # Disable dropout for deterministic comparison
        x = torch.randint(0, 100, (2, 16))

        # Full forward
        with torch.no_grad():
            full_out = host(x)

        # Segment round-trip through "mid"
        with torch.no_grad():
            h = host.forward_to_segment("mid", x)
            segment_out = host.forward_from_segment("mid", h)

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
            mid_h = host.forward_to_segment("mid", x)
            segment_out = host.forward_from_segment("mid", mid_h)

            early_h = host.forward_to_segment("early", x)
            late_h_direct = host.forward_to_segment("late", x)
            late_h_from_early = host.forward_to_segment("late", early_h, from_segment="early")
            late_out_from_early = host.forward_from_segment("late", late_h_from_early)

        torch.testing.assert_close(segment_out, full_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(late_h_from_early, late_h_direct, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(late_out_from_early, full_out, rtol=1e-5, atol=1e-5)
