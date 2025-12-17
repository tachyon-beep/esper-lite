"""Test host networks are torch.compile compatible."""
import pytest
import torch

from esper.kasmina.host import CNNHost, TransformerHost


class TestTransformerHostCompile:
    """Verify TransformerHost compiles without graph breaks from assertions."""

    def test_forward_no_graph_break_from_assert(self):
        """TransformerHost.forward should not have assertion graph breaks."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=3, block_size=32, num_segments=3)

        # This would cause graph break if assert is present
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randint(0, 100, (2, 16))
        result = compiled_host(x)

        assert result.shape == (2, 16, 100)

    def test_sequence_length_validation_still_works(self):
        """Sequence length > block_size should still raise error."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=3, block_size=32, num_segments=3)

        x = torch.randint(0, 100, (2, 64))  # 64 > 32 block_size

        with pytest.raises(ValueError, match="exceeds block_size"):
            host(x)


class TestCNNHostCompile:
    """Verify CNNHost compiles efficiently."""

    def test_forward_uses_precomputed_keys(self):
        """CNNHost should not format strings in forward loop."""
        host = CNNHost(num_classes=10, n_blocks=3)

        # Should compile without string formatting graph breaks
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randn(2, 3, 32, 32)
        result = compiled_host(x)

        assert result.shape == (2, 10)

    def test_segment_routing_compiles(self):
        """Verify segment routing methods compile without graph breaks."""
        host = CNNHost(num_classes=10, n_blocks=3)

        x = torch.randn(2, 3, 32, 32)
        # Segment routing should work without errors
        features = host.forward_to_segment("r0c1", x)
        assert features.shape[0] == 2

        out = host.forward_from_segment("r0c1", features)
        assert out.shape == (2, 10)
