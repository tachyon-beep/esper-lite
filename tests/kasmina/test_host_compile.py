"""Test host networks are torch.compile compatible."""
import pytest
import torch

from esper.kasmina.host import TransformerHost


class TestTransformerHostCompile:
    """Verify TransformerHost compiles without graph breaks from assertions."""

    def test_forward_no_graph_break_from_assert(self):
        """TransformerHost.forward should not have assertion graph breaks."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=2, block_size=32)

        # This would cause graph break if assert is present
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randint(0, 100, (2, 16))
        result = compiled_host(x)

        assert result.shape == (2, 16, 100)

    def test_sequence_length_validation_still_works(self):
        """Sequence length > block_size should still raise error."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=2, block_size=32)

        x = torch.randint(0, 100, (2, 64))  # 64 > 32 block_size

        with pytest.raises(ValueError, match="exceeds block_size"):
            host(x)
