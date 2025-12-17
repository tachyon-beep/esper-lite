"""Tests for HostProtocol compliance."""

import torch


def test_host_protocol_is_importable():
    """HostProtocol can be imported."""
    from esper.kasmina.protocol import HostProtocol

    assert HostProtocol is not None


def test_host_protocol_has_required_methods():
    """HostProtocol should define required interface methods."""
    from esper.kasmina.protocol import HostProtocol

    assert hasattr(HostProtocol, "injection_points")
    assert hasattr(HostProtocol, "segment_channels")
    assert hasattr(HostProtocol, "forward")
    assert hasattr(HostProtocol, "forward_to_segment")
    assert hasattr(HostProtocol, "forward_from_segment")
    # register_slot/unregister_slot removed - slots managed by MorphogeneticModel


def test_host_cnn_implements_protocol():
    """CNNHost implements HostProtocol."""
    from esper.kasmina.protocol import HostProtocol
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    assert isinstance(host, HostProtocol)


def test_host_cnn_injection_points():
    """CNNHost declares injection points using canonical IDs."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    points = host.injection_points

    # Uses canonical IDs (r0c0, r0c1, r0c2), not internal keys
    assert "r0c0" in points
    assert "r0c1" in points
    assert "r0c2" in points
    assert points["r0c1"] == 64  # Second block output channels


def test_host_cnn_forward():
    """CNNHost forward should work without slots."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    out = host(x)

    assert out.shape == (2, 10)


def test_transformer_host_implements_protocol():
    """TransformerHost implements HostProtocol."""
    from esper.kasmina.protocol import HostProtocol
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=3, num_segments=3)
    assert isinstance(host, HostProtocol)


def test_transformer_host_injection_points():
    """TransformerHost declares injection points using canonical IDs."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, num_segments=3)
    points = host.injection_points

    # Uses canonical IDs (r0c0, r0c1, r0c2), not internal keys
    assert len(points) == 3  # 3 segments
    assert "r0c0" in points
    assert "r0c1" in points
    assert "r0c2" in points
    assert all(dim == 64 for dim in points.values())


def test_transformer_host_weight_tying():
    """TransformerHost has weight tying between embedding and output."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=3, num_segments=3)

    assert host.head.weight.data_ptr() == host.tok_emb.weight.data_ptr()


def test_transformer_host_forward():
    """TransformerHost forward produces logits."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=3, block_size=32, num_segments=3)

    x = torch.randint(0, 1000, (2, 16))
    out = host(x)

    assert out.shape == (2, 16, 1000)
