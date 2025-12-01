"""Tests for HostProtocol compliance."""

import pytest
import torch
import torch.nn as nn


def test_host_protocol_is_importable():
    """HostProtocol can be imported."""
    from esper.kasmina.protocol import HostProtocol

    assert HostProtocol is not None


def test_host_protocol_has_required_methods():
    """HostProtocol defines required interface."""
    from esper.kasmina.protocol import HostProtocol

    assert hasattr(HostProtocol, "injection_points")
    assert hasattr(HostProtocol, "register_slot")
    assert hasattr(HostProtocol, "unregister_slot")
    assert hasattr(HostProtocol, "forward")


def test_host_cnn_implements_protocol():
    """HostCNN implements HostProtocol."""
    from esper.kasmina.protocol import HostProtocol
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    assert isinstance(host, HostProtocol)


def test_host_cnn_injection_points():
    """HostCNN declares injection points."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    points = host.injection_points

    assert "block2_post" in points
    assert points["block2_post"] == 64


def test_host_cnn_register_slot():
    """HostCNN can register a slot module."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    slot = torch.nn.Conv2d(64, 64, 1)

    host.register_slot("block2_post", slot)

    assert host.slots["block2_post"] is slot


def test_host_cnn_register_invalid_slot_raises():
    """HostCNN raises ValueError for invalid slot_id."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    slot = torch.nn.Conv2d(64, 64, 1)

    with pytest.raises(ValueError, match="Unknown injection point"):
        host.register_slot("invalid_slot", slot)


def test_host_cnn_unregister_slot():
    """HostCNN can unregister a slot (resets to Identity)."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    slot = torch.nn.Conv2d(64, 64, 1)

    host.register_slot("block2_post", slot)
    host.unregister_slot("block2_post")

    assert isinstance(host.slots["block2_post"], torch.nn.Identity)


def test_host_cnn_forward_with_identity():
    """HostCNN forward works with Identity slots (no-op)."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    x = torch.randn(2, 3, 32, 32)

    out = host(x)

    assert out.shape == (2, 10)


def test_host_cnn_forward_with_slot():
    """HostCNN forward passes through registered slot."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()

    class DoubleSlot(torch.nn.Module):
        def forward(self, x):
            return x * 2

    host.register_slot("block2_post", DoubleSlot())

    x = torch.randn(2, 3, 32, 32)
    out = host(x)

    assert out.shape == (2, 10)


def test_transformer_host_implements_protocol():
    """TransformerHost implements HostProtocol."""
    from esper.kasmina.protocol import HostProtocol
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2)
    assert isinstance(host, HostProtocol)


def test_transformer_host_injection_points():
    """TransformerHost declares injection points per layer."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=4)
    points = host.injection_points

    assert len(points) == 4
    assert "layer_0_post_block" in points
    assert "layer_3_post_block" in points
    assert all(dim == 64 for dim in points.values())


def test_transformer_host_weight_tying():
    """TransformerHost has weight tying between embedding and output."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2)

    assert host.head.weight.data_ptr() == host.tok_emb.weight.data_ptr()


def test_transformer_host_forward():
    """TransformerHost forward produces logits."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2, block_size=32)

    x = torch.randint(0, 1000, (2, 16))
    out = host(x)

    assert out.shape == (2, 16, 1000)


def test_transformer_host_register_unregister():
    """TransformerHost can register and unregister slots."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2)
    slot = torch.nn.Linear(64, 64)

    host.register_slot("layer_0_post_block", slot)
    assert host.slots["layer_0_post_block"] is slot

    host.unregister_slot("layer_0_post_block")
    assert isinstance(host.slots["layer_0_post_block"], torch.nn.Identity)
