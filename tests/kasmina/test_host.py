"""Tests for CNNHost segment boundaries for multi-slot support."""

import torch
import pytest


def test_host_segment_channels():
    """CNNHost should expose channel counts at each segment boundary."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()

    # Should expose channels at each injection point (access directly - AttributeError if missing)
    assert host.segment_channels == {
        "early": 32,   # After block1
        "mid": 64,     # After block2
        "late": 128,   # After block3
    }


def test_host_forward_segments():
    """CNNHost should support segmented forward pass."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to each segment
    x_early = host.forward_to_segment("early", x)
    assert x_early.shape == (2, 32, 16, 16)  # After block1 + pool

    x_mid = host.forward_to_segment("mid", x)
    assert x_mid.shape == (2, 64, 8, 8)  # After block2 + pool

    x_late = host.forward_to_segment("late", x)
    assert x_late.shape == (2, 128, 4, 4)  # After block3 + pool

    # Forward from segment to output
    out = host.forward_from_segment("late", x_late)
    assert out.shape == (2, 10)


def test_forward_from_early_segment():
    """Should be able to forward from early segment through rest of network."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to early, then from early to output
    x_early = host.forward_to_segment("early", x)
    out = host.forward_from_segment("early", x_early)
    assert out.shape == (2, 10)


def test_forward_from_mid_segment():
    """Should be able to forward from mid segment through rest of network."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to mid, then from mid to output
    x_mid = host.forward_to_segment("mid", x)
    out = host.forward_from_segment("mid", x_mid)
    assert out.shape == (2, 10)


def test_segment_channels_match_injection_points():
    """segment_channels should match the injection_points property."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()

    # Both should expose the same channel information
    assert "block2_post" in host.injection_points
    assert host.injection_points["block2_post"] == host.segment_channels["mid"]
