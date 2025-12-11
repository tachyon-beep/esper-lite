"""Tests for Kasmina blueprints."""

import pytest
import torch


def test_noop_seed_is_identity():
    """NoopSeed should pass through input unchanged."""
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "noop", dim=64)
    x = torch.randn(2, 64, 8, 8)
    y = seed(x)

    assert torch.allclose(x, y), "NoopSeed should be identity"
    assert sum(p.numel() for p in seed.parameters()) == 0, "NoopSeed should have no params"
