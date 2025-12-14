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


def test_transformer_attention_blueprint_validates_heads_divide_dim():
    """Attention blueprint should fail fast on invalid dim/head combos."""
    from esper.kasmina.blueprints import BlueprintRegistry

    with pytest.raises(
        ValueError,
        match=r"dim % n_head == 0",
    ):
        BlueprintRegistry.create("transformer", "attention", dim=10, n_head=3)


def test_transformer_flex_attention_blueprint_validates_heads_divide_dim():
    """FlexAttention blueprint should fail fast on invalid dim/head combos (when available)."""
    from esper.kasmina.blueprints import BlueprintRegistry

    names = {spec.name for spec in BlueprintRegistry.list_for_topology("transformer")}
    if "flex_attention" not in names:
        pytest.skip("flex_attention blueprint not available on this torch build")

    with pytest.raises(
        ValueError,
        match=r"dim % n_head == 0",
    ):
        BlueprintRegistry.create("transformer", "flex_attention", dim=10, n_head=3)
