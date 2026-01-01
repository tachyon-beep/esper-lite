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


@pytest.mark.parametrize("topology", ["cnn", "transformer"])
def test_param_estimate_accuracy(topology: str):
    """param_estimate should be within 25% of actual param count for canonical dim.

    This test prevents estimate drift where action ordering (sorted by param_estimate)
    becomes misleading about actual compute/memory costs.
    """
    from esper.kasmina.blueprints import BlueprintRegistry

    # Canonical dimensions for each topology
    canonical_dim = {"cnn": 64, "transformer": 384}[topology]
    tolerance = 0.25  # 25% tolerance

    for spec in BlueprintRegistry.list_for_topology(topology):
        if spec.param_estimate == 0:
            # noop blueprints have 0 estimate and 0 actual - skip division
            actual = spec.actual_param_count(canonical_dim)
            assert actual == 0, f"{spec.name}: noop should have 0 params, got {actual}"
            continue

        actual = spec.actual_param_count(canonical_dim)
        estimate = spec.param_estimate

        # Check within tolerance band
        lower = estimate * (1 - tolerance)
        upper = estimate * (1 + tolerance)

        assert lower <= actual <= upper, (
            f"{topology}:{spec.name} param_estimate={estimate:,} but "
            f"actual={actual:,} for dim={canonical_dim} (outside ±{tolerance:.0%} tolerance)"
        )
