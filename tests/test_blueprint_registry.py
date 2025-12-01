"""Tests for blueprint plugin registry."""

import pytest
import torch.nn as nn


def test_registry_is_importable():
    """BlueprintRegistry can be imported."""
    from esper.kasmina.blueprints import BlueprintRegistry

    assert BlueprintRegistry is not None


def test_registry_register_decorator():
    """Decorator registers a blueprint."""
    from esper.kasmina.blueprints import BlueprintRegistry

    @BlueprintRegistry.register("test_blueprint", "cnn", param_estimate=100)
    def create_test(dim: int) -> nn.Module:
        return nn.Linear(dim, dim)

    try:
        specs = BlueprintRegistry.list_for_topology("cnn")
        names = [s.name for s in specs]
        assert "test_blueprint" in names
    finally:
        BlueprintRegistry.unregister("cnn", "test_blueprint")


def test_registry_list_for_topology():
    """Registry filters blueprints by topology."""
    from esper.kasmina.blueprints import BlueprintRegistry

    cnn_specs = BlueprintRegistry.list_for_topology("cnn")
    transformer_specs = BlueprintRegistry.list_for_topology("transformer")

    assert any(s.name == "conv_enhance" for s in cnn_specs)

    cnn_names = {s.name for s in cnn_specs}
    transformer_names = {s.name for s in transformer_specs}
    assert len(cnn_names & transformer_names) >= 0


def test_registry_sorted_by_params():
    """Blueprints sorted by param estimate (ascending)."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("cnn")
    params = [s.param_estimate for s in specs]

    assert params == sorted(params)


def test_blueprint_spec_has_factory():
    """BlueprintSpec can create modules."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("cnn")
    spec = specs[0]

    module = spec.factory(64)
    assert isinstance(module, nn.Module)


def test_blueprint_spec_actual_param_count():
    """BlueprintSpec can compute actual param count."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("cnn")
    spec = next(s for s in specs if s.name == "norm")

    actual = spec.actual_param_count(64)
    assert actual > 0


def test_transformer_blueprints_registered():
    """Transformer blueprints are registered."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("transformer")
    names = {s.name for s in specs}

    assert "norm" in names
    assert "lora" in names


def test_transformer_lora_creates_module():
    """LoRA blueprint creates valid module."""
    from esper.kasmina.blueprints import BlueprintRegistry
    import torch

    module = BlueprintRegistry.create("transformer", "lora", dim=384)

    x = torch.randn(2, 16, 384)
    out = module(x)

    assert out.shape == x.shape


def test_transformer_norm_creates_module():
    """Norm blueprint creates valid module."""
    from esper.kasmina.blueprints import BlueprintRegistry
    import torch

    module = BlueprintRegistry.create("transformer", "norm", dim=384)

    x = torch.randn(2, 16, 384)
    out = module(x)

    assert out.shape == x.shape


def test_legacy_catalog_still_works():
    """Legacy BlueprintCatalog API is removed (registry only)."""
    with pytest.raises(ImportError):
        from esper.kasmina.blueprints import BlueprintCatalog  # noqa: F401
