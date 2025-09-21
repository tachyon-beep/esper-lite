from esper.karn import (
    BlueprintDescriptor,
    BlueprintTier,
    KarnCatalog,
)


def _descriptor() -> BlueprintDescriptor:
    descriptor = BlueprintDescriptor(
        blueprint_id="bp-1",
        name="Test Blueprint",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.2,
        stage=1,
        description="Test",
    )
    bounds = descriptor.allowed_parameters["alpha"]
    bounds.min_value = 0.0
    bounds.max_value = 1.0
    return descriptor


def test_karn_catalog_register_and_list() -> None:
    catalog = KarnCatalog(load_defaults=False)
    metadata = _descriptor()
    catalog.register(metadata)
    assert catalog.get("bp-1") == metadata
    safe = list(catalog.list_by_tier(BlueprintTier.BLUEPRINT_TIER_SAFE))
    assert safe and safe[0].blueprint_id == "bp-1"


def test_default_catalog_contains_expected_counts() -> None:
    catalog = KarnCatalog()
    assert len(catalog) == 50
    safe = list(catalog.list_by_tier(BlueprintTier.BLUEPRINT_TIER_SAFE))
    experimental = list(
        catalog.list_by_tier(BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL)
    )
    adversarial = list(
        catalog.list_by_tier(BlueprintTier.BLUEPRINT_TIER_HIGH_RISK)
    )
    assert len(safe) == 35
    assert len(experimental) == 7
    assert len(adversarial) == 8
    assert safe[0].stage == 0
    assert safe[0].risk == 0.0


def test_choose_template_deterministic_by_context() -> None:
    catalog = KarnCatalog()
    first = catalog.choose_template(context="control-run")
    second = catalog.choose_template(context="control-run")
    assert first.blueprint_id == second.blueprint_id
    safe_conservative = catalog.choose_template(conservative=True)
    assert safe_conservative.tier == BlueprintTier.BLUEPRINT_TIER_SAFE
    experimental = catalog.choose_template(
        tier=BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL, context="exp"
    )
    assert experimental.tier == BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL
