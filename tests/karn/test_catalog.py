from esper.karn import BlueprintMetadata, BlueprintTier, KarnCatalog


def test_karn_catalog_register_and_list() -> None:
    catalog = KarnCatalog(load_defaults=False)
    metadata = BlueprintMetadata(
        blueprint_id="bp-1",
        name="Test Blueprint",
        tier=BlueprintTier.SAFE,
        description="Test",
        allowed_parameters={"alpha": (0.0, 1.0)},
    )
    catalog.register(metadata)
    assert catalog.get("bp-1") == metadata
    safe = list(catalog.list_by_tier(BlueprintTier.SAFE))
    assert safe and safe[0].blueprint_id == "bp-1"


def test_default_catalog_contains_expected_counts() -> None:
    catalog = KarnCatalog()
    assert len(catalog) == 50
    safe = list(catalog.list_by_tier(BlueprintTier.SAFE))
    experimental = list(catalog.list_by_tier(BlueprintTier.EXPERIMENTAL))
    adversarial = list(catalog.list_by_tier(BlueprintTier.HIGH_RISK))
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
    assert safe_conservative.tier is BlueprintTier.SAFE
    experimental = catalog.choose_template(tier=BlueprintTier.EXPERIMENTAL, context="exp")
    assert experimental.tier is BlueprintTier.EXPERIMENTAL
