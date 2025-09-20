from esper.karn import BlueprintMetadata, BlueprintTier, KarnCatalog


def test_karn_catalog_register_and_list() -> None:
    catalog = KarnCatalog()
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
