from __future__ import annotations

import json
from pathlib import Path

import pytest

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask.bsds import compute_bsds
from esper.urza import UrzaLibrary
from esper.urabrask.service import produce_bsds_via_crucible


def _load_schema() -> dict:
    schema_path = Path("docs/prototype-delta/speculative/bsds-lite/schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _validator():  # noqa: ANN001 - third-party
    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"jsonschema not available: {exc}")
    return jsonschema


def test_examples_validate_against_schema() -> None:
    jsonschema = _validator()
    schema = _load_schema()
    examples_dir = Path("docs/prototype-delta/speculative/bsds-lite/examples")
    for path in examples_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)


def _descriptor(bp_id: str, *, risk: float = 0.4) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=risk,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="bsds-schema-producer-tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_producer_mirrors_validate_schema(tmp_path: Path) -> None:
    jsonschema = _validator()
    schema = _load_schema()

    # Heuristic mirror
    proto, mirror = compute_bsds(_descriptor("bp-schem-heur", risk=0.3))
    assert proto.blueprint_id == "bp-schem-heur"
    jsonschema.validate(instance=mirror, schema=schema)

    # Crucible mirror via service extras
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-schem-crucible", risk=0.35), artifact, extras={})

    _ = produce_bsds_via_crucible(lib, "bp-schem-crucible")
    rec = lib.get("bp-schem-crucible")
    assert rec is not None and isinstance(rec.extras, dict)
    mirror2 = rec.extras.get("bsds")
    assert isinstance(mirror2, dict)
    jsonschema.validate(instance=mirror2, schema=schema)

