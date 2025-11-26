from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask.service import produce_bsds_via_crucible
from esper.urza import UrzaLibrary


def _descriptor(bp_id: str, *, risk: float = 0.4) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=risk,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="crucible service tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_produce_bsds_via_crucible_attaches_to_urza(tmp_path: Path) -> None:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    d = _descriptor("bp-crucible")
    lib.save(d, artifact, extras={})

    bsds = produce_bsds_via_crucible(lib, "bp-crucible", hints={"resource_profile": "gpu"})
    assert bsds.blueprint_id == "bp-crucible"
    rec = lib.get("bp-crucible")
    assert rec is not None and "bsds" in (rec.extras or {})
    assert rec.extras["bsds"]["resource_profile"] == "gpu"
    assert "hazards" in rec.extras["bsds"]
