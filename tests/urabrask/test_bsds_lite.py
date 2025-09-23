from __future__ import annotations

from pathlib import Path

from esper.leyline import leyline_pb2
from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urza import UrzaLibrary
from esper.urabrask import BsdsHeuristicConfig, compute_bsds, produce_and_attach_bsds


def _descriptor(bp_id: str, *, tier: int = BlueprintTier.BLUEPRINT_TIER_SAFE, risk: float = 0.2,
                quarantine_only: bool = False) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=tier,
        risk=risk,
        stage=2,
        quarantine_only=quarantine_only,
        approval_required=False,
        description="bsds-lite tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_compute_bsds_safe_descriptor() -> None:
    proto, js = compute_bsds(_descriptor("bp-safe", risk=0.2))
    assert proto.hazard_band == leyline_pb2.HAZARD_BAND_LOW
    assert proto.handling_class == leyline_pb2.HANDLING_CLASS_STANDARD
    assert 0.19 <= proto.risk_score <= 0.25
    assert js["hazard_band"] == "LOW"
    assert js["handling_class"] == "standard"


def test_compute_bsds_experimental_high() -> None:
    proto, js = compute_bsds(_descriptor("bp-exp", tier=BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL, risk=0.55))
    assert proto.hazard_band in (leyline_pb2.HAZARD_BAND_HIGH, leyline_pb2.HAZARD_BAND_MEDIUM)
    # With default priors, 0.55 + 0.05 -> 0.60 => HIGH
    assert proto.hazard_band == leyline_pb2.HAZARD_BAND_HIGH
    assert js["handling_class"] == "restricted"


def test_compute_bsds_quarantine_override() -> None:
    proto, js = compute_bsds(_descriptor("bp-q", risk=0.2, quarantine_only=True))
    assert proto.handling_class == leyline_pb2.HANDLING_CLASS_QUARANTINE
    assert js["handling_class"] == "quarantine"


def test_compute_bsds_hints_resource_profile() -> None:
    proto, js = compute_bsds(_descriptor("bp-rp"), hints={"resource_profile": "gpu"})
    assert proto.resource_profile == leyline_pb2.RESOURCE_PROFILE_GPU
    assert js["resource_profile"] == "gpu"


def test_produce_and_attach_bsds_in_urza(tmp_path: Path) -> None:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    d = _descriptor("bp-attach", risk=0.4)
    lib.save(d, artifact, extras={})

    descriptor, bsds_json = produce_and_attach_bsds(lib, "bp-attach", hints={"resource_profile": "gpu"})
    rec = lib.get("bp-attach")
    assert rec is not None
    assert rec.extras and "bsds" in rec.extras
    assert rec.extras["bsds"]["resource_profile"] == "gpu"
    assert descriptor.blueprint_id == "bp-attach"

