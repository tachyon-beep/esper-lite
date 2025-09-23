from __future__ import annotations

from esper.leyline import leyline_pb2
from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask.crucible import run_crucible, CrucibleConfig


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
        description="crucible tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_run_crucible_safe_descriptor_low_hazard() -> None:
    bsds = run_crucible(_descriptor("bp-safe", risk=0.2))
    assert bsds.hazard_band == leyline_pb2.HAZARD_BAND_LOW
    assert bsds.provenance == leyline_pb2.PROVENANCE_URABRASK


def test_run_crucible_experimental_bumps_to_high() -> None:
    bsds = run_crucible(_descriptor("bp-exp", tier=BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL, risk=0.55), config=CrucibleConfig())
    assert bsds.hazard_band == leyline_pb2.HAZARD_BAND_HIGH
    assert bsds.handling_class == leyline_pb2.HANDLING_CLASS_RESTRICTED


def test_run_crucible_quarantine_override_critical() -> None:
    bsds = run_crucible(_descriptor("bp-q", risk=0.2, quarantine_only=True))
    assert bsds.handling_class == leyline_pb2.HANDLING_CLASS_QUARANTINE
    assert bsds.hazard_band in (leyline_pb2.HAZARD_BAND_HIGH, leyline_pb2.HAZARD_BAND_CRITICAL)

