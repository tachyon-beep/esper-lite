from __future__ import annotations

import json
from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask.crucible import run_crucible_v1, CrucibleConfigV1


def _descriptor(bp_id: str, *, risk: float = 0.2) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=risk,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="crucible artifacts tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_crucible_artifact_written_and_retained(tmp_path: Path, monkeypatch) -> None:
    # Configure artifacts dir and retention
    art_dir = tmp_path / "artifacts"
    monkeypatch.setenv("URABRASK_CRUCIBLE_ARTIFACTS_DIR", str(art_dir))
    monkeypatch.setenv("URABRASK_CRUCIBLE_ARTIFACTS_KEEP", "2")

    # Run three times to trigger retention trimming
    for i in range(3):
        bsds, hazards = run_crucible_v1(_descriptor("bp-art", risk=0.1 * (i + 1)), config=CrucibleConfigV1())
        assert bsds.blueprint_id == "bp-art"

    bundle_dir = art_dir / "bp-art"
    assert bundle_dir.exists()
    files = sorted(bundle_dir.glob("*.json"))
    assert len(files) == 2  # retained
    # Validate content shape of the latest
    latest = files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    for key in ("artifact_version", "crucible_version", "blueprint_id", "bsds", "hazards", "timings", "config"):
        assert key in payload
    bsds_block = payload["bsds"]
    for key in ("risk_score", "hazard_band", "handling_class", "resource_profile", "provenance", "issued_at"):
        assert key in bsds_block

