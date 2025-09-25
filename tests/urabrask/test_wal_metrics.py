from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask import metrics as ura_metrics
from esper.urabrask.service import produce_and_attach_bsds
from esper.urza import UrzaLibrary


def _descriptor(bp_id: str) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.3,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="wal metrics test",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_wal_append_error_increments_counter(tmp_path: Path, monkeypatch) -> None:
    # Enable signing so WAL append is attempted
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    monkeypatch.setenv("URABRASK_SIGNING_ENABLED", "true")
    # Point WAL path to a directory to force append error
    monkeypatch.setenv("URABRASK_WAL_PATH", str(tmp_path))

    before = ura_metrics.snapshot().get("wal.append_errors_total", 0.0)

    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-walerr"), artifact, extras={})

    # Should not raise even if WAL append fails
    _desc, _bsds = produce_and_attach_bsds(lib, "bp-walerr")
    after = ura_metrics.snapshot().get("wal.append_errors_total", 0.0)
    assert after >= before + 1.0
