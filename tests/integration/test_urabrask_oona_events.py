from __future__ import annotations

import pytest
from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urza import UrzaLibrary
from esper.urabrask.service import produce_and_attach_bsds, produce_bsds_via_crucible
from esper.leyline import leyline_pb2


class _OonaStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def publish_bsds_issued(self, report: leyline_pb2.BSDSIssued) -> bool:  # type: ignore[override]
        self.events.append(("issued", report))
        return True

    def publish_bsds_failed(self, report: leyline_pb2.BSDSFailed) -> bool:  # type: ignore[override]
        self.events.append(("failed", report))
        return True


def _descriptor(bp_id: str) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.3,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="oona events test",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_bsds_issued_event_published_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("URABRASK_OONA_PUBLISH_ENABLED", "true")
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-oona"), artifact, extras={})

    stub = _OonaStub()
    _desc, _bsds = produce_and_attach_bsds(lib, "bp-oona", oona=stub)
    # One issued event captured
    assert any(kind == "issued" for kind, _ in stub.events)
    # Check payload type and blueprint_id
    kinds = [(k, v) for (k, v) in stub.events if k == "issued"]
    assert isinstance(kinds[-1][1], leyline_pb2.BSDSIssued)
    assert kinds[-1][1].bsds.blueprint_id == "bp-oona"


def test_bsds_failed_event_published_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("URABRASK_OONA_PUBLISH_ENABLED", "true")
    root = tmp_path / "urza"
    UrzaLibrary(root=root)  # empty library
    stub = _OonaStub()
    with pytest.raises(ValueError):
        _ = produce_bsds_via_crucible(UrzaLibrary(root=root), "bp-missing", oona=stub)
    assert any(kind == "failed" for kind, _ in stub.events)
    kinds = [(k, v) for (k, v) in stub.events if k == "failed"]
    assert isinstance(kinds[-1][1], leyline_pb2.BSDSFailed)
    assert kinds[-1][1].blueprint_id == "bp-missing"


def test_oona_events_gated_by_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("URABRASK_OONA_PUBLISH_ENABLED", raising=False)
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-oona-gated"), artifact, extras={})
    stub = _OonaStub()
    _ = produce_bsds_via_crucible(lib, "bp-oona-gated", oona=stub)
    assert not stub.events

