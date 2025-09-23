from __future__ import annotations

import os
import time
from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urza import UrzaLibrary


def _descriptor(bp_id: str) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.2,
        stage=1,
        quarantine_only=False,
        approval_required=False,
        description="bsds test",
    )
    # Provide one allowed parameter so guard_spec fallback works in some flows
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_urza_bsds_extras_round_trip_persist_reload(tmp_path: Path) -> None:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")

    bsds = {
        "risk_score": 0.82,
        "hazard_band": "HIGH",
        "handling_class": "restricted",
        "resource_profile": "gpu",
        "provenance": "HEURISTIC",
    }
    lib.save(_descriptor("bp-bsds"), artifact, extras={"bsds": bsds})

    # New instance simulates process restart; should hydrate from DB
    lib2 = UrzaLibrary(root=root)
    rec = lib2.get("bp-bsds")
    assert rec is not None
    assert "bsds" in (rec.extras or {})
    assert rec.extras["bsds"]["hazard_band"].upper() == "HIGH"
    assert float(rec.extras["bsds"]["risk_score"]) == 0.82


def test_urza_bsds_extras_ttl_eviction(tmp_path: Path) -> None:
    root = tmp_path / "urza"
    # Short TTL so old mtime triggers eviction
    lib = UrzaLibrary(root=root, cache_ttl_seconds=1)
    artifact = tmp_path / "artifact2.pt"
    artifact.write_bytes(b"dummy2")
    lib.save(_descriptor("bp-expire"), artifact, extras={"bsds": {"hazard_band": "LOW"}})

    rec = lib.get("bp-expire")
    assert rec is not None

    # Force mtime to the distant past so TTL check expires the record
    past = time.time() - 3600
    os.utime(rec.artifact_path, (past, past))

    # Second get should detect expiration and evict; returns None
    gone = lib.get("bp-expire")
    assert gone is None

    # After eviction the record is removed (DB row deleted), list_all is empty
    all_now = lib.list_all()
    assert "bp-expire" not in all_now

