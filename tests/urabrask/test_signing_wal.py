from __future__ import annotations

import json
from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urza import UrzaLibrary
from esper.urabrask.service import produce_and_attach_bsds
from esper.security.signing import SignatureContext
from esper.urabrask.wal import verify_bsds_signature_in_extras


def _descriptor(bp_id: str, *, risk: float = 0.4) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=risk,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="signing wal tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_bsds_signing_and_wal_append(tmp_path: Path, monkeypatch) -> None:
    # Enable signing and point WAL to a temp file
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    monkeypatch.setenv("URABRASK_SIGNING_ENABLED", "true")
    wal_path = tmp_path / "urabrask_wal.jsonl"
    monkeypatch.setenv("URABRASK_WAL_PATH", str(wal_path))

    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    d = _descriptor("bp-wp8")
    lib.save(d, artifact, extras={})

    # First attach
    _, _ = produce_and_attach_bsds(lib, "bp-wp8")
    rec1 = lib.get("bp-wp8")
    assert rec1 is not None and isinstance(rec1.extras, dict)
    sig1 = rec1.extras.get("bsds_sig")
    assert isinstance(sig1, dict)
    assert sig1.get("algo") == "HMAC-SHA256"
    assert sig1.get("prev_sig", "") == ""
    ctx = SignatureContext(secret=b"test-secret")
    assert verify_bsds_signature_in_extras(rec1.extras, ctx=ctx)

    # Second attach (new signature, prev_sig should link to sig1)
    _, _ = produce_and_attach_bsds(lib, "bp-wp8")
    rec2 = lib.get("bp-wp8")
    assert rec2 is not None and isinstance(rec2.extras, dict)
    sig2 = rec2.extras.get("bsds_sig")
    assert isinstance(sig2, dict)
    assert sig2.get("prev_sig") == sig1.get("sig")
    assert verify_bsds_signature_in_extras(rec2.extras, ctx=ctx)

    # WAL should contain at least two lines with matching sigs
    lines = wal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    last = json.loads(lines[-1])
    assert last["blueprint_id"] == "bp-wp8"
    assert last["sig"] == sig2["sig"]

    # Tamper with extras → verification should fail
    extras = dict(rec2.extras or {})
    bsds = dict(extras.get("bsds") or {})
    bsds["hazard_band"] = "LOW" if str(bsds.get("hazard_band", "")).upper() != "LOW" else "HIGH"
    extras["bsds"] = bsds
    assert not verify_bsds_signature_in_extras(extras, ctx=ctx)

