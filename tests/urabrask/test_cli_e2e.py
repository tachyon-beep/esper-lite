from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
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
        description="cli e2e tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def _run_cli(urza_root: Path, blueprint_id: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Ensure the src/ path is importable for `python -m ...`
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(repo_root / "src")
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable,
        "-m",
        "esper.urabrask.cli",
        "--urza-root",
        str(urza_root),
        "--blueprint-id",
        blueprint_id,
        "--resource-profile",
        "gpu",
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)


def test_cli_produces_bsds_and_updates_extras(tmp_path: Path) -> None:
    # Arrange: Urza with one blueprint
    urza_root = tmp_path / "urza"
    lib = UrzaLibrary(root=urza_root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-cli"), artifact, extras={})

    # Act: run CLI
    proc = _run_cli(urza_root, "bp-cli")
    assert proc.returncode == 0, proc.stderr

    # Parse output JSON and check blueprint_id is present
    data = json.loads(proc.stdout)
    assert data.get("blueprint_id") == "bp-cli"

    # Extras should contain bsds
    # Reopen Urza to hydrate from DB (CLI runs in a separate process)
    lib2 = UrzaLibrary(root=urza_root)
    rec = lib2.get("bp-cli")
    assert rec is not None and isinstance(rec.extras, dict)
    assert "bsds" in rec.extras


def test_cli_with_signing_enabled_attaches_signature(tmp_path: Path, monkeypatch) -> None:
    # Arrange: enable signing
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    monkeypatch.setenv("URABRASK_SIGNING_ENABLED", "true")
    urza_root = tmp_path / "urza"
    lib = UrzaLibrary(root=urza_root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-cli-signed"), artifact, extras={})

    # Act
    proc = _run_cli(urza_root, "bp-cli-signed")
    assert proc.returncode == 0, proc.stderr

    # Verify signature block present in extras
    # Reopen Urza to hydrate from DB
    lib2 = UrzaLibrary(root=urza_root)
    rec = lib2.get("bp-cli-signed")
    assert rec is not None and isinstance(rec.extras, dict)
    sig = rec.extras.get("bsds_sig")
    assert isinstance(sig, dict)
    assert sig.get("algo") == "HMAC-SHA256"
    assert sig.get("sig")


def test_cli_missing_blueprint_returns_error(tmp_path: Path) -> None:
    urza_root = tmp_path / "urza"
    UrzaLibrary(root=urza_root)  # initialize empty Urza
    proc = _run_cli(urza_root, "bp-missing")
    assert proc.returncode != 0
    # stderr should contain some error indication
    assert proc.stderr
