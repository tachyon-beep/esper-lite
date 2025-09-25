"""Urabrask WAL and signature verification helpers (ops tools).

Provides utilities to verify the append-only WAL hash chain and to validate
BSDS signatures stored in Urza extras for a given blueprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext, sign


def _canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(slots=True)
class WalVerificationResult:
    ok: bool
    checked: int
    by_blueprint: dict[str, dict[str, Any]]


def wal_verify(
    wal_path: Path,
    *,
    verify_signatures: bool = True,
    secret: bytes | None = None,
) -> WalVerificationResult:
    """Verify the Urabrask WAL hash chain, optionally checking signatures.

    Returns a structured result per blueprint with any issues encountered.
    """

    by_bp: dict[str, dict[str, Any]] = {}
    count = 0
    if not wal_path.exists() or not wal_path.is_file():
        return WalVerificationResult(ok=True, checked=0, by_blueprint={})

    ctx: SignatureContext | None = None
    if verify_signatures:
        try:
            if secret is not None:
                ctx = SignatureContext(secret=secret)
            else:
                ctx = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
        except Exception:
            ctx = None

    last_sig_by_bp: dict[str, str] = {}
    errors = 0
    for line in wal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        count += 1
        try:
            entry = json.loads(line)
            bp = str(entry.get("blueprint_id", ""))
            sig = str(entry.get("sig", ""))
            prev_sig = str(entry.get("prev_sig", ""))
            payload = entry.get("payload")
        except Exception:  # malformed line
            errors += 1
            continue
        bprec = by_bp.setdefault(bp or "", {"ok": True, "length": 0, "errors": []})
        bprec["length"] = int(bprec.get("length", 0)) + 1
        # Check prev_sig chaining per blueprint
        last_sig = last_sig_by_bp.get(bp, "")
        if last_sig and prev_sig != last_sig:
            bprec["ok"] = False
            bprec["errors"].append("prev_sig_mismatch")
            errors += 1
        last_sig_by_bp[bp] = sig
        # Optional signature verification
        if verify_signatures and ctx is not None and payload is not None:
            try:
                expected = sign(_canonical_dumps(payload).encode("utf-8"), ctx)
                if expected != sig:
                    bprec["ok"] = False
                    bprec["errors"].append("signature_mismatch")
                    errors += 1
            except Exception:
                # treat as non-fatal error
                bprec["ok"] = False
                bprec["errors"].append("signature_check_error")
                errors += 1

    return WalVerificationResult(ok=(errors == 0), checked=count, by_blueprint=by_bp)


def bsds_verify(urza: Any, blueprint_id: str, *, secret: bytes | None = None) -> bool:
    """Verify BSDS signature for a blueprint using the secret or environment.

    Accepts a duck-typed UrzaLibrary exposing `get()` that returns a record
    with an `extras` mapping containing `bsds` and `bsds_sig`.
    """

    record = urza.get(blueprint_id)
    if record is None or not isinstance(getattr(record, "extras", None), dict):
        return False
    try:
        from esper.urabrask.wal import verify_bsds_signature_in_extras

        if secret is not None:
            ctx = SignatureContext(secret=secret)
        else:
            ctx = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
        return verify_bsds_signature_in_extras(record.extras, ctx=ctx)
    except Exception:
        return False


__all__ = ["wal_verify", "bsds_verify", "WalVerificationResult"]
