"""Signing + WAL helpers for Urabrask BSDS attachments (prototype).

Provides:
- Canonical JSON serialization for BSDS mirrors
- HMAC-SHA256 signing using the global Leyline secret
- Append-only WAL writer with per-blueprint hash chaining (prev_sig)
- Verification helpers for tests and tools
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from esper.core import EsperSettings
from esper.security.signing import SignatureContext, sign, verify, DEFAULT_SECRET_ENV


def _canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _iso_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class BsdsSignature:
    algo: str
    sig: str
    prev_sig: str
    issued_at: str


def compute_signature(bsds_json: dict, *, prev_sig: str, ctx: SignatureContext) -> BsdsSignature:
    payload = _canonical_dumps(bsds_json).encode("utf-8")
    signature = sign(payload, ctx)
    issued_at = str(bsds_json.get("issued_at") or _iso_now())
    return BsdsSignature(algo="HMAC-SHA256", sig=signature, prev_sig=str(prev_sig or ""), issued_at=issued_at)


def verify_signature(bsds_json: dict, signature_block: dict, *, ctx: SignatureContext) -> bool:
    try:
        sig = str(signature_block.get("sig") or "")
        if not sig:
            return False
        payload = _canonical_dumps(bsds_json).encode("utf-8")
        return verify(payload, sig, ctx)
    except Exception:
        return False


def append_wal_entry(
    *,
    wal_path: Path,
    blueprint_id: str,
    bsds_json: dict,
    sig_block: BsdsSignature,
) -> None:
    wal_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "blueprint_id": blueprint_id,
        "issued_at": sig_block.issued_at,
        "sig": sig_block.sig,
        "prev_sig": sig_block.prev_sig,
        "payload": json.loads(_canonical_dumps(bsds_json)),  # concise, stable ordering
    }
    line = _canonical_dumps(entry) + "\n"
    with wal_path.open("a", encoding="utf-8") as f:
        f.write(line)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass


def attach_signature_and_wal(
    *,
    extras: dict,
    blueprint_id: str,
    bsds_json: dict,
    settings: EsperSettings,
) -> None:
    if not settings.urabrask_signing_enabled:
        return
    try:
        ctx = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
    except Exception:
        # Fail-open for prototype; skip signing
        return
    prev_sig = ""
    try:
        prev_sig = str((extras.get("bsds_sig") or {}).get("sig") or "")
    except Exception:
        prev_sig = ""
    sig_block = compute_signature(bsds_json, prev_sig=prev_sig, ctx=ctx)
    extras["bsds_sig"] = {
        "algo": sig_block.algo,
        "sig": sig_block.sig,
        "prev_sig": sig_block.prev_sig,
        "issued_at": sig_block.issued_at,
    }
    try:
        append_wal_entry(
            wal_path=Path(settings.urabrask_wal_path),
            blueprint_id=blueprint_id,
            bsds_json=bsds_json,
            sig_block=sig_block,
        )
    except Exception:
        # Fail-open; WAL append is best-effort in prototype
        try:
            from . import metrics as _metrics

            _metrics.inc_wal_append_errors()
        except Exception:
            pass


def verify_bsds_signature_in_extras(extras: dict, *, ctx: SignatureContext) -> bool:
    try:
        bsds = extras.get("bsds")
        sig = extras.get("bsds_sig")
        if not isinstance(bsds, dict) or not isinstance(sig, dict):
            return False
        return verify_signature(bsds, sig, ctx=ctx)
    except Exception:
        return False


__all__ = [
    "BsdsSignature",
    "compute_signature",
    "verify_signature",
    "append_wal_entry",
    "attach_signature_and_wal",
    "verify_bsds_signature_in_extras",
]
