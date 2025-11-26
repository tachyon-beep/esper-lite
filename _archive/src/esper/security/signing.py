"""Static secret-based signing utilities for Leyline messages (TKT-501)."""

from __future__ import annotations

import base64
import hmac
import os
from dataclasses import dataclass
from hashlib import sha256

DEFAULT_SECRET_ENV = "ESPER_LEYLINE_SECRET"


@dataclass(slots=True)
class SignatureContext:
    """Holds signing configuration and secret material."""

    secret: bytes
    header: str = "X-Leyline-Signature"

    @classmethod
    def from_environment(cls, env_var: str = DEFAULT_SECRET_ENV) -> "SignatureContext":
        raw = os.getenv(env_var)
        if not raw:
            raise RuntimeError(f"Signing secret missing: set {env_var}")
        return cls(secret=raw.encode("utf-8"))


def sign(payload: bytes, ctx: SignatureContext) -> str:
    """Return base64-encoded HMAC-SHA256 signature for payload."""

    mac = hmac.new(ctx.secret, payload, sha256)
    return base64.b64encode(mac.digest()).decode("ascii")


def verify(payload: bytes, signature: str, ctx: SignatureContext) -> bool:
    """Verify base64-encoded signature for payload."""

    try:
        expected = sign(payload, ctx)
    except Exception:
        return False
    return hmac.compare_digest(expected, signature)


__all__ = ["SignatureContext", "sign", "verify", "DEFAULT_SECRET_ENV"]
