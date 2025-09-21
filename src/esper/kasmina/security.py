"""Security utilities for Kasmina command handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import time
from typing import Callable

from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, verify


@dataclass(slots=True)
class VerificationResult:
    accepted: bool
    reason: str = ""


class NonceLedger:
    """Tracks recently seen nonces to prevent replay."""

    def __init__(self, *, ttl_seconds: float = 300.0, clock: Callable[[], float] | None = None) -> None:
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._entries: dict[str, float] = {}

    def register(self, nonce: str) -> bool:
        now = self._clock()
        self._cleanup(now)
        expiry = self._entries.get(nonce)
        if expiry is not None and expiry > now:
            return False
        self._entries[nonce] = now + self._ttl
        return True

    def _cleanup(self, now: float) -> None:
        stale = [nonce for nonce, expiry in self._entries.items() if expiry <= now]
        for nonce in stale:
            del self._entries[nonce]


class CommandVerifier:
    """Validates adaptation commands for signature, nonce, and freshness."""

    def __init__(
        self,
        *,
        signing_context: SignatureContext,
        nonce_ledger: NonceLedger,
        freshness_window_seconds: float = 60.0,
    ) -> None:
        self._ctx = signing_context
        self._nonce_ledger = nonce_ledger
        self._freshness_window = timedelta(seconds=freshness_window_seconds)

    def verify(self, command: leyline_pb2.AdaptationCommand, signature: str) -> VerificationResult:
        payload = command.SerializeToString()
        if not signature:
            return VerificationResult(False, "missing_signature")
        if not verify(payload, signature, self._ctx):
            return VerificationResult(False, "invalid_signature")
        if not command.command_id:
            return VerificationResult(False, "missing_command_id")
        if not self._nonce_ledger.register(command.command_id):
            return VerificationResult(False, "nonce_replayed")

        if command.HasField("issued_at"):
            issued_at = command.issued_at.ToDatetime().replace(tzinfo=UTC)
            now = datetime.now(tz=UTC)
            if issued_at > now + timedelta(seconds=5):
                return VerificationResult(False, "issued_in_future")
            if now - issued_at > self._freshness_window:
                return VerificationResult(False, "stale_command")
        else:
            return VerificationResult(False, "missing_timestamp")

        return VerificationResult(True, "ok")


__all__ = ["CommandVerifier", "NonceLedger", "VerificationResult"]
