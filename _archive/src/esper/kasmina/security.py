"""Security utilities for Kasmina command handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, verify


@dataclass(slots=True)
class VerificationResult:
    accepted: bool
    reason: str = ""


@dataclass(slots=True)
class NonceLedgerSnapshot:
    """Summary of ledger state for telemetry/metrics."""

    size: int
    ttl_seconds: float
    evictions_total: int


class NonceLedger:
    """Tracks recently seen nonces to prevent replay."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int | None = 10_000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._entries: dict[str, float] = {}
        self._evictions_total = 0
        self._max_entries = max_entries if max_entries and max_entries > 0 else None
        self._recent_truncation = 0

    def register(self, nonce: str) -> bool:
        now = self._clock()
        self._evictions_total += self._cleanup(now)
        expiry = self._entries.get(nonce)
        if expiry is not None and expiry > now:
            return False
        self._entries[nonce] = now + self._ttl
        self._evictions_total += self._enforce_capacity()
        return True

    def maintenance(self) -> int:
        """Perform background cleanup and return number of stale nonces removed."""

        evicted = self._cleanup(self._clock())
        self._evictions_total += evicted
        return evicted

    def clear(self) -> None:
        """Remove all tracked nonces and reset counters."""

        self._entries.clear()
        self._recent_truncation = 0
        self._evictions_total = 0

    def snapshot(self) -> NonceLedgerSnapshot:
        """Return a snapshot suitable for telemetry emission."""

        return NonceLedgerSnapshot(
            size=len(self._entries),
            ttl_seconds=self._ttl,
            evictions_total=self._evictions_total,
        )

    def pop_recent_truncation(self) -> int:
        """Return and clear the most recent truncation count."""

        value = self._recent_truncation
        self._recent_truncation = 0
        return value

    def _cleanup(self, now: float) -> int:
        stale = [nonce for nonce, expiry in self._entries.items() if expiry <= now]
        for nonce in stale:
            del self._entries[nonce]
        return len(stale)

    def _enforce_capacity(self) -> int:
        if self._max_entries is None or len(self._entries) <= self._max_entries:
            return 0
        truncated = 0
        while len(self._entries) > self._max_entries:
            # Remove the nonce with the earliest expiry to preserve newer entries.
            oldest_nonce = min(self._entries.items(), key=lambda item: item[1])[0]
            del self._entries[oldest_nonce]
            truncated += 1
        if truncated:
            self._recent_truncation += truncated
        return truncated


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
        stored_signature = command.annotations.get("signature")
        if stored_signature is not None:
            del command.annotations["signature"]
        payload = command.SerializeToString(deterministic=True)
        if stored_signature is not None:
            command.annotations["signature"] = stored_signature
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


__all__ = ["CommandVerifier", "NonceLedger", "NonceLedgerSnapshot", "VerificationResult"]
