"""Asynchronous kernel prefetch worker for Urza."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from esper.leyline import leyline_pb2
from esper.oona import OonaClient

from .library import UrzaLibrary, UrzaRecord


@dataclass(slots=True)
class PrefetchMetrics:
    hits: int = 0
    misses: int = 0
    errors: int = 0
    latency_ms: float = 0.0


class UrzaPrefetchWorker:
    """Consumes kernel prefetch requests and responds via Oona."""

    def __init__(
        self,
        oona: OonaClient,
        library: UrzaLibrary,
        *,
        checksum_fn: Callable[[Path], str] | None = None,
    ) -> None:
        self._oona = oona
        self._library = library
        self._metrics = PrefetchMetrics()
        self._checksum_fn = checksum_fn or self._default_checksum

    @property
    def metrics(self) -> PrefetchMetrics:
        return self._metrics

    async def poll_once(self, *, count: int = 1, block_ms: int = 100) -> None:
        await self._oona.consume_kernel_requests(
            self._handle_request,
            count=count,
            block_ms=block_ms,
        )

    async def run_forever(self, *, interval_ms: int = 100) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(interval_ms / 1000.0)

    async def _handle_request(self, message) -> None:  # type: ignore[override]
        request = leyline_pb2.KernelPrefetchRequest()
        request.ParseFromString(message.payload)
        start = time.perf_counter()
        record = self._library.get(request.blueprint_id)
        if record is None:
            await self._emit_error(request, reason="missing_artifact")
            self._metrics.misses += 1
            self._metrics.latency_ms = (time.perf_counter() - start) * 1000.0
            return
        artifact_path = Path(record.artifact_path)
        if not artifact_path.exists():
            await self._emit_error(request, reason="artifact_not_found")
            self._metrics.errors += 1
            self._metrics.latency_ms = (time.perf_counter() - start) * 1000.0
            return
        checksum = self._checksum_fn(artifact_path)
        if record.checksum and checksum != record.checksum:
            await self._emit_error(request, reason="checksum_mismatch")
            self._metrics.errors += 1
            self._metrics.latency_ms = (time.perf_counter() - start) * 1000.0
            return
        guard_digest = record.guard_digest
        if not guard_digest:
            guard_digest = self._fallback_guard_digest(record)
        samples = record.prewarm_samples
        prewarm_p50 = _percentile(samples, 0.5) if samples else 0.0
        prewarm_p95 = _percentile(samples, 0.95) if samples else 0.0
        ready = leyline_pb2.KernelArtifactReady(
            request_id=request.request_id,
            blueprint_id=request.blueprint_id,
            artifact_ref=str(artifact_path),
            checksum=checksum,
            guard_digest=guard_digest,
            prewarm_p50_ms=prewarm_p50,
            prewarm_p95_ms=prewarm_p95,
        )
        await self._oona.publish_kernel_artifact_ready(ready)
        self._metrics.hits += 1
        self._metrics.latency_ms = (time.perf_counter() - start) * 1000.0

    async def _emit_error(self, request: leyline_pb2.KernelPrefetchRequest, *, reason: str) -> None:
        error = leyline_pb2.KernelArtifactError(
            request_id=request.request_id,
            blueprint_id=request.blueprint_id,
            reason=reason,
        )
        await self._oona.publish_kernel_artifact_error(error)

    @staticmethod
    def _default_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fallback_guard_digest(record: UrzaRecord) -> str:
        base = f"{record.metadata.blueprint_id}:{record.metadata.tier}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()


__all__ = ["UrzaPrefetchWorker", "PrefetchMetrics"]


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    quantile = max(0.0, min(1.0, quantile))
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = position - lower
    return float(lower_value + (upper_value - lower_value) * fraction)
