"""Urabrask producer worker for Weatherlight (prototype).

Scans Urza for blueprints lacking URABRASK provenance BSDS and attaches a BSDS
via the Crucible v0. Bounded work per cycle; safe failure handling.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Mapping

from esper.leyline import leyline_pb2
from esper.urabrask.service import produce_bsds_via_crucible
from esper.urza import UrzaLibrary


class UrabraskProducer:
    def __init__(
        self,
        urza: UrzaLibrary,
        *,
        interval_s: int = 300,
        topn: int = 5,
        only_safe: bool = True,
        timeout_ms: int = 200,
    ) -> None:
        self._urza = urza
        self._interval_s = max(1, int(interval_s))
        self._topn = max(1, int(topn))
        self._only_safe = bool(only_safe)
        self._timeout_s = max(0, int(timeout_ms)) / 1000.0
        self._produced_total: int = 0
        self._failures_total: int = 0
        self._last_duration_ms: float = 0.0
        self._last_processed: int = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="urabrask")

    async def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception:
                # Defensive guard; errors reflected in counters
                pass
            await self._sleep(self._interval_s)

    async def _sleep(self, seconds: int) -> None:  # pragma: no cover - trivial
        import asyncio

        await asyncio.sleep(seconds)

    def run_once(self) -> Mapping[str, float]:
        start = time.perf_counter()
        processed = 0
        produced = 0
        failed = 0
        # Snapshot current records
        records = list(self._urza.list_all().values())
        # Filter candidates
        candidates = []
        for rec in records:
            extras = rec.extras or {}
            bsds = extras.get("bsds") if isinstance(extras, dict) else None
            prov = None
            if isinstance(bsds, dict):
                prov = str(bsds.get("provenance") or "").upper()
            if prov == "URABRASK":
                continue
            if self._only_safe and rec.metadata.tier != getattr(
                leyline_pb2, "BLUEPRINT_TIER_SAFE", 1
            ):
                continue
            candidates.append(rec.metadata.blueprint_id)
            if len(candidates) >= self._topn:
                break
        # Attach BSDS for candidates
        for bp in candidates:
            processed += 1
            if self._timeout_s > 0:
                future = self._executor.submit(produce_bsds_via_crucible, self._urza, bp, None)
                try:
                    _ = future.result(timeout=self._timeout_s)
                    self._produced_total += 1
                    produced += 1
                except FuturesTimeout:
                    future.cancel()
                    self._failures_total += 1
                    failed += 1
                except Exception:
                    self._failures_total += 1
                    failed += 1
            else:
                try:
                    _ = produce_bsds_via_crucible(self._urza, bp)
                    self._produced_total += 1
                    produced += 1
                except Exception:
                    self._failures_total += 1
                    failed += 1

        self._last_duration_ms = (time.perf_counter() - start) * 1000.0
        self._last_processed = processed
        return {
            "processed": float(processed),
            "produced": float(produced),
            "failed": float(failed),
            "duration_ms": float(self._last_duration_ms),
        }

    def metrics(self) -> Mapping[str, float]:
        return {
            "produced_total": float(self._produced_total),
            "failures_total": float(self._failures_total),
            "last_duration_ms": float(self._last_duration_ms),
            "last_processed": float(self._last_processed),
        }
