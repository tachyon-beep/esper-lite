"""Urabrask benchmark worker for Weatherlight (prototype).

Runs the Benchmark Suite v1 for top-N Urza blueprints that are missing
`extras["benchmarks"]`, attaching the JSON mirror via the urabrask service.

Feature-gated by settings; bounded work per cycle; safe failure handling.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from typing import Mapping

from esper.urza import UrzaLibrary, UrzaRuntime
from esper.urabrask.service import produce_benchmarks


class UrabraskBenchWorker:
    def __init__(
        self,
        urza: UrzaLibrary,
        runtime: UrzaRuntime,
        *,
        interval_s: int = 1800,
        topn: int = 3,
        timeout_ms: int = 500,
    ) -> None:
        self._urza = urza
        self._runtime = runtime
        self._interval_s = max(1, int(interval_s))
        self._topn = max(1, int(topn))
        self._timeout_s = max(0, int(timeout_ms)) / 1000.0
        self._profiles_total: int = 0
        self._failures_total: int = 0
        self._last_duration_ms: float = 0.0
        self._last_processed: int = 0
        self._skipped_cooldown_total: int = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="urabrask-bench")

    async def run_forever(self) -> None:  # pragma: no cover - exercised in integration
        while True:
            try:
                self.run_once()
            except Exception:
                pass
            await self._sleep(self._interval_s)

    async def _sleep(self, seconds: int) -> None:  # pragma: no cover - trivial
        import asyncio

        await asyncio.sleep(seconds)

    def run_once(self) -> Mapping[str, float]:
        start = time.perf_counter()
        processed = 0
        attached_profiles = 0
        failed = 0
        skipped_cooldown = 0
        from esper.core import EsperSettings
        settings = EsperSettings()
        min_interval_s = max(0, int(getattr(settings, "urabrask_bench_min_interval_s", 3600)))
        # Snapshot current records
        records = list(self._urza.list_all().values())
        candidates: list[str] = []
        for rec in records:
            extras = rec.extras or {}
            if not isinstance(extras, dict):
                candidates.append(rec.metadata.blueprint_id)
            elif "benchmarks" not in extras:
                candidates.append(rec.metadata.blueprint_id)
            else:
                # Cooldown check
                last_run_raw = str(extras.get("benchmarks_last_run") or "")
                last_ok = True
                if last_run_raw:
                    try:
                        # Accept RFC3339 with Z
                        if last_run_raw.endswith("Z"):
                            last_run = datetime.fromisoformat(last_run_raw.replace("Z", "+00:00"))
                        else:
                            last_run = datetime.fromisoformat(last_run_raw)
                        age_s = max(0.0, (datetime.now(timezone.utc) - last_run).total_seconds())
                        if age_s < min_interval_s:
                            last_ok = False
                    except Exception:
                        last_ok = True
                if last_ok:
                    candidates.append(rec.metadata.blueprint_id)
            if len(candidates) >= self._topn:
                break
        # Attach benchmarks for candidates
        for bp in candidates:
            processed += 1
            if self._timeout_s > 0:
                future = self._executor.submit(produce_benchmarks, self._urza, self._runtime, bp)
                try:
                    proto = future.result(timeout=self._timeout_s)
                    attached_profiles += len(getattr(proto, "profiles", []))
                    # Update last_run timestamp
                    try:
                        rec = self._urza.get(bp)
                        if rec is not None:
                            extras = dict(rec.extras or {})
                            extras["benchmarks_last_run"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                            self._urza.save(rec.metadata, rec.artifact_path, extras=extras)
                    except Exception:
                        pass
                except FuturesTimeout:
                    future.cancel()
                    self._failures_total += 1
                    failed += 1
                except Exception:
                    self._failures_total += 1
                    failed += 1
            else:
                try:
                    proto = produce_benchmarks(self._urza, self._runtime, bp)
                    attached_profiles += len(getattr(proto, "profiles", []))
                    try:
                        rec = self._urza.get(bp)
                        if rec is not None:
                            extras = dict(rec.extras or {})
                            extras["benchmarks_last_run"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                            self._urza.save(rec.metadata, rec.artifact_path, extras=extras)
                    except Exception:
                        pass
                except Exception:
                    self._failures_total += 1
                    failed += 1

        self._last_duration_ms = (time.perf_counter() - start) * 1000.0
        self._last_processed = processed
        self._profiles_total += attached_profiles
        # Compute skipped via selection pass
        if processed == 0 and records:
            # If no candidate chosen due to cooldown, count as skipped
            skipped_cooldown += 1
        self._skipped_cooldown_total += float(skipped_cooldown)
        return {
            "processed": float(processed),
            "attached_profiles": float(attached_profiles),
            "failed": float(failed),
            "duration_ms": float(self._last_duration_ms),
            "skipped_cooldown": float(skipped_cooldown),
        }

    def metrics(self) -> Mapping[str, float]:
        return {
            "bench.profiles_total": float(self._profiles_total),
            "bench.failures_total": float(self._failures_total),
            "bench.last_duration_ms": float(self._last_duration_ms),
            "bench.last_processed": float(self._last_processed),
            "bench.skipped_cooldown_total": float(self._skipped_cooldown_total),
        }


__all__ = ["UrabraskBenchWorker"]
