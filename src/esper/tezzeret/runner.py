"""Tezzeret compilation forge for Esper-Lite (TKT-202)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import logging

from esper.karn import BlueprintDescriptor, KarnCatalog
from esper.urza import UrzaLibrary

from .compiler import TezzeretCompiler

LOGGER = logging.getLogger("esper.tezzeret")


def _midpoint(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2.0


@dataclass(slots=True)
class ForgeMetrics:
    breaker_state: int = 0
    breaker_open_total: int = 0
    consecutive_failures: int = 0
    last_error: str = ""


@dataclass(slots=True)
class CompilationJob:
    blueprint_id: str


class TezzeretForge:
    """Coordinates blueprint compilation and persistence with WAL recovery."""

    def __init__(
        self,
        catalog: KarnCatalog,
        library: UrzaLibrary,
        compiler: TezzeretCompiler,
        *,
        wal_path: Path,
        compile_timeout_s: float = 60.0,
        breaker_threshold: int = 3,
    ) -> None:
        self._catalog = catalog
        self._library = library
        self._compiler = compiler
        self._wal_path = wal_path
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)
        self._compile_timeout_s = max(compile_timeout_s, 1.0)
        self._breaker_threshold = max(breaker_threshold, 1)
        self._consecutive_failures = 0
        self._breaker_open_until: float | None = None
        self._metrics = ForgeMetrics()

    def run(self) -> None:
        pending = self._load_pending_jobs()
        if not pending:
            pending = [job.blueprint_id for job in self._discover_jobs()]
            self._persist_pending(pending)

        for blueprint_id in list(pending):
            if self._breaker_open():
                LOGGER.warning(
                    "Tezzeret breaker open; skipping compilation for %s",
                    blueprint_id,
                )
                break
            metadata = self._catalog.get(blueprint_id)
            if metadata is None:
                pending.remove(blueprint_id)
                self._persist_pending(pending)
                continue

            if self._library.get(blueprint_id):
                pending.remove(blueprint_id)
                self._persist_pending(pending)
                continue

            parameters = {
                key: _midpoint((bounds.min_value, bounds.max_value))
                for key, bounds in metadata.allowed_parameters.items()
            }

            artifact_path = self._compile_with_breaker(metadata, parameters)
            update = self._compiler.latest_catalog_update()
            result = self._compiler.latest_result()
            extras = None
            if result is not None:
                extras = {
                    "guard_spec": result.guard_spec,
                    "guard_digest": result.guard_digest,
                    "compile_ms": result.compile_ms,
                    "prewarm_ms": result.prewarm_ms,
                    "compile_strategy": result.compile_strategy,
                    "eager_fallback": result.eager_fallback,
                    "inductor_cache_dir": result.inductor_cache_dir,
                }
            self._library.save(
                metadata,
                artifact_path,
                catalog_update=update,
                extras=extras,
            )
            self._consecutive_failures = 0
            self._metrics.consecutive_failures = 0
            self._metrics.breaker_state = 0
            pending.remove(blueprint_id)
            self._persist_pending(pending)

        if not pending and self._wal_path.exists():
            self._wal_path.unlink()

    def _discover_jobs(self) -> Iterable[CompilationJob]:
        for metadata in self._catalog.all():
            yield CompilationJob(blueprint_id=metadata.blueprint_id)

    def _load_pending_jobs(self) -> list[str]:
        if not self._wal_path.exists():
            return []
        try:
            payload = json.loads(self._wal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return list(payload.get("pending", []))

    def _persist_pending(self, pending: Iterable[str]) -> None:
        data = {"pending": list(pending)}
        self._wal_path.write_text(json.dumps(data), encoding="utf-8")

    def _breaker_open(self) -> bool:
        if self._breaker_open_until is None:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._breaker_open_until = None
            self._consecutive_failures = 0
            self._metrics.breaker_state = 0
            return False
        return True

    def _open_breaker(self) -> None:
        backoff = min(60.0, self._compile_timeout_s * 2)
        self._breaker_open_until = time.monotonic() + backoff
        self._metrics.breaker_state = 2
        self._metrics.breaker_open_total += 1
        LOGGER.error("Tezzeret breaker opened for %.1f seconds", backoff)

    def _compile_with_breaker(
        self,
        metadata: BlueprintDescriptor,
        parameters: dict[str, float],
    ) -> Path:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._compiler.compile, metadata, parameters)
            try:
                artifact_path = future.result(timeout=self._compile_timeout_s)
                return artifact_path
            except FuturesTimeout as exc:
                future.cancel()
                self._consecutive_failures += 1
                self._metrics.consecutive_failures = self._consecutive_failures
                self._metrics.last_error = "timeout"
                if self._consecutive_failures >= self._breaker_threshold:
                    self._open_breaker()
                LOGGER.error(
                    "Tezzeret compile timed out for %s after %.1fs",
                    metadata.blueprint_id,
                    self._compile_timeout_s,
                )
                raise RuntimeError("compile timeout") from exc
            except Exception as exc:  # pragma: no cover - defensive path
                self._consecutive_failures += 1
                self._metrics.consecutive_failures = self._consecutive_failures
                self._metrics.last_error = type(exc).__name__
                if self._consecutive_failures >= self._breaker_threshold:
                    self._open_breaker()
                LOGGER.error("Tezzeret compile failed for %s: %s", metadata.blueprint_id, exc)
                raise

    def metrics_snapshot(self) -> dict[str, float]:
        snapshot = self._compiler.metrics_snapshot()
        snapshot.update(
            {
                "tezzeret.breaker.state": float(self._metrics.breaker_state),
                "tezzeret.breaker.open_total": float(self._metrics.breaker_open_total),
                "tezzeret.compile.consecutive_failures": float(self._metrics.consecutive_failures),
            }
        )
        return snapshot


__all__ = ["TezzeretForge", "CompilationJob"]
