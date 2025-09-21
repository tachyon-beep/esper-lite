"""Tezzeret compilation forge for Esper-Lite (TKT-202)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from esper.karn import BlueprintDescriptor, KarnCatalog
from esper.urza import UrzaLibrary

from .compiler import TezzeretCompiler


def _midpoint(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2.0


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
    ) -> None:
        self._catalog = catalog
        self._library = library
        self._compiler = compiler
        self._wal_path = wal_path
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        pending = self._load_pending_jobs()
        if not pending:
            pending = [job.blueprint_id for job in self._discover_jobs()]
            self._persist_pending(pending)

        for blueprint_id in list(pending):
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

            artifact_path = self._compiler.compile(metadata, parameters)
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


__all__ = ["TezzeretForge", "CompilationJob"]
