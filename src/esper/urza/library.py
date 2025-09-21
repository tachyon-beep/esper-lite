"""Urza artifact storage scaffold.

Persists blueprint metadata and compiled kernel references in accordance with
`docs/design/detailed_design/08-urza.md`.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from google.protobuf.json_format import MessageToDict, ParseDict
from sqlalchemy import Column, MetaData, String, Table, create_engine, select, delete
from sqlalchemy.engine import Engine

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2


@dataclass(slots=True)
class UrzaRecord:
    metadata: BlueprintDescriptor
    artifact_path: Path
    guard_digest: str | None = None
    prewarm_samples: tuple[float, ...] = ()
    artifact_mtime: float | None = None
    checksum: str | None = None
    compile_ms: float | None = None
    prewarm_ms: float | None = None
    compile_strategy: str | None = None
    eager_fallback: bool = False
    guard_spec: tuple[dict[str, Any], ...] = ()
    inductor_cache_dir: str | None = None


class UrzaLibrary:
    """SQLite-backed Urza store with filesystem artifacts."""

    def __init__(
        self,
        root: Path,
        database_url: str | None = None,
        *,
        wal_path: Path | None = None,
        cache_size: int = 128,
        cache_ttl_seconds: int | None = None,
        max_prewarm_samples: int = 20,
    ) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._cache_size = cache_size
        self._records: OrderedDict[str, UrzaRecord] = OrderedDict()
        self._prewarm_samples: dict[str, list[float]] = {}
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_prewarm_samples = max(1, max_prewarm_samples)
        self._metrics: dict[str, float] = {
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "cache_errors": 0.0,
            "cache_expired": 0.0,
            "lookup_latency_ms": 0.0,
        }
        db_url = database_url or f"sqlite:///{(self._root / 'catalog.db').resolve()}"
        self._engine: Engine = create_engine(db_url, future=True)
        self._metadata = MetaData()
        self._table = Table(
            "blueprints",
            self._metadata,
            Column("blueprint_id", String, primary_key=True),
            Column("name", String, nullable=False),
            Column("tier", String, nullable=False),
            Column("artifact_path", String, nullable=False),
            Column("metadata_json", String, nullable=False),
        )
        self._metadata.create_all(self._engine)
        self._wal_path = wal_path or (self._root / "urza_wal.json")
        self._recover_from_wal()
        self._hydrate_cache()

    def save(
        self,
        metadata: BlueprintDescriptor,
        artifact_path: Path,
        *,
        catalog_update: leyline_pb2.KernelCatalogUpdate | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        destination = self._root / artifact_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path != destination:
            destination.write_bytes(artifact_path.read_bytes())
        extras_payload = self._build_extras(
            metadata.blueprint_id,
            destination,
            catalog_update,
            extra_metadata=extras,
        )
        self._persist_wal(metadata, destination, extras_payload)
        self._upsert(metadata, destination, extras_payload)
        self._clear_wal()

    def get(self, blueprint_id: str) -> UrzaRecord | None:
        start = time.perf_counter()
        try:
            record = self._records.get(blueprint_id)
            if record:
                if self._is_expired(record):
                    self._metrics["cache_expired"] += 1.0
                    self._evict_record(blueprint_id, record, delete_artifact=True)
                    record = None
                else:
                    self._metrics["cache_hits"] += 1.0
                    self._metrics["lookup_latency_ms"] = (time.perf_counter() - start) * 1000.0
                    self._touch_cache(blueprint_id, record)
                    return _clone_record(record)

            with self._engine.begin() as conn:
                result = conn.execute(
                    select(self._table).where(self._table.c.blueprint_id == blueprint_id)
                ).mappings().first()
            if not result:
                self._metrics["cache_misses"] += 1.0
                self._metrics["lookup_latency_ms"] = (time.perf_counter() - start) * 1000.0
                return None
            record = self._record_from_row(result)
            if self._is_expired(record):
                self._metrics["cache_expired"] += 1.0
                self._evict_record(blueprint_id, record, delete_artifact=True)
                self._metrics["lookup_latency_ms"] = (time.perf_counter() - start) * 1000.0
                return None
            self._touch_cache(blueprint_id, record)
            self._metrics["cache_misses"] += 1.0
            self._metrics["lookup_latency_ms"] = (time.perf_counter() - start) * 1000.0
            return _clone_record(record)
        except Exception:  # pragma: no cover - defensive guard for IO/DB issues
            self._metrics["cache_errors"] += 1.0
            return None

    def list_all(self) -> dict[str, UrzaRecord]:
        return {
            blueprint_id: _clone_record(record)
            for blueprint_id, record in self._records.items()
        }

    def fetch_by_tier(self, tier: BlueprintTier) -> Iterable[UrzaRecord]:
        return [
            _clone_record(record)
            for record in self._records.values()
            if record.metadata.tier == tier
        ]

    def metrics_snapshot(self) -> dict[str, float]:
        return dict(self._metrics)

    def _hydrate_cache(self) -> None:
        with self._engine.begin() as conn:
            rows = conn.execute(select(self._table)).mappings()
            for row in rows:
                record = self._record_from_row(row)
                self._touch_cache(record.metadata.blueprint_id, record)

    def _record_from_row(self, row: Any) -> UrzaRecord:
        metadata_json = json.loads(row["metadata_json"])
        extras = metadata_json.pop("_urza", {})
        metadata = BlueprintDescriptor()
        ParseDict(metadata_json, metadata, ignore_unknown_fields=True)
        if not metadata.blueprint_id:
            metadata.blueprint_id = row["blueprint_id"]
        if not metadata.name:
            metadata.name = row["name"]
        tier_value = row.get("tier")
        if tier_value:
            try:
                metadata.tier = BlueprintTier.Value(tier_value)
            except ValueError:  # stored as numeric string
                metadata.tier = BlueprintTier(int(tier_value))  # type: ignore[arg-type]
        artifact_path = Path(row["artifact_path"])
        samples = tuple(float(x) for x in extras.get("prewarm_samples", []))
        if samples:
            self._prewarm_samples[metadata.blueprint_id] = list(samples)
        else:
            self._prewarm_samples.pop(metadata.blueprint_id, None)
        record = UrzaRecord(
            metadata=_clone_descriptor(metadata),
            artifact_path=artifact_path,
            guard_digest=extras.get("guard_digest"),
            prewarm_samples=samples,
            artifact_mtime=extras.get("artifact_mtime"),
            checksum=extras.get("checksum"),
            compile_ms=float(extras.get("compile_ms", 0.0)) if extras.get("compile_ms") is not None else None,
            prewarm_ms=float(extras.get("prewarm_ms", 0.0)) if extras.get("prewarm_ms") is not None else None,
            compile_strategy=extras.get("compile_strategy"),
            eager_fallback=bool(extras.get("eager_fallback", False)),
            guard_spec=tuple(extras.get("guard_spec", [])),
            inductor_cache_dir=extras.get("inductor_cache_dir"),
        )
        return record

    def _build_extras(
        self,
        blueprint_id: str,
        destination: Path,
        catalog_update: leyline_pb2.KernelCatalogUpdate | None,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        guard_digest: str | None = None
        samples = list(self._prewarm_samples.get(blueprint_id, []))
        checksum: str | None = None
        if catalog_update is not None:
            if catalog_update.guard_digest:
                guard_digest = catalog_update.guard_digest
            if catalog_update.prewarm_ms:
                samples.append(float(catalog_update.prewarm_ms))
            if catalog_update.checksum:
                checksum = catalog_update.checksum
        if len(samples) > self._max_prewarm_samples:
            samples = samples[-self._max_prewarm_samples :]
        if samples:
            self._prewarm_samples[blueprint_id] = samples
        else:
            self._prewarm_samples.pop(blueprint_id, None)
        artifact_mtime = None
        try:
            artifact_mtime = destination.stat().st_mtime
        except FileNotFoundError:  # pragma: no cover - filesystem races
            artifact_mtime = None
        extras: dict[str, Any] = {
            "guard_digest": guard_digest,
            "prewarm_samples": samples,
            "artifact_mtime": artifact_mtime,
        }
        if checksum:
            extras["checksum"] = checksum
        if catalog_update is not None:
            extras["compile_ms"] = float(catalog_update.compile_ms)
            extras["prewarm_ms"] = float(catalog_update.prewarm_ms)
        if extra_metadata:
            extras.update(extra_metadata)
        return extras

    def _current_mtime(self, record: UrzaRecord) -> float | None:
        try:
            mtime = record.artifact_path.stat().st_mtime
        except FileNotFoundError:
            return None
        record.artifact_mtime = mtime
        return mtime

    def _is_expired(self, record: UrzaRecord) -> bool:
        if self._cache_ttl_seconds is None:
            return False
        mtime = self._current_mtime(record)
        if mtime is None:
            return True
        return (time.time() - mtime) > max(self._cache_ttl_seconds, 0)

    def _evict_record(
        self,
        blueprint_id: str,
        record: UrzaRecord,
        *,
        delete_artifact: bool = False,
    ) -> None:
        self._records.pop(blueprint_id, None)
        self._prewarm_samples.pop(blueprint_id, None)
        if delete_artifact:
            try:
                record.artifact_path.unlink()
            except FileNotFoundError:
                pass
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(self._table.c.blueprint_id == blueprint_id)
            )

    def _persist_wal(
        self,
        metadata: BlueprintDescriptor,
        destination: Path,
        extras: dict[str, Any],
    ) -> None:
        payload = {
            "blueprint_id": metadata.blueprint_id,
            "artifact_path": str(destination),
            "descriptor": MessageToDict(
                metadata,
                preserving_proto_field_name=True,
            ),
            "extras": extras,
        }
        self._wal_path.write_text(json.dumps(payload), encoding="utf-8")

    def _clear_wal(self) -> None:
        if self._wal_path.exists():
            self._wal_path.unlink()

    def _recover_from_wal(self) -> None:
        if not self._wal_path.exists():
            return
        try:
            payload = json.loads(self._wal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        blueprint_id = payload.get("blueprint_id")
        if not blueprint_id:
            return
        descriptor_json = payload.get("descriptor", {})
        extras = payload.get("extras", {})
        metadata = BlueprintDescriptor()
        ParseDict(descriptor_json, metadata, ignore_unknown_fields=True)
        if not metadata.blueprint_id:
            metadata.blueprint_id = blueprint_id
        if metadata.tier == BlueprintTier.BLUEPRINT_TIER_UNSPECIFIED:
            metadata.tier = BlueprintTier.BLUEPRINT_TIER_SAFE
        artifact = Path(payload.get("artifact_path", ""))
        if artifact.exists():
            self._upsert(metadata, artifact, extras)
            self._clear_wal()

    def _upsert(self, metadata: BlueprintDescriptor, destination: Path, extras: dict[str, Any]) -> None:
        record = UrzaRecord(
            metadata=_clone_descriptor(metadata),
            artifact_path=destination,
            guard_digest=extras.get("guard_digest"),
            prewarm_samples=tuple(float(x) for x in extras.get("prewarm_samples", [])),
            artifact_mtime=extras.get("artifact_mtime"),
            checksum=extras.get("checksum"),
            compile_ms=float(extras.get("compile_ms", 0.0)) if extras.get("compile_ms") is not None else None,
            prewarm_ms=float(extras.get("prewarm_ms", 0.0)) if extras.get("prewarm_ms") is not None else None,
            compile_strategy=extras.get("compile_strategy"),
            eager_fallback=bool(extras.get("eager_fallback", False)),
            guard_spec=tuple(extras.get("guard_spec", [])),
            inductor_cache_dir=extras.get("inductor_cache_dir"),
        )
        self._touch_cache(metadata.blueprint_id, record)
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(self._table.c.blueprint_id == metadata.blueprint_id)
            )
            descriptor_dict = MessageToDict(
                metadata,
                preserving_proto_field_name=True,
            )
            descriptor_dict["_urza"] = extras
            conn.execute(
                self._table.insert().values(
                    blueprint_id=metadata.blueprint_id,
                    name=metadata.name,
                    tier=BlueprintTier.Name(metadata.tier),
                    artifact_path=str(destination),
                    metadata_json=json.dumps(descriptor_dict),
                )
            )

    def _touch_cache(self, blueprint_id: str, record: UrzaRecord) -> None:
        if blueprint_id in self._records:
            self._records.pop(blueprint_id)
        self._records[blueprint_id] = _clone_record(record)
        while len(self._records) > self._cache_size:
            self._records.popitem(last=False)


__all__ = ["UrzaLibrary", "UrzaRecord"]


def _clone_descriptor(descriptor: BlueprintDescriptor) -> BlueprintDescriptor:
    clone = BlueprintDescriptor()
    clone.CopyFrom(descriptor)
    return clone


def _clone_record(record: UrzaRecord) -> UrzaRecord:
    return UrzaRecord(
        metadata=_clone_descriptor(record.metadata),
        artifact_path=record.artifact_path,
        guard_digest=record.guard_digest,
        prewarm_samples=tuple(record.prewarm_samples),
        artifact_mtime=record.artifact_mtime,
        checksum=record.checksum,
        compile_ms=record.compile_ms,
        prewarm_ms=record.prewarm_ms,
        compile_strategy=record.compile_strategy,
        eager_fallback=record.eager_fallback,
        guard_spec=tuple(record.guard_spec),
        inductor_cache_dir=record.inductor_cache_dir,
    )
