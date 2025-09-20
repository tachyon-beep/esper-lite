"""Urza artifact storage scaffold.

Persists blueprint metadata and compiled kernel references in accordance with
`docs/design/detailed_design/08-urza.md`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import Column, MetaData, String, Table, create_engine, select, delete
from sqlalchemy.engine import Engine

from esper.karn import BlueprintMetadata, BlueprintTier


@dataclass(slots=True)
class UrzaRecord:
    metadata: BlueprintMetadata
    artifact_path: Path


class UrzaLibrary:
    """SQLite-backed Urza store with filesystem artifacts."""

    def __init__(
        self,
        root: Path,
        database_url: str | None = None,
        *,
        wal_path: Path | None = None,
        cache_size: int = 128,
    ) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._cache_size = cache_size
        self._records: OrderedDict[str, UrzaRecord] = OrderedDict()
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

    def save(self, metadata: BlueprintMetadata, artifact_path: Path) -> None:
        destination = self._root / artifact_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path != destination:
            destination.write_bytes(artifact_path.read_bytes())
        self._persist_wal(metadata, destination)
        self._upsert(metadata, destination)
        self._clear_wal()

    def get(self, blueprint_id: str) -> UrzaRecord | None:
        record = self._records.get(blueprint_id)
        if record:
            self._touch_cache(blueprint_id, record)
            return record

        with self._engine.begin() as conn:
            result = conn.execute(
                select(self._table).where(self._table.c.blueprint_id == blueprint_id)
            ).mappings().first()
        if not result:
            return None
        record = self._record_from_row(result)
        self._touch_cache(blueprint_id, record)
        return record

    def list_all(self) -> dict[str, UrzaRecord]:
        return dict(self._records)

    def fetch_by_tier(self, tier: BlueprintTier) -> Iterable[UrzaRecord]:
        return [record for record in self._records.values() if record.metadata.tier is tier]

    def _hydrate_cache(self) -> None:
        with self._engine.begin() as conn:
            rows = conn.execute(select(self._table)).mappings()
            for row in rows:
                record = self._record_from_row(row)
                self._touch_cache(record.metadata.blueprint_id, record)

    def _record_from_row(self, row: Any) -> UrzaRecord:
        metadata_json = json.loads(row["metadata_json"])
        metadata = BlueprintMetadata(
            blueprint_id=row["blueprint_id"],
            name=row["name"],
            tier=BlueprintTier(metadata_json.get("tier", row["tier"])),
            description=metadata_json.get("description", ""),
            allowed_parameters={
                key: tuple(value)
                for key, value in metadata_json.get("allowed_parameters", {}).items()
            },
            risk=float(metadata_json.get("risk", 0.0)),
            stage=int(metadata_json.get("stage", 0)),
            quarantine_only=bool(metadata_json.get("quarantine_only", False)),
            approval_required=bool(metadata_json.get("approval_required", False)),
        )
        artifact_path = Path(row["artifact_path"])
        return UrzaRecord(metadata=metadata, artifact_path=artifact_path)

    def _persist_wal(self, metadata: BlueprintMetadata, destination: Path) -> None:
        payload = {
            "blueprint_id": metadata.blueprint_id,
            "artifact_path": str(destination),
            "metadata": {
                "name": metadata.name,
                "tier": metadata.tier.value,
                "description": metadata.description,
                "allowed_parameters": metadata.allowed_parameters,
                "risk": metadata.risk,
                "stage": metadata.stage,
                "quarantine_only": metadata.quarantine_only,
                "approval_required": metadata.approval_required,
            },
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
        meta_json = payload.get("metadata", {})
        metadata = BlueprintMetadata(
            blueprint_id=blueprint_id,
            name=meta_json.get("name", ""),
            tier=BlueprintTier(meta_json.get("tier", BlueprintTier.SAFE.value)),
            description=meta_json.get("description", ""),
            allowed_parameters={
                key: tuple(value)
                for key, value in meta_json.get("allowed_parameters", {}).items()
            },
            risk=float(meta_json.get("risk", 0.0)),
            stage=int(meta_json.get("stage", 0)),
            quarantine_only=bool(meta_json.get("quarantine_only", False)),
            approval_required=bool(meta_json.get("approval_required", False)),
        )
        artifact = Path(payload.get("artifact_path", ""))
        if artifact.exists():
            self._upsert(metadata, artifact)
            self._clear_wal()

    def _upsert(self, metadata: BlueprintMetadata, destination: Path) -> None:
        record = UrzaRecord(metadata, destination)
        self._touch_cache(metadata.blueprint_id, record)
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(self._table.c.blueprint_id == metadata.blueprint_id)
            )
            conn.execute(
                self._table.insert().values(
                    blueprint_id=metadata.blueprint_id,
                    name=metadata.name,
                    tier=metadata.tier.value,
                    artifact_path=str(destination),
                    metadata_json=json.dumps(
                        {
                            "allowed_parameters": metadata.allowed_parameters,
                            "description": metadata.description,
                            "tier": metadata.tier.value,
                            "risk": metadata.risk,
                            "stage": metadata.stage,
                            "quarantine_only": metadata.quarantine_only,
                            "approval_required": metadata.approval_required,
                        }
                    ),
                )
            )

    def _touch_cache(self, blueprint_id: str, record: UrzaRecord) -> None:
        if blueprint_id in self._records:
            self._records.pop(blueprint_id)
        self._records[blueprint_id] = record
        while len(self._records) > self._cache_size:
            self._records.popitem(last=False)


__all__ = ["UrzaLibrary", "UrzaRecord"]
