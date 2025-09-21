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

from google.protobuf.json_format import MessageToDict, ParseDict
from sqlalchemy import Column, MetaData, String, Table, create_engine, select, delete
from sqlalchemy.engine import Engine

from esper.karn import BlueprintDescriptor, BlueprintTier


@dataclass(slots=True)
class UrzaRecord:
    metadata: BlueprintDescriptor
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

    def save(self, metadata: BlueprintDescriptor, artifact_path: Path) -> None:
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
            clone = UrzaRecord(_clone(record.metadata), record.artifact_path)
            self._touch_cache(blueprint_id, record)
            return clone

        with self._engine.begin() as conn:
            result = conn.execute(
                select(self._table).where(self._table.c.blueprint_id == blueprint_id)
            ).mappings().first()
        if not result:
            return None
        record = self._record_from_row(result)
        self._touch_cache(blueprint_id, record)
        return UrzaRecord(_clone(record.metadata), record.artifact_path)

    def list_all(self) -> dict[str, UrzaRecord]:
        return {
            blueprint_id: UrzaRecord(_clone(record.metadata), record.artifact_path)
            for blueprint_id, record in self._records.items()
        }

    def fetch_by_tier(self, tier: BlueprintTier) -> Iterable[UrzaRecord]:
        return [
            UrzaRecord(_clone(record.metadata), record.artifact_path)
            for record in self._records.values()
            if record.metadata.tier == tier
        ]

    def _hydrate_cache(self) -> None:
        with self._engine.begin() as conn:
            rows = conn.execute(select(self._table)).mappings()
            for row in rows:
                record = self._record_from_row(row)
                self._touch_cache(record.metadata.blueprint_id, record)

    def _record_from_row(self, row: Any) -> UrzaRecord:
        metadata_json = json.loads(row["metadata_json"])
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
        return UrzaRecord(metadata=_clone(metadata), artifact_path=artifact_path)

    def _persist_wal(self, metadata: BlueprintDescriptor, destination: Path) -> None:
        payload = {
            "blueprint_id": metadata.blueprint_id,
            "artifact_path": str(destination),
            "descriptor": MessageToDict(
                metadata,
                preserving_proto_field_name=True,
            ),
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
        metadata = BlueprintDescriptor()
        ParseDict(descriptor_json, metadata, ignore_unknown_fields=True)
        if not metadata.blueprint_id:
            metadata.blueprint_id = blueprint_id
        if metadata.tier == BlueprintTier.BLUEPRINT_TIER_UNSPECIFIED:
            metadata.tier = BlueprintTier.BLUEPRINT_TIER_SAFE
        artifact = Path(payload.get("artifact_path", ""))
        if artifact.exists():
            self._upsert(metadata, artifact)
            self._clear_wal()

    def _upsert(self, metadata: BlueprintDescriptor, destination: Path) -> None:
        record = UrzaRecord(_clone(metadata), destination)
        self._touch_cache(metadata.blueprint_id, record)
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(self._table.c.blueprint_id == metadata.blueprint_id)
            )
            conn.execute(
                self._table.insert().values(
                    blueprint_id=metadata.blueprint_id,
                    name=metadata.name,
                    tier=BlueprintTier.Name(metadata.tier),
                    artifact_path=str(destination),
                    metadata_json=json.dumps(
                        MessageToDict(
                            metadata,
                            preserving_proto_field_name=True,
                        )
                    ),
                )
            )

    def _touch_cache(self, blueprint_id: str, record: UrzaRecord) -> None:
        if blueprint_id in self._records:
            self._records.pop(blueprint_id)
        self._records[blueprint_id] = UrzaRecord(_clone(record.metadata), record.artifact_path)
        while len(self._records) > self._cache_size:
            self._records.popitem(last=False)


__all__ = ["UrzaLibrary", "UrzaRecord"]


def _clone(descriptor: BlueprintDescriptor) -> BlueprintDescriptor:
    clone = BlueprintDescriptor()
    clone.CopyFrom(descriptor)
    return clone
