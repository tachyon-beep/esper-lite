"""Urza artifact storage scaffold.

Persists blueprint metadata and compiled kernel references in accordance with
`docs/design/detailed_design/08-urza.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine

from esper.karn import BlueprintMetadata, BlueprintTier


@dataclass(slots=True)
class UrzaRecord:
    metadata: BlueprintMetadata
    artifact_path: Path


class UrzaLibrary:
    """SQLite-backed Urza store with filesystem artifacts."""

    def __init__(self, root: Path, database_url: str | None = None) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, UrzaRecord] = {}
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
        self._hydrate_cache()

    def save(self, metadata: BlueprintMetadata, artifact_path: Path) -> None:
        destination = self._root / artifact_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path != destination:
            destination.write_bytes(artifact_path.read_bytes())
        record = UrzaRecord(metadata, destination)
        self._records[metadata.blueprint_id] = record
        with self._engine.begin() as conn:
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
                        }
                    ),
                )
            )

    def get(self, blueprint_id: str) -> UrzaRecord | None:
        record = self._records.get(blueprint_id)
        if record:
            return record

        with self._engine.begin() as conn:
            result = conn.execute(
                select(self._table).where(self._table.c.blueprint_id == blueprint_id)
            ).mappings().first()
        if not result:
            return None
        record = self._record_from_row(result)
        self._records[blueprint_id] = record
        return record

    def list_all(self) -> dict[str, UrzaRecord]:
        return dict(self._records)

    def _hydrate_cache(self) -> None:
        with self._engine.begin() as conn:
            rows = conn.execute(select(self._table)).mappings()
            for row in rows:
                record = self._record_from_row(row)
                self._records[record.metadata.blueprint_id] = record

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
        )
        artifact_path = Path(row["artifact_path"])
        return UrzaRecord(metadata=metadata, artifact_path=artifact_path)


__all__ = ["UrzaLibrary", "UrzaRecord"]
