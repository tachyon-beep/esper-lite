"""Urza artifact storage scaffold.

Persists blueprint metadata and compiled kernel references in accordance with
`docs/design/detailed_design/08-urza.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from esper.karn import BlueprintMetadata


@dataclass(slots=True)
class UrzaRecord:
    metadata: BlueprintMetadata
    artifact_path: Path


class UrzaLibrary:
    """In-memory Urza store with filesystem-backed artifacts."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._records: dict[str, UrzaRecord] = {}

    def save(self, metadata: BlueprintMetadata, artifact_path: Path) -> None:
        destination = self._root / artifact_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path != destination:
            destination.write_bytes(artifact_path.read_bytes())
        self._records[metadata.blueprint_id] = UrzaRecord(metadata, destination)

    def get(self, blueprint_id: str) -> UrzaRecord | None:
        return self._records.get(blueprint_id)

    def list_all(self) -> dict[str, UrzaRecord]:
        return dict(self._records)


__all__ = ["UrzaLibrary", "UrzaRecord"]
