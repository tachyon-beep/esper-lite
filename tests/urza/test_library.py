from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.protobuf.json_format import MessageToDict

from esper.karn import (
    BlueprintDescriptor,
    BlueprintTier,
)
from esper.urza import UrzaLibrary


def _metadata(blueprint_id: str, *, risk: float = 0.2, stage: int = 1) -> BlueprintDescriptor:
    descriptor = BlueprintDescriptor(
        blueprint_id=blueprint_id,
        name=f"Blueprint {blueprint_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="Unit test blueprint",
        risk=risk,
        stage=stage,
        quarantine_only=False,
        approval_required=False,
    )
    bounds = descriptor.allowed_parameters["alpha"]
    bounds.min_value = 0.0
    bounds.max_value = 1.0
    return descriptor


def test_urza_library_persists_metadata(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path)
    metadata = _metadata("BPTEST")
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"data")

    library.save(metadata, artifact)

    record = library.get("BPTEST")
    assert record is not None
    assert record.metadata.risk == pytest.approx(0.2)
    assert record.metadata.stage == 1
    assert record.artifact_path.exists()

    relaunch = UrzaLibrary(root=tmp_path)
    persisted = relaunch.get("BPTEST")
    assert persisted is not None
    bounds = persisted.metadata.allowed_parameters["alpha"]
    assert bounds.min_value == pytest.approx(0.0)
    assert bounds.max_value == pytest.approx(1.0)


def test_urza_library_cache_eviction(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path, cache_size=2)
    for idx in range(3):
        metadata = _metadata(f"BP{idx}")
        artifact = tmp_path / f"artifact{idx}.pt"
        artifact.write_bytes(b"a")
        library.save(metadata, artifact)

    # Access to populate cache order
    assert library.get("BP0") is not None
    assert library.get("BP1") is not None
    assert library.get("BP2") is not None
    assert len(library.list_all()) <= 2


def test_urza_library_recovers_from_wal(tmp_path: Path) -> None:
    wal = tmp_path / "wal.json"
    library = UrzaLibrary(root=tmp_path, wal_path=wal)
    metadata = _metadata("BPWAL")
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"wal")

    payload = {
        "blueprint_id": metadata.blueprint_id,
        "artifact_path": str(artifact),
        "descriptor": MessageToDict(
            metadata,
            preserving_proto_field_name=True,
        ),
    }
    wal.write_text(json.dumps(payload), encoding="utf-8")

    recovered = UrzaLibrary(root=tmp_path, wal_path=wal)
    record = recovered.get("BPWAL")
    assert record is not None
    assert not wal.exists()
