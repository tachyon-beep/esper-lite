from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from google.protobuf.json_format import MessageToDict
from torch import nn

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.tezzeret.compiler import CompiledBlueprint
from esper.urza import UrzaLibrary
from esper.urza.runtime import UrzaRuntime


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
    assert persisted.checksum is None


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


def test_urza_library_cache_ttl_enforces_expiry(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    source_root = tmp_path / "source"
    source_root.mkdir()
    library = UrzaLibrary(root=library_root, cache_ttl_seconds=1)
    metadata = _metadata("BPTTL")
    artifact = source_root / "artifact.pt"
    artifact.write_bytes(b"ttl")

    library.save(metadata, artifact)
    record = library.get("BPTTL")
    assert record is not None
    stored_path = record.artifact_path
    old = os.path.getmtime(stored_path) - 10
    os.utime(stored_path, (old, old))

    expired = library.get("BPTTL")
    assert expired is None
    metrics = library.metrics_snapshot()
    assert metrics["cache_expired"] >= 1.0
    assert metrics["cache_misses"] >= 1.0


def test_urza_library_fetch_by_stage_and_tag(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path)

    metadata_stage_one = _metadata("BPSTAGE1", stage=1)
    artifact_one = tmp_path / "artifact1.pt"
    artifact_one.write_bytes(b"stage1")

    metadata_stage_two = _metadata("BPSTAGE2", stage=2)
    artifact_two = tmp_path / "artifact2.pt"
    artifact_two.write_bytes(b"stage2")

    library.save(metadata_stage_one, artifact_one, extras={"tags": ["cnn", "gpu"]})
    library.save(metadata_stage_two, artifact_two, extras={"tags": ["transformer"]})

    stage_one_records = library.fetch_by_stage(1)
    assert {record.metadata.blueprint_id for record in stage_one_records} == {"BPSTAGE1"}

    gpu_records = library.fetch_by_tag("GPU")
    assert {record.metadata.blueprint_id for record in gpu_records} == {"BPSTAGE1"}
    transformer_records = library.fetch_by_tag("transformer")
    assert {record.metadata.blueprint_id for record in transformer_records} == {"BPSTAGE2"}


def test_urza_runtime_verifies_checksum(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path)
    metadata = _metadata("BPCHK")
    artifact = tmp_path / "artifact.pt"
    module = CompiledBlueprint(nn.Identity(), blueprint_id=metadata.blueprint_id, parameters={})
    torch.save(module, artifact)
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    catalog_update = leyline_pb2.KernelCatalogUpdate(
        blueprint_id=metadata.blueprint_id,
        artifact_ref=str(artifact),
        checksum=checksum,
    )
    library.save(metadata, artifact, catalog_update=catalog_update)
    runtime = UrzaRuntime(library)
    module, latency_ms = runtime.fetch_kernel("BPCHK")
    assert isinstance(module, nn.Module)
    assert latency_ms >= 0.0

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        runtime.fetch_kernel("BPCHK")
    assert library.get("BPCHK") is None
    metrics = library.metrics_snapshot()
    assert metrics["integrity_failures"] >= 1.0
    assert metrics["evictions"] >= 1.0


def test_urza_library_maintenance_removes_expired(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path, cache_ttl_seconds=0)
    metadata = _metadata("BPMNT")
    artifact = tmp_path / "artifact-mnt.pt"
    artifact.write_bytes(b"mnt")
    library.save(metadata, artifact)

    # Force expiry by backdating mtime.
    old = artifact.stat().st_mtime - 10
    os.utime(artifact, (old, old))

    report = library.maintenance()
    assert report["expired"] >= 1.0
    assert library.get("BPMNT") is None


def test_urza_library_maintenance_removes_missing_artifact(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path)
    metadata = _metadata("BPMISS")
    artifact = tmp_path / "artifact-miss.pt"
    artifact.write_bytes(b"miss")
    library.save(metadata, artifact)
    artifact.unlink()

    report = library.maintenance()
    assert report["missing"] >= 1.0
    assert library.get("BPMISS") is None


def test_urza_library_breaker_enters_conservative_mode(tmp_path: Path) -> None:
    library = UrzaLibrary(
        root=tmp_path,
        breaker_latency_threshold_ms=0.0,
        breaker_failure_threshold=1,
        breaker_success_threshold=1,
        breaker_timeout_ms=1.0,
    )

    metadata = _metadata("BPBREAKER", stage=3)
    artifact = tmp_path / "artifact-breaker.pt"
    artifact.write_bytes(b"breaker")
    library.save(metadata, artifact)

    # Drop cache to force database access and trigger latency failure.
    library._records.clear()

    result = library.get("BPBREAKER")
    assert result is not None

    metrics = library.metrics_snapshot()
    assert metrics["breaker_state"] == float(leyline_pb2.CIRCUIT_STATE_OPEN)
    assert metrics["conservative_mode"] == 1.0
    assert metrics["slow_queries"] >= 1.0

    denied_result = library.get("BP_UNKNOWN")
    assert denied_result is None
    metrics = library.metrics_snapshot()
    assert metrics["breaker_denied"] >= 1.0


def test_urza_library_persists_extras_graph_metadata(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path)
    metadata = _metadata("BPGM")
    artifact = tmp_path / "artifact-gm.pt"
    artifact.write_bytes(b"gm")

    graph_metadata = {
        "layers": [{"layer_id": "BPGM-L0", "type": "linear", "depth": 0, "latency_ms": 1.0}],
        "activations": [{"activation_id": "BPGM-A0", "type": "relu"}],
        "parameters": [{"name": "alpha", "min": 0.0, "max": 1.0, "span": 1.0, "default": 0.5}],
        "adjacency": {"layer": [[0, 0]]},
    }

    library.save(metadata, artifact, extras={"graph_metadata": graph_metadata})
    record = library.get("BPGM")
    assert record is not None
    assert isinstance(record.extras, dict)
    gm = record.extras.get("graph_metadata")  # type: ignore[assignment]
    assert isinstance(gm, dict)
    assert gm.get("layers") and gm.get("activations") and gm.get("parameters")

    # Save without extras: should degrade to empty extras gracefully
    metadata2 = _metadata("BPGM2")
    art2 = tmp_path / "artifact-gm2.pt"
    art2.write_bytes(b"gm2")
    library.save(metadata2, art2)
    rec2 = library.get("BPGM2")
    assert rec2 is not None
    assert isinstance(rec2.extras, dict)
    # No graph metadata when not provided; baseline extras may include housekeeping fields
    assert "graph_metadata" not in rec2.extras
