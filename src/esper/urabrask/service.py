"""Urabrask producer service helpers (prototype).

Provides a simple function to compute BSDS using heuristics and attach it to an
existing Urza record via `extras["bsds"]` for Tamiyo to consume.
"""

from __future__ import annotations

from typing import Mapping

from datetime import UTC
from esper.urza import UrzaLibrary
from esper.leyline import leyline_pb2
from esper.karn import BlueprintDescriptor
from esper.core import EsperSettings
from esper.urabrask.wal import attach_signature_and_wal

from .bsds import compute_bsds
from .benchmarks import run_benchmarks, BenchmarkConfig


def produce_and_attach_bsds(
    urza: UrzaLibrary,
    blueprint_id: str,
    *,
    hints: Mapping[str, object] | None = None,
) -> tuple[BlueprintDescriptor, dict]:
    """Compute BSDS for an existing Urza record and attach it via extras.

    Returns a pair of (descriptor, bsds_json_mirror). The canonical protobuf is
    returned by `compute_bsds` but the JSON mirror is persisted in extras for
    the prototype transport.
    """

    record = urza.get(blueprint_id)
    if record is None:
        raise ValueError(f"Blueprint not found in Urza: {blueprint_id}")

    bsds_proto, bsds_json = compute_bsds(record.metadata, artifact_path=record.artifact_path, hints=hints)
    # Merge extras to preserve existing metadata
    extras = dict(record.extras or {})
    extras["bsds"] = bsds_json
    # Optional signing + WAL append
    settings = EsperSettings()
    attach_signature_and_wal(
        extras=extras,
        blueprint_id=record.metadata.blueprint_id,
        bsds_json=bsds_json,
        settings=settings,
    )
    urza.save(record.metadata, record.artifact_path, extras=extras)
    return record.metadata, bsds_json


def produce_bsds_via_crucible(
    urza: UrzaLibrary,
    blueprint_id: str,
    *,
    hints: Mapping[str, object] | None = None,
) -> leyline_pb2.BSDS:
    """Run the minimal crucible against a blueprint and attach BSDS to Urza.

    Returns the canonical BSDS (provenance=URABRASK). Also mirrors the payload
    to Urza extras under `bsds` for prototype transport.
    """

    # Lazy import to avoid hard torch dependency at module import time
    from .crucible import run_crucible_v1, CrucibleConfigV1  # type: ignore

    record = urza.get(blueprint_id)
    if record is None:
        raise ValueError(f"Blueprint not found in Urza: {blueprint_id}")
    bsds, hazards = run_crucible_v1(record.metadata, artifact_path=record.artifact_path, hints=hints, config=CrucibleConfigV1())
    # Mirror to JSON for extras (with hazards)
    bsds_json = {
        "risk_score": float(bsds.risk_score),
        "hazard_band": leyline_pb2.HazardBand.Name(bsds.hazard_band).replace("HAZARD_BAND_", ""),
        "handling_class": leyline_pb2.HandlingClass.Name(bsds.handling_class).replace("HANDLING_CLASS_", "").lower(),
        "resource_profile": leyline_pb2.ResourceProfile.Name(bsds.resource_profile).replace("RESOURCE_PROFILE_", "").lower(),
        "provenance": leyline_pb2.Provenance.Name(bsds.provenance).replace("PROVENANCE_", ""),
        "issued_at": bsds.issued_at.ToDatetime().replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
    }
    if hazards:
        bsds_json["hazards"] = dict(hazards)
    extras = dict(record.extras or {})
    extras["bsds"] = bsds_json
    # Optional signing + WAL append
    settings = EsperSettings()
    attach_signature_and_wal(
        extras=extras,
        blueprint_id=record.metadata.blueprint_id,
        bsds_json=bsds_json,
        settings=settings,
    )
    urza.save(record.metadata, record.artifact_path, extras=extras)
    return bsds


def produce_benchmarks(
    urza: UrzaLibrary,
    runtime: object | None,
    blueprint_id: str,
    *,
    config: BenchmarkConfig | None = None,
) -> leyline_pb2.BlueprintBenchmark:
    """Run benchmarks and attach JSON mirror to Urza extras.

    - Returns the canonical `BlueprintBenchmark` proto
    - Mirrors compact summary to `extras["benchmarks"]`
    - Does not raise if runtime is unavailable; falls back to synthetic path
    """

    record = urza.get(blueprint_id)
    if record is None:
        raise ValueError(f"Blueprint not found in Urza: {blueprint_id}")
    proto, mirror = run_benchmarks(blueprint_id, runtime=runtime, config=config)
    extras = dict(record.extras or {})
    extras["benchmarks"] = mirror
    urza.save(record.metadata, record.artifact_path, extras=extras)
    return proto
