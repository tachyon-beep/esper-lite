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

from .bsds import compute_bsds
from .crucible import run_crucible


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

    record = urza.get(blueprint_id)
    if record is None:
        raise ValueError(f"Blueprint not found in Urza: {blueprint_id}")
    bsds = run_crucible(record.metadata, artifact_path=record.artifact_path, hints=hints)
    # Mirror to JSON for extras
    bsds_json = {
        "risk_score": float(bsds.risk_score),
        "hazard_band": leyline_pb2.HazardBand.Name(bsds.hazard_band).replace("HAZARD_BAND_", ""),
        "handling_class": leyline_pb2.HandlingClass.Name(bsds.handling_class).replace("HANDLING_CLASS_", "").lower(),
        "resource_profile": leyline_pb2.ResourceProfile.Name(bsds.resource_profile).replace("RESOURCE_PROFILE_", "").lower(),
        "provenance": leyline_pb2.Provenance.Name(bsds.provenance).replace("PROVENANCE_", ""),
        "issued_at": bsds.issued_at.ToDatetime().replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
    }
    extras = dict(record.extras or {})
    extras["bsds"] = bsds_json
    urza.save(record.metadata, record.artifact_path, extras=extras)
    return bsds
