"""Urabrask producer service helpers (prototype).

Provides a simple function to compute BSDS using heuristics and attach it to an
existing Urza record via `extras["bsds"]` for Tamiyo to consume.
"""

from __future__ import annotations

from typing import Mapping

from esper.urza import UrzaLibrary
from esper.karn import BlueprintDescriptor

from .bsds import compute_bsds


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
