"""BSDS‑Lite heuristic producer and canonical BSDS builder (prototype).

Implements lightweight, deterministic heuristics to derive a Blueprint Safety
Data Sheet from existing blueprint descriptors. Returns both a canonical
Leyline BSDS message and a JSON mirror suitable for Urza `extras["bsds"]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Tuple

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2


@dataclass(slots=True)
class BsdsHeuristicConfig:
    """Tunable thresholds and priors for BSDS heuristics."""

    high_threshold: float = 0.60
    critical_threshold: float = 0.85
    experimental_risk_prior: float = 0.05
    high_risk_prior: float = 0.10
    quarantine_prior: float = 0.10


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _hazard_from_risk(risk: float, cfg: BsdsHeuristicConfig) -> leyline_pb2.HazardBand:
    if risk >= cfg.critical_threshold:
        return leyline_pb2.HAZARD_BAND_CRITICAL
    if risk >= cfg.high_threshold:
        return leyline_pb2.HAZARD_BAND_HIGH
    if risk >= 0.30:
        return leyline_pb2.HAZARD_BAND_MEDIUM
    return leyline_pb2.HAZARD_BAND_LOW


def _handling_from_hazard(
    hazard: leyline_pb2.HazardBand,
    *,
    quarantine_only: bool,
) -> leyline_pb2.HandlingClass:
    if quarantine_only or hazard == leyline_pb2.HAZARD_BAND_CRITICAL:
        return leyline_pb2.HANDLING_CLASS_QUARANTINE
    if hazard == leyline_pb2.HAZARD_BAND_HIGH:
        return leyline_pb2.HANDLING_CLASS_RESTRICTED
    return leyline_pb2.HANDLING_CLASS_STANDARD


def _resource_profile_from_hints(hints: Mapping[str, object] | None) -> leyline_pb2.ResourceProfile:
    if not hints:
        return leyline_pb2.RESOURCE_PROFILE_MIXED
    raw = str(hints.get("resource_profile", "") or "").lower()
    if raw in {"gpu", "cuda"}:
        return leyline_pb2.RESOURCE_PROFILE_GPU
    if raw in {"cpu"}:
        return leyline_pb2.RESOURCE_PROFILE_CPU
    if raw in {"memory_heavy", "mem", "memory"}:
        return leyline_pb2.RESOURCE_PROFILE_MEMORY_HEAVY
    if raw in {"io_heavy", "io"}:
        return leyline_pb2.RESOURCE_PROFILE_IO_HEAVY
    return leyline_pb2.RESOURCE_PROFILE_MIXED


def _stringify_hazard(hazard: int) -> str:
    return leyline_pb2.HazardBand.Name(hazard).replace("HAZARD_BAND_", "")


def _stringify_handling(handling: int) -> str:
    name = leyline_pb2.HandlingClass.Name(handling).replace("HANDLING_CLASS_", "").lower()
    return name


def _stringify_resource(profile: int) -> str:
    return leyline_pb2.ResourceProfile.Name(profile).replace("RESOURCE_PROFILE_", "").lower()


def compute_bsds(
    descriptor: BlueprintDescriptor,
    *,
    artifact_path: Path | None = None,  # reserved for future use
    hints: Mapping[str, object] | None = None,
    config: BsdsHeuristicConfig | None = None,
) -> Tuple[leyline_pb2.BSDS, dict]:
    """Compute BSDS using deterministic heuristics.

    Returns a pair of (proto, json_dict_mirror) suitable for Urza extras.
    """

    cfg = config or BsdsHeuristicConfig()
    risk = float(getattr(descriptor, "risk", 0.0) or 0.0)
    tier = getattr(descriptor, "tier", BlueprintTier.BLUEPRINT_TIER_UNSPECIFIED)
    quarantine_only = bool(getattr(descriptor, "quarantine_only", False))

    if tier == BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL:
        risk += cfg.experimental_risk_prior
    if tier == BlueprintTier.BLUEPRINT_TIER_HIGH_RISK:
        risk += cfg.high_risk_prior
    if quarantine_only:
        risk += cfg.quarantine_prior
    risk = _clamp01(risk)

    hazard = _hazard_from_risk(risk, cfg)
    handling = _handling_from_hazard(hazard, quarantine_only=quarantine_only)
    resource_profile = _resource_profile_from_hints(hints)

    bsds = leyline_pb2.BSDS(
        version=1,
        blueprint_id=getattr(descriptor, "blueprint_id", ""),
        risk_score=risk,
        hazard_band=hazard,
        handling_class=handling,
        resource_profile=resource_profile,
        recommendation=(
            "Prefer optimizer downgrade; avoid aggressive grafting"
            if hazard in (leyline_pb2.HAZARD_BAND_HIGH, leyline_pb2.HAZARD_BAND_CRITICAL)
            else ""
        ),
        provenance=leyline_pb2.PROVENANCE_HEURISTIC,
    )
    bsds.issued_at.FromDatetime(datetime.now(tz=UTC))

    # JSON mirror for Urza extras
    json_block = {
        "risk_score": float(bsds.risk_score),
        "hazard_band": _stringify_hazard(bsds.hazard_band),
        "handling_class": _stringify_handling(bsds.handling_class),
        "resource_profile": _stringify_resource(bsds.resource_profile),
        "provenance": leyline_pb2.Provenance.Name(bsds.provenance).replace("PROVENANCE_", ""),
        "issued_at": bsds.issued_at.ToDatetime()
        .replace(tzinfo=UTC)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    if bsds.recommendation:
        json_block["mitigation"] = {"recommendation": bsds.recommendation}

    return bsds, json_block
