"""Urabrask Crucible v0 (prototype).

Minimal, deterministic hazard evaluator that produces a canonical Leyline BSDS
without executing kernels. Intended as an initial producer before full hazard
battery lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2


@dataclass(slots=True)
class CrucibleConfig:
    high_threshold: float = 0.60
    critical_threshold: float = 0.85


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def run_crucible(
    descriptor: BlueprintDescriptor,
    *,
    artifact_path: Path | None = None,  # reserved for future, not used in v0
    hints: Mapping[str, object] | None = None,
    config: CrucibleConfig | None = None,
) -> leyline_pb2.BSDS:
    cfg = config or CrucibleConfig()
    risk = float(getattr(descriptor, "risk", 0.0) or 0.0)
    tier = getattr(descriptor, "tier", BlueprintTier.BLUEPRINT_TIER_UNSPECIFIED)
    quarantine_only = bool(getattr(descriptor, "quarantine_only", False))

    # Priors: small bump for experimental/high risk tiers; aggressive for quarantine_only
    if tier == BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL:
        risk += 0.05
    if tier == BlueprintTier.BLUEPRINT_TIER_HIGH_RISK:
        risk += 0.10
    if quarantine_only:
        risk += 0.10
    risk = _clamp01(risk)

    if risk >= cfg.critical_threshold:
        hazard = leyline_pb2.HAZARD_BAND_CRITICAL
    elif risk >= cfg.high_threshold:
        hazard = leyline_pb2.HAZARD_BAND_HIGH
    elif risk >= 0.30:
        hazard = leyline_pb2.HAZARD_BAND_MEDIUM
    else:
        hazard = leyline_pb2.HAZARD_BAND_LOW

    if quarantine_only or hazard == leyline_pb2.HAZARD_BAND_CRITICAL:
        handling = leyline_pb2.HANDLING_CLASS_QUARANTINE
    elif hazard == leyline_pb2.HAZARD_BAND_HIGH:
        handling = leyline_pb2.HANDLING_CLASS_RESTRICTED
    else:
        handling = leyline_pb2.HANDLING_CLASS_STANDARD
    # Ensure quarantine is reflected as at least HIGH hazard in v0
    if quarantine_only and hazard < leyline_pb2.HAZARD_BAND_HIGH:
        hazard = leyline_pb2.HAZARD_BAND_HIGH

    resource_profile = leyline_pb2.RESOURCE_PROFILE_MIXED
    raw_profile = str((hints or {}).get("resource_profile", "") or "").lower()
    if raw_profile in {"gpu", "cuda"}:
        resource_profile = leyline_pb2.RESOURCE_PROFILE_GPU
    elif raw_profile == "cpu":
        resource_profile = leyline_pb2.RESOURCE_PROFILE_CPU

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
        provenance=leyline_pb2.PROVENANCE_URABRASK,
    )
    bsds.issued_at.FromDatetime(datetime.now(tz=UTC))
    return bsds


__all__ = ["CrucibleConfig", "run_crucible"]
