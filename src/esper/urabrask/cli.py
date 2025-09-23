"""CLI to produce BSDS via Crucible v0 and attach to Urza.

Usage:
  esper-urabrask-produce --urza-root ./var/urza --blueprint-id BP001 [--resource-profile gpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esper.urza import UrzaLibrary
from esper.urabrask.service import produce_bsds_via_crucible


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce BSDS via Crucible and attach to Urza extras")
    p.add_argument("--urza-root", type=Path, default=Path("./var/urza"), help="Urza root directory (catalog.db, artifacts/)")
    p.add_argument("--blueprint-id", required=True, help="Blueprint identifier to process")
    p.add_argument("--resource-profile", choices=["cpu", "gpu", "memory_heavy", "io_heavy", "mixed"], default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse_args(list(argv or sys.argv[1:]))
    urza = UrzaLibrary(root=ns.urza_root)
    hints = {"resource_profile": ns.resource_profile} if ns.resource_profile else None
    bsds = produce_bsds_via_crucible(urza, ns.blueprint_id, hints=hints)
    payload = {
        "blueprint_id": bsds.blueprint_id,
        "risk_score": float(bsds.risk_score),
        "hazard_band": bsds.HazardBand.Name(bsds.hazard_band).replace("HAZARD_BAND_", "") if hasattr(bsds, "HazardBand") else "",
        "handling_class": bsds.HandlingClass.Name(bsds.handling_class).replace("HANDLING_CLASS_", "").lower() if hasattr(bsds, "HandlingClass") else "",
        "resource_profile": bsds.ResourceProfile.Name(bsds.resource_profile).replace("RESOURCE_PROFILE_", "").lower() if hasattr(bsds, "ResourceProfile") else "",
        "provenance": bsds.Provenance.Name(bsds.provenance).replace("PROVENANCE_", "") if hasattr(bsds, "Provenance") else "",
        "issued_at": bsds.issued_at.ToDatetime().isoformat(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

