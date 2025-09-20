"""Blueprint metadata scaffolding for Karn.

Provides in-memory catalog behaviour for the tiered blueprint set described in
`docs/design/detailed_design/05-karn.md` and `old/05-karn.md`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class BlueprintTier(str, Enum):
    SAFE = "safe"
    EXPERIMENTAL = "experimental"
    HIGH_RISK = "high_risk"


@dataclass(slots=True)
class BlueprintMetadata:
    blueprint_id: str
    name: str
    tier: BlueprintTier
    description: str
    allowed_parameters: dict[str, tuple[float, float]]


class KarnCatalog:
    """In-memory blueprint metadata registry."""

    def __init__(self) -> None:
        self._catalog: dict[str, BlueprintMetadata] = {}

    def register(self, metadata: BlueprintMetadata) -> None:
        self._catalog[metadata.blueprint_id] = metadata

    def get(self, blueprint_id: str) -> BlueprintMetadata | None:
        return self._catalog.get(blueprint_id)

    def list_by_tier(self, tier: BlueprintTier) -> Iterable[BlueprintMetadata]:
        return (meta for meta in self._catalog.values() if meta.tier is tier)

    def remove(self, blueprint_id: str) -> None:
        self._catalog.pop(blueprint_id, None)


__all__ = ["BlueprintMetadata", "BlueprintTier", "KarnCatalog"]
