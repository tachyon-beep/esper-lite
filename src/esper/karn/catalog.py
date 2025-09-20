"""Blueprint metadata scaffolding for Karn.

Provides in-memory catalog behaviour for the tiered blueprint set described in
`docs/design/detailed_design/05-karn.md` and `old/05-karn.md`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


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
    allowed_parameters: dict[str, tuple[float, float]] = field(default_factory=dict)
    risk: float = 0.0
    stage: int = 0
    quarantine_only: bool = False
    approval_required: bool = False

    def validate_parameters(self, parameters: dict[str, float]) -> None:
        if not self.allowed_parameters:
            return
        for key, bounds in self.allowed_parameters.items():
            value = parameters.get(key)
            if value is None:
                raise ValueError(f"Missing required parameter '{key}'")
            lower, upper = bounds
            if not (lower <= float(value) <= upper):
                raise ValueError(
                    f"Parameter '{key}'={value} outside bounds [{lower}, {upper}]"
                )


# Forward declaration for type checking; avoid circular import at runtime.
DEFAULT_BLUEPRINTS: Sequence[BlueprintMetadata]


class KarnCatalog:
    """In-memory blueprint metadata registry."""

    def __init__(self, *, load_defaults: bool = True) -> None:
        self._catalog: dict[str, BlueprintMetadata] = {}
        if load_defaults:
            from .templates import DEFAULT_BLUEPRINTS as _DEFAULT_BLUEPRINTS

            for metadata in _DEFAULT_BLUEPRINTS:
                self.register(metadata)

    def register(self, metadata: BlueprintMetadata) -> None:
        self._catalog[metadata.blueprint_id] = metadata

    def get(self, blueprint_id: str) -> BlueprintMetadata | None:
        return self._catalog.get(blueprint_id)

    def list_by_tier(self, tier: BlueprintTier) -> Iterable[BlueprintMetadata]:
        return (meta for meta in self._catalog.values() if meta.tier is tier)

    def remove(self, blueprint_id: str) -> None:
        self._catalog.pop(blueprint_id, None)

    def validate_request(
        self, blueprint_id: str, parameters: dict[str, float]
    ) -> BlueprintMetadata:
        metadata = self.get(blueprint_id)
        if metadata is None:
            raise KeyError(f"Blueprint '{blueprint_id}' not found")
        metadata.validate_parameters(parameters)
        return metadata

    def choose_template(
        self,
        *,
        tier: BlueprintTier | None = None,
        context: str | None = None,
        allow_adversarial: bool = False,
        conservative: bool = False,
    ) -> BlueprintMetadata:
        """Deterministically choose a blueprint matching the request."""

        candidates = list(self._catalog.values())
        if tier is not None:
            candidates = [meta for meta in candidates if meta.tier is tier]
        if not allow_adversarial:
            candidates = [meta for meta in candidates if meta.tier is not BlueprintTier.HIGH_RISK]
        if conservative:
            safe = [meta for meta in candidates if meta.tier is BlueprintTier.SAFE]
            if safe:
                candidates = safe[:10] if len(safe) > 10 else safe
        if not candidates:
            raise ValueError("No blueprints available for the requested criteria")

        candidates.sort(key=lambda meta: meta.blueprint_id)
        if not context:
            return candidates[0]

        digest = hashlib.sha256(context.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(candidates)
        return candidates[index]

    def __len__(self) -> int:
        return len(self._catalog)

    def all(self) -> Iterable[BlueprintMetadata]:
        return self._catalog.values()


__all__ = ["BlueprintMetadata", "BlueprintTier", "KarnCatalog"]
