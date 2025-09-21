"""Karn blueprint catalog using canonical Leyline descriptors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from esper.leyline import leyline_pb2

BlueprintDescriptor = leyline_pb2.BlueprintDescriptor
BlueprintTier = leyline_pb2.BlueprintTier
BlueprintParameterBounds = leyline_pb2.BlueprintParameterBounds

# Forward declaration populated by templates module at import time.
DEFAULT_BLUEPRINTS: Sequence[BlueprintDescriptor]


def _clone(descriptor: BlueprintDescriptor) -> BlueprintDescriptor:
    clone = BlueprintDescriptor()
    clone.CopyFrom(descriptor)
    return clone


def _validate_parameters(descriptor: BlueprintDescriptor, parameters: dict[str, float]) -> None:
    if not descriptor.allowed_parameters:
        return
    for key, bounds in descriptor.allowed_parameters.items():
        value = parameters.get(key)
        if value is None:
            raise ValueError(f"Missing required parameter '{key}'")
        lower = bounds.min_value
        upper = bounds.max_value
        if not (lower <= float(value) <= upper):
            raise ValueError(
                f"Parameter '{key}'={value} outside bounds [{lower}, {upper}]"
            )


class KarnCatalog:
    """In-memory blueprint metadata registry backed by Leyline descriptors."""

    def __init__(self, *, load_defaults: bool = True) -> None:
        self._catalog: dict[str, BlueprintDescriptor] = {}
        if load_defaults:
            from .templates import DEFAULT_BLUEPRINTS as _DEFAULT  # local import to avoid cycles

            for descriptor in _DEFAULT:
                self.register(descriptor)

    def register(self, descriptor: BlueprintDescriptor) -> None:
        clone = _clone(descriptor)
        self._catalog[clone.blueprint_id] = clone

    def get(self, blueprint_id: str) -> BlueprintDescriptor | None:
        descriptor = self._catalog.get(blueprint_id)
        if descriptor is None:
            return None
        return _clone(descriptor)

    def list_by_tier(self, tier: BlueprintTier) -> Iterable[BlueprintDescriptor]:
        for descriptor in self._catalog.values():
            if descriptor.tier == tier:
                yield _clone(descriptor)

    def remove(self, blueprint_id: str) -> None:
        self._catalog.pop(blueprint_id, None)

    def validate_request(
        self, blueprint_id: str, parameters: dict[str, float]
    ) -> BlueprintDescriptor:
        descriptor = self.get(blueprint_id)
        if descriptor is None:
            raise KeyError(f"Blueprint '{blueprint_id}' not found")
        _validate_parameters(descriptor, parameters)
        return descriptor

    def choose_template(
        self,
        *,
        tier: BlueprintTier | None = None,
        context: str | None = None,
        allow_adversarial: bool = False,
        conservative: bool = False,
    ) -> BlueprintDescriptor:
        candidates = list(self._catalog.values())
        if tier is not None:
            candidates = [descriptor for descriptor in candidates if descriptor.tier == tier]
        if not allow_adversarial:
            candidates = [
                descriptor
                for descriptor in candidates
                if descriptor.tier != BlueprintTier.BLUEPRINT_TIER_HIGH_RISK
            ]
        if conservative:
            safe = [
                descriptor
                for descriptor in candidates
                if descriptor.tier == BlueprintTier.BLUEPRINT_TIER_SAFE
            ]
            if safe:
                candidates = safe[:10] if len(safe) > 10 else safe
        if not candidates:
            raise ValueError("No blueprints available for the requested criteria")

        candidates.sort(key=lambda descriptor: descriptor.blueprint_id)
        if not context:
            return _clone(candidates[0])

        import hashlib

        digest = hashlib.sha256(context.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(candidates)
        return _clone(candidates[index])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._catalog)

    def all(self) -> Iterable[BlueprintDescriptor]:
        for descriptor in self._catalog.values():
            yield _clone(descriptor)


__all__ = [
    "BlueprintDescriptor",
    "BlueprintTier",
    "BlueprintParameterBounds",
    "KarnCatalog",
    "DEFAULT_BLUEPRINTS",
]

