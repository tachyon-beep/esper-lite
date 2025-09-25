"""Karn blueprint catalog using canonical Leyline descriptors."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from esper.core import TelemetryEvent
from esper.leyline import leyline_pb2
from esper.oona.messaging import BreakerSnapshot, CircuitBreaker

BlueprintDescriptor = leyline_pb2.BlueprintDescriptor
BlueprintTier = leyline_pb2.BlueprintTier
BlueprintParameterBounds = leyline_pb2.BlueprintParameterBounds

# Forward declaration populated by templates module at import time.
DEFAULT_BLUEPRINTS: Sequence[BlueprintDescriptor]


@dataclass(slots=True)
class BlueprintQuery:
    """Lightweight representation of a Karn template request."""

    blueprint_id: str | None = None
    tier: BlueprintTier | None = None
    parameters: Mapping[str, float] = field(default_factory=dict)
    context: str | None = None
    conservative: bool = False
    allow_adversarial: bool = False


@dataclass(slots=True)
class KarnSelection:
    """Outcome details for a Karn catalogue lookup."""

    descriptor: BlueprintDescriptor
    parameters: Mapping[str, float] = field(default_factory=dict)
    conservative: bool = False
    tier: BlueprintTier = BlueprintTier.BLUEPRINT_TIER_UNSPECIFIED


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
            raise ValueError(f"Parameter '{key}'={value} outside bounds [{lower}, {upper}]")


class KarnCatalog:
    """In-memory blueprint metadata registry backed by Leyline descriptors."""

    def __init__(
        self,
        *,
        load_defaults: bool = True,
        breaker_failure_threshold: int = 3,
        breaker_success_threshold: int = 1,
        breaker_timeout_s: float = 30.0,
    ) -> None:
        self._catalog: dict[str, BlueprintDescriptor] = {}
        if load_defaults:
            from .templates import DEFAULT_BLUEPRINTS as _DEFAULT  # local import to avoid cycles

            for descriptor in _DEFAULT:
                self.register(descriptor)
        self._breaker = CircuitBreaker(
            failure_threshold=breaker_failure_threshold,
            success_threshold=breaker_success_threshold,
            timeout_ms=max(breaker_timeout_s, 0.0) * 1000.0,
        )
        snapshot = self._breaker.snapshot()
        self._breaker_state = snapshot.state
        self._metrics: dict[str, float] = {
            "karn.requests.total": 0.0,
            "karn.requests.failed": 0.0,
            "karn.breaker.denied": 0.0,
            "karn.breaker.open_total": 0.0,
            "karn.breaker.state": float(snapshot.state),
            "karn.selection.safe": 0.0,
            "karn.selection.experimental": 0.0,
            "karn.selection.adversarial": 0.0,
            "karn.selection.latency_ms": 0.0,
        }
        self._telemetry_events: list[TelemetryEvent] = []

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

    def handle_query(self, query: BlueprintQuery) -> KarnSelection:
        allowed, snapshot = self._breaker.allow()
        if snapshot is not None:
            self._update_breaker_state(snapshot)
        if not allowed:
            self._metrics["karn.breaker.denied"] += 1.0
            self._emit_event(
                "karn.breaker_denied",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"reason": "breaker_open"},
            )
            raise RuntimeError("Karn breaker open; rejecting blueprint query")

        start = time.perf_counter()
        try:
            descriptor: BlueprintDescriptor
            if query.blueprint_id:
                descriptor = self.validate_request(
                    query.blueprint_id,
                    dict(query.parameters),
                )
            else:
                descriptor = self.choose_template(
                    tier=query.tier,
                    context=query.context,
                    allow_adversarial=query.allow_adversarial,
                    conservative=query.conservative,
                )
                if query.parameters:
                    _validate_parameters(descriptor, dict(query.parameters))
        except Exception as exc:
            self._metrics["karn.requests.failed"] += 1.0
            failure_snapshot = self._breaker.record_failure()
            self._update_breaker_state(failure_snapshot)
            self._emit_event(
                "karn.selection_failed",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"error": type(exc).__name__},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._metrics["karn.requests.total"] += 1.0
        self._metrics["karn.selection.latency_ms"] = latency_ms
        tier_metric = self._tier_metric_name(descriptor.tier)
        if tier_metric:
            self._metrics[tier_metric] += 1.0
        success_snapshot = self._breaker.record_success()
        self._update_breaker_state(success_snapshot)
        self._emit_event(
            "karn.selection_succeeded",
            attributes={
                "blueprint_id": descriptor.blueprint_id,
                "tier": BlueprintTier.Name(descriptor.tier),
                "latency_ms": f"{latency_ms:.2f}",
                "conservative": str(query.conservative).lower(),
            },
        )
        return KarnSelection(
            descriptor=descriptor,
            parameters=dict(query.parameters),
            conservative=query.conservative,
            tier=descriptor.tier,
        )

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

    def metrics_snapshot(self) -> dict[str, float]:
        return dict(self._metrics)

    def drain_telemetry_events(self) -> list[TelemetryEvent]:
        events = list(self._telemetry_events)
        self._telemetry_events.clear()
        return events

    def breaker_snapshot(self) -> BreakerSnapshot:
        snapshot = self._breaker.snapshot()
        self._update_breaker_state(snapshot)
        return snapshot

    def _update_breaker_state(self, snapshot: BreakerSnapshot) -> None:
        self._breaker_state = snapshot.state
        self._metrics["karn.breaker.state"] = float(snapshot.state)
        if snapshot.state == leyline_pb2.CIRCUIT_STATE_OPEN:
            self._metrics["karn.breaker.open_total"] += 1.0
            self._emit_event(
                "karn.breaker_opened",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                attributes={"failures": str(snapshot.failure_count)},
            )

    def _tier_metric_name(self, tier: BlueprintTier) -> str | None:
        if tier == BlueprintTier.BLUEPRINT_TIER_SAFE:
            return "karn.selection.safe"
        if tier == BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL:
            return "karn.selection.experimental"
        if tier == BlueprintTier.BLUEPRINT_TIER_HIGH_RISK:
            return "karn.selection.adversarial"
        return None

    def _emit_event(
        self,
        description: str,
        *,
        level: leyline_pb2.TelemetryLevel = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        payload = {k: str(v) for k, v in (attributes or {}).items()}
        self._telemetry_events.append(
            TelemetryEvent(
                description=description,
                level=level,
                attributes=payload,
            )
        )


__all__ = [
    "BlueprintDescriptor",
    "BlueprintTier",
    "BlueprintParameterBounds",
    "BlueprintQuery",
    "KarnCatalog",
    "KarnSelection",
    "DEFAULT_BLUEPRINTS",
]
