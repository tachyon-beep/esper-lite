"""Shared dependency guard utilities for fail-fast enforcement (Risk R2).

This module centralises strict dependency checks so subsystems can drop
synthetic fallbacks while emitting actionable diagnostics and telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from esper.core.telemetry import TelemetryEvent


class DependencyViolationError(RuntimeError):
    """Raised when a mandatory dependency is missing or invalid."""

    def __init__(self, subsystem: str, reason: str, *, context: Mapping[str, Any] | None = None) -> None:
        message = f"{subsystem} dependency violation: {reason}"
        super().__init__(message)
        self.subsystem = subsystem
        self.reason = reason
        self.context = dict(context or {})


@dataclass(slots=True)
class DependencyContext:
    subsystem: str
    dependency_type: str
    identifier: str | None = None
    details: Mapping[str, Any] | None = None

    def to_telemetry_event(self) -> TelemetryEvent:
        attributes: dict[str, str] = {
            "dependency_type": self.dependency_type,
        }
        if self.identifier:
            attributes["identifier"] = self.identifier
        for key, value in (self.details or {}).items():
            attributes[key] = str(value)
        return TelemetryEvent(
            description="dependency_guard.violation",
            level=3,  # CRITICAL
            attributes=attributes,
        )


class RegistryLookup(Protocol):
    def __call__(self, identifier: str) -> Any:  # pragma: no cover - protocol
        ...


def ensure_present(condition: bool, context: DependencyContext, *, reason: str) -> None:
    if not condition:
        raise DependencyViolationError(context.subsystem, reason, context=context.details or {})


def verify_registry_entry(
    identifier: str,
    lookup: RegistryLookup,
    context: DependencyContext,
    *,
    empty_ok: bool = False,
) -> Any:
    value = lookup(identifier)
    if value is None or (not empty_ok and value == ""):
        ctx = dict(context.details or {})
        ctx["identifier"] = identifier
        raise DependencyViolationError(
            context.subsystem,
            f"missing {context.dependency_type} {identifier}",
            context=ctx,
        )
    return value


__all__ = [
    "DependencyViolationError",
    "DependencyContext",
    "ensure_present",
    "verify_registry_entry",
]
