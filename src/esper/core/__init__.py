"""Core shared primitives for Esper-Lite.

Contains configuration helpers, base telemetry contracts, and utility types
used across subsystem packages. Aligns with contracts described in
`docs/design/detailed_design/00-leyline.md` and related design notes.
"""

from .async_runner import AsyncTimeoutError, AsyncWorker, AsyncWorkerHandle
from .dependency_guard import (
    DependencyContext,
    DependencyViolationError,
    ensure_present,
    verify_registry_entry,
)
from .config import EsperSettings
from .telemetry import TelemetryEvent, TelemetryMetric, build_telemetry_packet

__all__ = [
    "AsyncTimeoutError",
    "AsyncWorker",
    "AsyncWorkerHandle",
    "DependencyContext",
    "DependencyViolationError",
    "ensure_present",
    "verify_registry_entry",
    "EsperSettings",
    "TelemetryMetric",
    "TelemetryEvent",
    "build_telemetry_packet",
]
