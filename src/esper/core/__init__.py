"""Core shared primitives for Esper-Lite.

Contains configuration helpers, base telemetry contracts, and utility types
used across subsystem packages. Aligns with contracts described in
`docs/design/detailed_design/00-leyline.md` and related design notes.
"""

from .config import EsperSettings
from .telemetry import TelemetryEvent, TelemetryMetric, build_telemetry_packet

__all__ = [
    "EsperSettings",
    "TelemetryMetric",
    "TelemetryEvent",
    "build_telemetry_packet",
]
