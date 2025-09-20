"""Core shared primitives for Esper-Lite.

Contains configuration helpers, base telemetry contracts, and utility types
used across subsystem packages. Aligns with contracts described in
`docs/design/detailed_design/00-leyline.md` and related design notes.
"""

from .config import EsperSettings
from .events import (
    AdaptationCommand,
    AdaptationDirective,
    FieldReport,
    FieldReportOutcome,
    SeedLifecycleStage,
    SystemStatePacket,
    TrainingPhase,
)

__all__ = [
    "EsperSettings",
    "SystemStatePacket",
    "AdaptationCommand",
    "FieldReport",
    "AdaptationDirective",
    "FieldReportOutcome",
    "SeedLifecycleStage",
    "TrainingPhase",
]
