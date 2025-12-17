"""Overwatch - Textual TUI for Esper training monitoring.

Provides real-time visibility into training environments, seed lifecycle,
and Tamiyo decision-making.
"""

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    EnvSummary,
    SlotChipState,
    TamiyoState,
    ConnectionStatus,
    DeviceVitals,
    FeedEvent,
)

from esper.karn.overwatch.replay import (
    SnapshotWriter,
    SnapshotReader,
)

# Lazy import for OverwatchApp - Textual may not be installed
try:
    from esper.karn.overwatch.app import OverwatchApp
except ImportError:
    OverwatchApp = None  # type: ignore[misc, assignment]

__all__ = [
    # Schema
    "TuiSnapshot",
    "EnvSummary",
    "SlotChipState",
    "TamiyoState",
    "ConnectionStatus",
    "DeviceVitals",
    "FeedEvent",
    # Replay
    "SnapshotWriter",
    "SnapshotReader",
    # App (may be None if Textual not installed)
    "OverwatchApp",
]
