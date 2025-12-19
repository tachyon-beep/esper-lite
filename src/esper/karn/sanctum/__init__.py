"""Sanctum - Developer diagnostic TUI for Esper training.

Provides deep inspection when training misbehaves:
- Per-environment metrics with sparklines
- Tamiyo policy health and action distribution
- Reward component breakdown
- System vitals with CPU display

Usage:
    from esper.karn.sanctum import SanctumBackend
    hub.add_backend(SanctumBackend(num_envs=16))

    # Launch TUI
    from esper.karn.sanctum import SanctumApp
    app = SanctumApp(backend=backend)
    app.run()
"""

from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    SystemVitals,
    GPUStats,
    RewardComponents,
    EventLogEntry,
)

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.backend import SanctumBackend

# Lazy import for SanctumApp - Textual may not be installed
try:
    from esper.karn.sanctum.app import SanctumApp
except ImportError:
    SanctumApp = None  # type: ignore[misc, assignment]

__all__ = [
    # Schema
    "SanctumSnapshot",
    "EnvState",
    "SeedState",
    "TamiyoState",
    "SystemVitals",
    "GPUStats",
    "RewardComponents",
    "EventLogEntry",
    # Aggregator & Backend
    "SanctumAggregator",
    "SanctumBackend",
    # App (may be None if Textual not installed)
    "SanctumApp",
]
