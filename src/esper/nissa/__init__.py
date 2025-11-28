"""Nissa - System telemetry hub for Esper.

The nissa package is the system telemetry hub that receives carbon copies of
events from all domains and routes them to configured output backends.

Components:
    - config: TelemetryConfig with Pydantic models and profile management
    - tracker: DiagnosticTracker for collecting rich training telemetry
    - output: Output backends (console, file) and the NissaHub router

Usage:
    from esper.nissa import TelemetryConfig, DiagnosticTracker, NissaHub

    # Load telemetry configuration
    config = TelemetryConfig.from_profile("diagnostic")

    # Create diagnostic tracker
    tracker = DiagnosticTracker(model, config)

    # Set up telemetry output
    hub = NissaHub()
    hub.add_backend(ConsoleOutput())
    hub.add_backend(FileOutput("telemetry.jsonl"))
"""

from esper.nissa.config import (
    TelemetryConfig,
    GradientConfig,
    LossLandscapeConfig,
    PerClassConfig,
)
from esper.nissa.tracker import (
    DiagnosticTracker,
    GradientStats,
    GradientHealth,
    EpochSnapshot,
)
from esper.nissa.output import (
    OutputBackend,
    ConsoleOutput,
    FileOutput,
    NissaHub,
    get_hub,
    emit,
)

__all__ = [
    # Config
    "TelemetryConfig",
    "GradientConfig",
    "LossLandscapeConfig",
    "PerClassConfig",
    # Tracker
    "DiagnosticTracker",
    "GradientStats",
    "GradientHealth",
    "EpochSnapshot",
    # Output
    "OutputBackend",
    "ConsoleOutput",
    "FileOutput",
    "NissaHub",
    "get_hub",
    "emit",
]
