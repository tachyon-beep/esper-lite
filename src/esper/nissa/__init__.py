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

NOTE: This module uses PEP 562 lazy imports. Heavy modules (tracker with torch)
are only loaded when accessed, not at package import time.
"""

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
    "DirectoryOutput",
    "NissaHub",
    "get_hub",
    "reset_hub",
    "emit",
    # Analytics
    "BlueprintStats",
    "SeedScoreboard",
    "BlueprintAnalytics",
    "BLUEPRINT_COMPUTE_MULTIPLIERS",
    "compute_cost_for_blueprint",
]


from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Heavy modules (tracker with torch dependency) are only loaded when accessed.
    """
    # Config (lightweight)
    if name in ("TelemetryConfig", "GradientConfig", "LossLandscapeConfig", "PerClassConfig"):
        from esper.nissa.config import (
            TelemetryConfig,
            GradientConfig,
            LossLandscapeConfig,
            PerClassConfig,
        )
        mapping: dict[str, Any] = {
            "TelemetryConfig": TelemetryConfig,
            "GradientConfig": GradientConfig,
            "LossLandscapeConfig": LossLandscapeConfig,
            "PerClassConfig": PerClassConfig,
        }
        return mapping[name]

    # Tracker (HEAVY - loads torch)
    if name in ("DiagnosticTracker", "GradientStats", "GradientHealth", "EpochSnapshot"):
        from esper.nissa.tracker import (
            DiagnosticTracker,
            GradientStats,
            GradientHealth,
            EpochSnapshot,
        )
        return {"DiagnosticTracker": DiagnosticTracker, "GradientStats": GradientStats,
                "GradientHealth": GradientHealth, "EpochSnapshot": EpochSnapshot}[name]

    # Output (lightweight)
    if name in ("OutputBackend", "ConsoleOutput", "FileOutput", "DirectoryOutput",
                "NissaHub", "get_hub", "reset_hub", "emit"):
        from esper.nissa.output import (
            OutputBackend,
            ConsoleOutput,
            FileOutput,
            DirectoryOutput,
            NissaHub,
            get_hub,
            reset_hub,
            emit,
        )
        return {"OutputBackend": OutputBackend, "ConsoleOutput": ConsoleOutput,
                "FileOutput": FileOutput, "DirectoryOutput": DirectoryOutput,
                "NissaHub": NissaHub, "get_hub": get_hub, "reset_hub": reset_hub, "emit": emit}[name]

    # Analytics (lightweight)
    if name in ("BlueprintStats", "SeedScoreboard", "BlueprintAnalytics",
                "BLUEPRINT_COMPUTE_MULTIPLIERS", "compute_cost_for_blueprint"):
        from esper.nissa.analytics import (
            BlueprintStats,
            SeedScoreboard,
            BlueprintAnalytics,
            BLUEPRINT_COMPUTE_MULTIPLIERS,
            compute_cost_for_blueprint,
        )
        return {"BlueprintStats": BlueprintStats, "SeedScoreboard": SeedScoreboard,
                "BlueprintAnalytics": BlueprintAnalytics,
                "BLUEPRINT_COMPUTE_MULTIPLIERS": BLUEPRINT_COMPUTE_MULTIPLIERS,
                "compute_cost_for_blueprint": compute_cost_for_blueprint}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
