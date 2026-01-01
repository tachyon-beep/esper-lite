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

from typing import TYPE_CHECKING, Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Config
    "TelemetryConfig": ("esper.nissa.config", "TelemetryConfig"),
    "GradientConfig": ("esper.nissa.config", "GradientConfig"),
    "LossLandscapeConfig": ("esper.nissa.config", "LossLandscapeConfig"),
    "PerClassConfig": ("esper.nissa.config", "PerClassConfig"),
    # Tracker (HEAVY - loads torch)
    "DiagnosticTracker": ("esper.nissa.tracker", "DiagnosticTracker"),
    "GradientStats": ("esper.nissa.tracker", "GradientStats"),
    "GradientHealth": ("esper.nissa.tracker", "GradientHealth"),
    "EpochSnapshot": ("esper.nissa.tracker", "EpochSnapshot"),
    # Output
    "OutputBackend": ("esper.nissa.output", "OutputBackend"),
    "ConsoleOutput": ("esper.nissa.output", "ConsoleOutput"),
    "FileOutput": ("esper.nissa.output", "FileOutput"),
    "DirectoryOutput": ("esper.nissa.output", "DirectoryOutput"),
    "NissaHub": ("esper.nissa.output", "NissaHub"),
    "get_hub": ("esper.nissa.output", "get_hub"),
    "reset_hub": ("esper.nissa.output", "reset_hub"),
    "emit": ("esper.nissa.output", "emit"),
    # Analytics
    "BlueprintStats": ("esper.nissa.analytics", "BlueprintStats"),
    "SeedScoreboard": ("esper.nissa.analytics", "SeedScoreboard"),
    "BlueprintAnalytics": ("esper.nissa.analytics", "BlueprintAnalytics"),
    "BLUEPRINT_COMPUTE_MULTIPLIERS": ("esper.nissa.analytics", "BLUEPRINT_COMPUTE_MULTIPLIERS"),
    "compute_cost_for_blueprint": ("esper.nissa.analytics", "compute_cost_for_blueprint"),
}

__all__ = list(_LAZY_IMPORTS.keys())

# TYPE_CHECKING imports for static analysis (mypy, IDE navigation).
# These are never executed at runtime, preserving lazy import semantics.
if TYPE_CHECKING:
    from esper.nissa.analytics import (
        BLUEPRINT_COMPUTE_MULTIPLIERS as BLUEPRINT_COMPUTE_MULTIPLIERS,
        BlueprintAnalytics as BlueprintAnalytics,
        BlueprintStats as BlueprintStats,
        SeedScoreboard as SeedScoreboard,
        compute_cost_for_blueprint as compute_cost_for_blueprint,
    )
    from esper.nissa.config import (
        GradientConfig as GradientConfig,
        LossLandscapeConfig as LossLandscapeConfig,
        PerClassConfig as PerClassConfig,
        TelemetryConfig as TelemetryConfig,
    )
    from esper.nissa.output import (
        ConsoleOutput as ConsoleOutput,
        DirectoryOutput as DirectoryOutput,
        FileOutput as FileOutput,
        NissaHub as NissaHub,
        OutputBackend as OutputBackend,
        emit as emit,
        get_hub as get_hub,
        reset_hub as reset_hub,
    )
    from esper.nissa.tracker import (
        DiagnosticTracker as DiagnosticTracker,
        EpochSnapshot as EpochSnapshot,
        GradientHealth as GradientHealth,
        GradientStats as GradientStats,
    )


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Heavy modules (tracker with torch dependency) are only loaded when accessed.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public API for introspection (dir(), IDE, debuggers)."""
    return sorted(set(__all__) | set(globals().keys()))
