"""Karn - Research Telemetry System.

Karn is the next-generation telemetry system for morphogenetic training,
replacing Nissa with research-focused metrics and causal attribution.

Named after the silver golem with perfect memory who witnessed millennia
of multiverse history - fitting for a system that captures and recalls
every training event.

Usage:
    from esper.karn import get_collector, KarnCollector

    # Get the global collector
    collector = get_collector()
    collector.start_episode(seed=42)
    collector.emit(event)

    # Or for dashboard:
    from esper.karn import DashboardServer
    hub.add_backend(DashboardServer(port=8000))
"""

# Dashboard components (lazy import - fastapi may not be installed)
try:
    from esper.karn.websocket_output import WebSocketOutput
    from esper.karn.integrated_dashboard import DashboardServer
except ImportError:
    WebSocketOutput = None  # type: ignore[misc, assignment]
    DashboardServer = None  # type: ignore[misc, assignment]

# Store (data models)
from esper.karn.store import (
    TelemetryStore,
    EpisodeContext,
    HostBaseline,
    EpochSnapshot,
    SlotSnapshot,
    HostSnapshot,
    PolicySnapshot,
    RewardComponents,
    AdvantageStats,
    RatioStats,
    DenseTrace,
    DenseTraceTrigger,
    BatchMetrics,
    GateEvaluationTrace,
)
from esper.leyline import SeedStage  # Re-export authoritative definition

# Collector (event handling)
from esper.karn.collector import (
    KarnCollector,
    KarnConfig,
    OutputBackend,
    get_collector,
    configure,
    emit,
)

# Triggers (anomaly detection)
from esper.karn.triggers import (
    AnomalyDetector,
    PolicyAnomalyDetector,
    RollingStats,
)

# Health (system monitoring)
from esper.karn.health import (
    HealthMonitor,
    SystemHealth,
    MemoryStats,
    GradientHealth,
    VitalSignsMonitor,
    VitalSigns,
)

# Overwatch (live telemetry backend for operators)
from esper.karn.overwatch.backend import OverwatchBackend

# Sanctum (developer diagnostic TUI backend)
from esper.karn.sanctum.backend import SanctumBackend

__all__ = [
    # Dashboard
    "WebSocketOutput",
    "DashboardServer",
    # Store
    "TelemetryStore",
    "EpisodeContext",
    "HostBaseline",
    "EpochSnapshot",
    "SlotSnapshot",
    "HostSnapshot",
    "PolicySnapshot",
    "RewardComponents",
    "AdvantageStats",
    "RatioStats",
    "DenseTrace",
    "DenseTraceTrigger",
    "BatchMetrics",
    "GateEvaluationTrace",
    "SeedStage",
    # Collector
    "KarnCollector",
    "KarnConfig",
    "OutputBackend",
    "get_collector",
    "configure",
    "emit",
    # Triggers
    "AnomalyDetector",
    "PolicyAnomalyDetector",
    "RollingStats",
    # Health
    "HealthMonitor",
    "SystemHealth",
    "MemoryStats",
    "GradientHealth",
    "VitalSignsMonitor",
    "VitalSigns",
    # Overwatch (operator monitoring)
    "OverwatchBackend",
    # Sanctum (developer diagnostics)
    "SanctumBackend",
]
