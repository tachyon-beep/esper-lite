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

    # For web dashboard, use Overwatch:
    from esper.karn.overwatch import OverwatchBackend
    hub.add_backend(OverwatchBackend(port=8080))
"""

# WebSocket output (lazy import - fastapi may not be installed)
try:
    from esper.karn.websocket_output import WebSocketOutput
except ImportError:
    WebSocketOutput = None  # type: ignore[misc, assignment]

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

# Sanctum (developer diagnostic TUI backend)
from esper.karn.sanctum.backend import SanctumBackend

__all__ = [
    # WebSocket
    "WebSocketOutput",
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
    # Sanctum (developer diagnostics)
    "SanctumBackend",
]
