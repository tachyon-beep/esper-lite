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

# Dashboard components
from esper.karn.websocket_output import WebSocketOutput
from esper.karn.dashboard_server import create_app, run_dashboard_server
from esper.karn.integrated_dashboard import DashboardServer

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

# Counterfactual (causal attribution)
from esper.karn.counterfactual import (
    CounterfactualEngine,
    CounterfactualConfig,
    CounterfactualResult,
    CounterfactualMatrix,
    ShapleyEstimate,
    InteractionTerm,
)
from esper.karn.counterfactual_helper import (
    CounterfactualHelper,
    ContributionResult,
    compute_simple_ablation,
)

# Triggers (anomaly detection)
from esper.karn.triggers import (
    AnomalyDetector,
    PolicyAnomalyDetector,
    RollingStats,
)

# Analytics (research queries)
from esper.karn.analytics import (
    EpisodeAnalytics,
    EpisodeSummary,
    SlotSummary,
    TrajectoryPoint,
    ConvergenceInfo,
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

# TUI (terminal user interface)
from esper.karn.tui import (
    TUIOutput,
    TUIState,
    ThresholdConfig,
    HealthStatus,
)

__all__ = [
    # Dashboard
    "WebSocketOutput",
    "DashboardServer",
    "create_app",
    "run_dashboard_server",
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
    # Counterfactual
    "CounterfactualEngine",
    "CounterfactualConfig",
    "CounterfactualResult",
    "CounterfactualMatrix",
    "ShapleyEstimate",
    "InteractionTerm",
    "CounterfactualHelper",
    "ContributionResult",
    "compute_simple_ablation",
    # Triggers
    "AnomalyDetector",
    "PolicyAnomalyDetector",
    "RollingStats",
    # Analytics
    "EpisodeAnalytics",
    "EpisodeSummary",
    "SlotSummary",
    "TrajectoryPoint",
    "ConvergenceInfo",
    # Health
    "HealthMonitor",
    "SystemHealth",
    "MemoryStats",
    "GradientHealth",
    "VitalSignsMonitor",
    "VitalSigns",
    # TUI
    "TUIOutput",
    "TUIState",
    "ThresholdConfig",
    "HealthStatus",
]
