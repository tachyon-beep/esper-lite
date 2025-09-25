"""Fault drill helpers for Nissa (TKT-304)."""

from __future__ import annotations

from collections.abc import Iterable

from esper.core.telemetry import TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from .observability import NissaIngestor


DrillResult = dict[str, object]


def _emit(ingestor: NissaIngestor, source: str, metrics: Iterable[TelemetryMetric]) -> None:
    packet = build_telemetry_packet(
        packet_id=f"drill-{source}",
        source=source,
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
        metrics=list(metrics),
    )
    ingestor.ingest_telemetry(packet)


def simulate_training_latency_spike(ingestor: NissaIngestor) -> DrillResult:
    """Trigger and resolve the training latency high alert."""

    metrics = TelemetryMetric("tolaria.training.latency_ms", 25.0)
    for _ in range(3):
        _emit(ingestor, "tolaria", [metrics])

    alert = ingestor.active_alerts.get("training_latency_high")

    _emit(ingestor, "tolaria", [TelemetryMetric("tolaria.training.latency_ms", 10.0)])
    _emit(ingestor, "tolaria", [TelemetryMetric("tolaria.training.latency_ms", 9.0)])

    cleared = "training_latency_high" not in ingestor.active_alerts
    return {
        "alert": alert,
        "cleared": cleared,
        "notifications": ingestor.alert_events,
    }


def simulate_kasmina_isolation_breach(ingestor: NissaIngestor) -> DrillResult:
    metric = TelemetryMetric("kasmina.isolation.violations", 5.0)
    _emit(ingestor, "kasmina", [metric])
    alert = ingestor.active_alerts.get("kasmina_isolation_violation")
    _emit(ingestor, "kasmina", [TelemetryMetric("kasmina.isolation.violations", 0.0)])
    cleared = "kasmina_isolation_violation" not in ingestor.active_alerts
    return {"alert": alert, "cleared": cleared}


def simulate_oona_queue_depth_spike(ingestor: NissaIngestor) -> DrillResult:
    metric_high = TelemetryMetric("oona.queue.depth", 5000.0)
    for _ in range(2):
        _emit(ingestor, "oona", [metric_high])
    alert = ingestor.active_alerts.get("oona_queue_depth")
    _emit(ingestor, "oona", [TelemetryMetric("oona.queue.depth", 1000.0)])
    cleared = "oona_queue_depth" not in ingestor.active_alerts
    return {"alert": alert, "cleared": cleared}


def simulate_tezzeret_retry_burst(ingestor: NissaIngestor) -> DrillResult:
    metric = TelemetryMetric("tezzeret.compile.retry_count", 2.0)
    _emit(ingestor, "tezzeret", [metric])
    alert = ingestor.active_alerts.get("tezzeret_compile_retry_high")
    _emit(ingestor, "tezzeret", [TelemetryMetric("tezzeret.compile.retry_count", 0.0)])
    cleared = "tezzeret_compile_retry_high" not in ingestor.active_alerts
    return {"alert": alert, "cleared": cleared}


def run_all_drills(ingestor: NissaIngestor) -> dict[str, DrillResult]:
    """Execute all fault drills in sequence."""

    return {
        "training_latency_high": simulate_training_latency_spike(ingestor),
        "kasmina_isolation_violation": simulate_kasmina_isolation_breach(ingestor),
        "oona_queue_depth": simulate_oona_queue_depth_spike(ingestor),
        "tezzeret_compile_retry_high": simulate_tezzeret_retry_burst(ingestor),
    }


__all__ = [
    "simulate_training_latency_spike",
    "simulate_kasmina_isolation_breach",
    "simulate_oona_queue_depth_spike",
    "simulate_tezzeret_retry_burst",
    "run_all_drills",
]
