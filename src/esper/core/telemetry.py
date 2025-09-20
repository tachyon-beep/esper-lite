"""Helpers for assembling Leyline telemetry packets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from google.protobuf import timestamp_pb2

from esper.leyline import leyline_pb2


@dataclass(slots=True)
class TelemetryMetric:
    """Represents a single metric sample attached to a telemetry packet."""

    name: str
    value: float
    unit: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TelemetryEvent:
    """Represents a discrete telemetry event."""

    description: str
    level: leyline_pb2.TelemetryLevel = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
    attributes: Mapping[str, str] = field(default_factory=dict)


def build_telemetry_packet(
    *,
    packet_id: str,
    source: str,
    level: leyline_pb2.TelemetryLevel,
    metrics: Iterable[TelemetryMetric],
    events: Iterable[TelemetryEvent] | None = None,
    health_status: leyline_pb2.HealthStatus = leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
    health_summary: str = "",
) -> leyline_pb2.TelemetryPacket:
    """Construct a `TelemetryPacket` populated with metrics and optional events."""

    telemetry = leyline_pb2.TelemetryPacket(
        packet_id=packet_id,
        source_subsystem=source,
        level=level,
    )
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(datetime.now(tz=UTC))
    telemetry.timestamp.CopyFrom(ts)

    for metric in metrics:
        point = telemetry.metrics.add()
        point.name = metric.name
        point.value = metric.value
        if metric.unit:
            point.unit = metric.unit
        for key, value in metric.attributes.items():
            point.attributes[key] = value

    for event in events or ():
        entry = telemetry.events.add()
        entry.event_id = f"evt-{telemetry.events.__len__()}"
        entry.description = event.description
        entry.level = event.level
        for key, value in event.attributes.items():
            entry.attributes[key] = value

    telemetry.system_health.status = health_status
    telemetry.system_health.summary = health_summary
    return telemetry


__all__ = ["TelemetryMetric", "TelemetryEvent", "build_telemetry_packet"]
