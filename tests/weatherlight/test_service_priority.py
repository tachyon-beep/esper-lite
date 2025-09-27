from __future__ import annotations

import asyncio

import pytest

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.weatherlight.service_runner import WeatherlightService
from esper.oona import OonaMessage


class _FakeOona:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def publish_telemetry(self, packet, priority: int) -> bool:  # type: ignore[no-untyped-def]
        self.calls.append((getattr(packet, "packet_id", ""), int(priority)))
        return True

    async def emit_metrics_telemetry(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    async def metrics_snapshot(self) -> dict[str, float]:  # type: ignore[no-untyped-def]
        return {}


class _FakeKasmina:
    def __init__(self, packets) -> None:
        self._packets = list(packets)

    def drain_telemetry_packets(self):  # type: ignore[no-untyped-def]
        pkts, self._packets = self._packets, []
        return pkts


def _kasmina_packet(packet_id: str, level: int, *, priority_indicator: str | None = None):
    pkt = build_telemetry_packet(
        packet_id=packet_id,
        source="kasmina",
        level=level,
        metrics=[TelemetryMetric("dummy", 1.0)],
        events=[TelemetryEvent("evt")],
        health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
        health_summary="nominal",
        health_indicators=(
            {"seed_id": "seed-x"}
            if priority_indicator is None
            else {"seed_id": "seed-x", "priority": priority_indicator}
        ),
    )
    return pkt


def _make_urza_worker():
    class _UrzaWorker:
        def __init__(self) -> None:
            self.metrics = type(
                "M", (), {"hits": 0, "misses": 0, "errors": 0, "latency_ms": 0.0}
            )()

    return _UrzaWorker()


@pytest.mark.asyncio
async def test_weatherlight_routes_kasmina_by_priority() -> None:
    service = WeatherlightService()
    fake = _FakeOona()
    service._oona = fake  # type: ignore[attr-defined]
    service._urza_worker = _make_urza_worker()  # type: ignore[attr-defined]

    # Build Kasmina packets
    crit = _kasmina_packet(
        "kas-critical",
        leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
        priority_indicator=leyline_pb2.MessagePriority.Name(
            leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        ),
    )
    warn = _kasmina_packet(
        "kas-warning",
        leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
        priority_indicator=leyline_pb2.MessagePriority.Name(
            leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
        ),
    )
    info = _kasmina_packet(
        "kas-info",
        leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    )
    service._kasmina_manager = _FakeKasmina([crit, warn, info])  # type: ignore[attr-defined]

    await service._flush_telemetry_once()

    # Collect priorities by packet_id
    priorities = {pid: prio for pid, prio in fake.calls}
    # Ensure our Kasmina packets were published with expected priorities
    assert priorities.get("kas-critical") == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
    assert priorities.get("kas-warning") == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    assert priorities.get("kas-info") == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL

    counters = service.telemetry_priority_counters()
    assert counters["critical"] >= 1
    assert counters["high"] >= 1
    assert counters["normal"] >= 1
    assert service.telemetry_publish_failures == 0


@pytest.mark.asyncio
async def test_weatherlight_tracks_tamiyo_emergency_telemetry() -> None:
    service = WeatherlightService()
    fake = _FakeOona()
    service._oona = fake  # type: ignore[attr-defined]
    service._urza_worker = _make_urza_worker()  # type: ignore[attr-defined]

    packet = build_telemetry_packet(
        packet_id="tamiyo-emergency",
        source="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
        metrics=[TelemetryMetric("tamiyo.gnn.feature_coverage", 0.05, unit="ratio")],
        events=[
            TelemetryEvent(
                description="degraded_inputs",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                attributes={"coverage_avg": "0.05"},
            )
        ],
        health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_CRITICAL,
        health_summary="degraded_inputs",
        health_indicators={
            "priority": leyline_pb2.MessagePriority.Name(
                leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
            )
        },
    )
    message = OonaMessage(
        stream="oona.emergency",
        message_id="1-0",
        message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_TELEMETRY,
        payload=packet.SerializeToString(),
        attributes={"priority": "critical"},
    )

    await service._on_emergency_message(message)

    assert service._emergency_telemetry_counts.get("tamiyo") == 1

    telemetry_packet = await service._build_telemetry_packet()  # type: ignore[attr-defined]
    metric_map = {m.name: m.value for m in telemetry_packet.metrics}
    assert metric_map.get("weatherlight.emergency.telemetry_total") == pytest.approx(1.0)
    assert metric_map.get("weatherlight.emergency.tamiyo_total") == pytest.approx(1.0)

    await service.shutdown()
