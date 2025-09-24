from __future__ import annotations

import asyncio

import pytest

from esper.core import TelemetryMetric, TelemetryEvent, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.weatherlight.service_runner import WeatherlightService


class _FakeOona:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def publish_telemetry(self, packet, priority: int) -> None:  # type: ignore[no-untyped-def]
        self.calls.append((getattr(packet, "packet_id", ""), int(priority)))

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
        health_indicators={"seed_id": "seed-x"} if priority_indicator is None else {"seed_id": "seed-x", "priority": priority_indicator},
    )
    return pkt


@pytest.mark.asyncio
async def test_weatherlight_routes_kasmina_by_priority() -> None:
    service = WeatherlightService()
    fake = _FakeOona()
    service._oona = fake  # type: ignore[attr-defined]
    # Provide a minimal Urza worker metrics stub
    class _UrzaWorker:
        def __init__(self) -> None:
            self.metrics = type("M", (), {"hits": 0, "misses": 0, "errors": 0, "latency_ms": 0.0})()

    service._urza_worker = _UrzaWorker()  # type: ignore[attr-defined]

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
