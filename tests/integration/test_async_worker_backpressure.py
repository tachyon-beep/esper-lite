from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from esper.core import TelemetryMetric, build_telemetry_packet
from esper.core.async_runner import AsyncWorker
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig

pytestmark = pytest.mark.integration


async def _publish_batch(client: OonaClient, count: int, *, concurrency: int = 8) -> None:
    packets = [
        build_telemetry_packet(
            packet_id=f"telemetry-{idx}",
            source="worker-test",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=[TelemetryMetric("worker.load", float(idx))],
        )
        for idx in range(count)
    ]

    async def _publish_packet(pkt: leyline_pb2.TelemetryPacket) -> bool:
        priority = None
        if pkt.system_health.indicators.get("priority"):
            priority = leyline_pb2.MessagePriority.Value(pkt.system_health.indicators["priority"])
        return await client.publish_telemetry(pkt, priority=priority)

    worker = AsyncWorker(max_concurrency=concurrency, name="test-worker")
    handles = [worker.submit(_publish_packet, pkt) for pkt in packets]
    try:
        for handle in handles:
            assert handle.result(timeout=5.0) is True
    finally:
        worker.close()


@pytest.mark.asyncio
async def test_async_worker_backpressure_metrics() -> None:
    redis = FakeRedis()
    cfg = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="backpressure",
        consumer="backpressure-client",
        backpressure_drop_threshold=128,
        max_stream_length=512,
    )
    client = OonaClient("redis://localhost", config=cfg, redis_client=redis)
    await client.ensure_consumer_group()

    await _publish_batch(client, 96, concurrency=6)

    metrics = await client.metrics_snapshot()
    assert metrics["publish_total"] == pytest.approx(96.0)
    assert metrics["publish_dropped"] == pytest.approx(0.0)
    assert metrics["queue_depth_max"] <= float(cfg.backpressure_drop_threshold)

    # Drain pending entries to avoid interference with subsequent tests
    await redis.delete(cfg.normal_stream)
    await client.close()
