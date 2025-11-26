from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig


@pytest.mark.asyncio
async def test_emergency_burst_rate_limit() -> None:
    redis = FakeRedis()
    cfg = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="oona-test",
        consumer="test",
        emergency_max_per_min=5,
    )
    client = OonaClient("redis://localhost", config=cfg, redis_client=redis)
    await client.ensure_consumer_group()
    # Publish more than budget in a tight loop
    pkt = leyline_pb2.TelemetryPacket(
        packet_id="p", source_subsystem="x", level=leyline_pb2.TELEMETRY_LEVEL_CRITICAL
    )
    sent = 0
    dropped = 0
    for _ in range(10):
        ok = await client.publish_telemetry(
            pkt, priority=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
        )
        if ok:
            sent += 1
        else:
            dropped += 1
    # Expect not all passed due to rate limit
    assert sent <= 5
    metrics = await client.metrics_snapshot()
    assert metrics.get("emergency_published", 0.0) >= 1.0
    assert metrics.get("emergency_rate_dropped", 0.0) >= 1.0
    await client.close()
