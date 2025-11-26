from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from esper.core import TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig


def _critical_packet(idx: int) -> leyline_pb2.TelemetryPacket:
    return build_telemetry_packet(
        packet_id=f"telemetry-critical-{idx}",
        source="test-harness",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
        metrics=[TelemetryMetric("load_test", float(idx))],
        events=[],
        health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED,
        health_summary="synthetic_emergency",
        health_indicators={"priority": "MESSAGE_PRIORITY_CRITICAL"},
    )


@pytest.mark.asyncio
async def test_emergency_stream_load_limits_backlog() -> None:
    redis = FakeRedis()
    cfg = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="telemetry-load",
        consumer="under-test",
        emergency_max_per_min=3,
        emergency_threshold=8,
    )
    client = OonaClient("redis://localhost", config=cfg, redis_client=redis)
    await client.ensure_consumer_group()

    results: list[bool] = []
    for idx in range(10):
        packet = _critical_packet(idx)
        ok = await client.publish_telemetry(
            packet, priority=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
        )
        results.append(ok)

    successful = sum(1 for ok in results if ok)
    dropped = len(results) - successful
    assert successful == cfg.emergency_max_per_min
    assert dropped == len(results) - cfg.emergency_max_per_min

    metrics = await client.metrics_snapshot()
    assert metrics["emergency_published"] == pytest.approx(float(successful))
    assert metrics["emergency_rate_dropped"] == pytest.approx(float(dropped))
    assert metrics["queue_depth_emergency"] == pytest.approx(float(successful))
    assert metrics["emergency_tokens_remaining"] <= 1.0
    assert metrics["emergency_bucket_capacity_per_min"] == pytest.approx(float(cfg.emergency_max_per_min))

    # Allow the token bucket to refill enough for another publish
    await asyncio.sleep(1.1)
    packet = _critical_packet(999)
    ok_after_refill = await client.publish_telemetry(
        packet, priority=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
    )
    # Depending on rate, this may still be dropped, but bucket should have more tokens than before
    metrics_after = await client.metrics_snapshot()
    assert metrics_after["emergency_tokens_remaining"] >= metrics["emergency_tokens_remaining"]
    await client.close()
