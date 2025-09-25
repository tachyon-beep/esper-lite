import inspect
import os

import pytest
import redis.asyncio as aioredis

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, OonaMessage, StreamConfig


async def _shutdown_redis(redis: aioredis.Redis) -> None:
    close = getattr(redis, "aclose", None)
    if close is not None:
        await close()
        return
    result = redis.close()
    if inspect.isawaitable(result):
        await result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_oona_docker_compose_redis_round_trip() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    # Use aggressive timeouts so environments without Redis don't hang
    redis = aioredis.from_url(
        redis_url,
        socket_connect_timeout=1.0,
        socket_timeout=2.0,
    )
    try:
        await redis.ping()
    except Exception:  # pragma: no cover - environment dependent
        await _shutdown_redis(redis)
        if os.getenv("REQUIRE_REDIS"):
            pytest.fail("REQUIRE_REDIS=1 set but Redis is not available at REDIS_URL")
        pytest.skip("Redis instance not available at REDIS_URL")

    config = StreamConfig(
        normal_stream="oona.int.normal",
        emergency_stream="oona.int.emergency",
        telemetry_stream="oona.int.telemetry",
        policy_stream="oona.int.policy",
        group="oona-int",
        emergency_threshold=5,
        max_stream_length=100,
    )
    client = OonaClient(redis_url=redis_url, config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=0,
        training_run_id="integration",
    )
    for _ in range(6):
        await client.publish_state(packet)

    # threshold triggers emergency routing
    assert await client.stream_length("oona.int.emergency") >= 1

    collected: list[OonaMessage] = []

    async def handler(message: OonaMessage) -> None:
        collected.append(message)

    await client.consume(handler, stream="oona.int.emergency")
    assert collected
    assert collected[0].message_type == leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_SYSTEM_STATE

    await client.close()
    await redis.delete(config.normal_stream)
    await redis.delete(config.emergency_stream)
    await _shutdown_redis(redis)
