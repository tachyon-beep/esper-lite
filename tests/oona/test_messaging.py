import pytest
from fakeredis.aioredis import FakeRedis

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, OonaMessage, StreamConfig


@pytest.mark.asyncio
async def test_oona_publish_and_consume() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        consumer="tester",
        max_stream_length=50,
        emergency_threshold=10,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=0, training_run_id="run")
    await client.publish_state(packet)

    collected: list[OonaMessage] = []

    async def handler(message: OonaMessage) -> None:
        collected.append(message)

    await client.consume(handler)

    assert collected
    restored = leyline_pb2.SystemStatePacket()
    restored.ParseFromString(collected[0].payload)
    assert restored.version == 1
    assert await client.backlog("oona.normal") == 0

    await client.close()


@pytest.mark.asyncio
async def test_oona_emergency_threshold() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        emergency_threshold=1,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1)
    await client.publish_state(packet)  # fills normal stream
    await client.publish_state(packet)  # should overflow to emergency

    normal_len = await client.stream_length("oona.normal")
    emergency_len = await client.stream_length("oona.emergency")
    assert normal_len == 1
    assert emergency_len == 1
    await client.close()


@pytest.mark.asyncio
async def test_oona_max_stream_length_trims() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        max_stream_length=5,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1)
    for _ in range(10):
        await client.publish_state(packet)

    assert await client.stream_length("oona.normal") <= 5
    await client.close()
