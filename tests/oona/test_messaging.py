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

    await client.close()
