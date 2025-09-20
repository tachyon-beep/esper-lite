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
        telemetry_stream="oona.telemetry",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=0, training_run_id="run")
    assert await client.publish_state(packet)

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
    assert await client.publish_state(packet)  # fills normal stream
    assert await client.publish_state(packet)  # should overflow to emergency

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
        telemetry_stream="oona.telemetry",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1)
    for _ in range(10):
        assert await client.publish_state(packet)

    assert await client.stream_length("oona.normal") <= 5
    await client.close()


@pytest.mark.asyncio
async def test_oona_publish_telemetry() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.TelemetryPacket(
        packet_id="pkt-telemetry",
        source_subsystem="tolaria",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    )
    assert await client.publish_telemetry(packet)
    assert await client.stream_length("oona.telemetry") == 1
    await client.close()


@pytest.mark.asyncio
async def test_oona_publish_policy_update() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        policy_stream="oona.policy",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    update = leyline_pb2.PolicyUpdate(
        version=1,
        policy_id="policy-1",
        training_run_id="run-1",
        tamiyo_policy_version="policy-v2",
    )
    assert await client.publish_policy_update(update)
    assert await client.stream_length("oona.policy") == 1
    await client.close()


@pytest.mark.asyncio
async def test_oona_backpressure_reroute_and_metrics() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        emergency_threshold=2,
        max_stream_length=100,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1)
    assert await client.publish_state(packet)
    assert await client.publish_state(packet)
    assert await client.stream_length("oona.normal") == 2

    assert await client.publish_state(packet)
    assert await client.stream_length("oona.emergency") == 1

    metrics = await client.metrics_snapshot()
    assert metrics["publish_rerouted"] >= 1.0
    assert metrics["queue_depth_max"] >= 2.0
    await client.close()


@pytest.mark.asyncio
async def test_oona_backpressure_drop_threshold() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        backpressure_drop_threshold=5,
        max_stream_length=100,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1)
    for _ in range(5):
        assert await client.publish_state(packet)

    dropped = await client.publish_state(packet)
    assert dropped is False

    metrics = await client.metrics_snapshot()
    assert metrics["publish_dropped"] >= 1.0
    assert metrics["queue_depth_normal"] >= 5.0
    await client.close()
