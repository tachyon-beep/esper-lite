from datetime import datetime, timedelta, timezone

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
    assert (
        collected[0].message_type
        == leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_SYSTEM_STATE
    )
    assert await client.backlog("oona.normal") == 0

    await client.close()


@pytest.mark.asyncio
async def test_oona_kernel_prefetch_publish_and_consume() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        backpressure_drop_threshold=2,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    request = leyline_pb2.KernelPrefetchRequest(
        request_id="req-1",
        blueprint_id="BP-1",
        training_run_id="run-1",
    )
    request.issued_at.GetCurrentTime()
    assert await client.publish_kernel_prefetch_request(request)

    collected: list[OonaMessage] = []

    async def handler(message: OonaMessage) -> None:
        collected.append(message)

    await client.consume_kernel_requests(handler)
    assert len(collected) == 1
    payload = leyline_pb2.KernelPrefetchRequest()
    payload.ParseFromString(collected[0].payload)
    assert payload.request_id == "req-1"

    # Backpressure drop threshold = 2
    assert await client.publish_kernel_prefetch_request(request)
    # Without consuming, next publish should drop once depth hits threshold
    assert await client.publish_kernel_prefetch_request(request) is False
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
async def test_oona_publish_telemetry_routes_by_priority() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.TelemetryPacket(
        packet_id="pkt-crit",
        source_subsystem="kasmina",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
    )
    assert await client.publish_telemetry(
        packet,
        priority=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL,
    )
    assert await client.stream_length("oona.emergency") == 1
    assert await client.stream_length("oona.telemetry") == 0
    await client.close()


@pytest.mark.asyncio
async def test_tamiyo_high_priority_routes_to_emergency() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="tamiyo-routing-test",
    )
    client = OonaClient("redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.TelemetryPacket(
        packet_id="tamiyo-timeout",
        source_subsystem="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
    )
    packet.system_health.indicators["priority"] = "MESSAGE_PRIORITY_HIGH"
    packet.events.add(description="timeout_inference")
    assert await client.publish_telemetry(
        packet,
        priority=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH,
    )
    assert await client.stream_length("oona.emergency") == 1
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


@pytest.mark.asyncio
async def test_oona_retry_and_dead_letter() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        retry_max_attempts=2,
        dead_letter_stream="oona.deadletter",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    packet = leyline_pb2.SystemStatePacket(version=1)
    assert await client.publish_state(packet)

    async def failing_handler(_: OonaMessage) -> None:
        raise RuntimeError("boom")

    await client.consume(failing_handler)
    # Message was re-queued with attempt metadata (the stream retains earlier IDs)
    entries = await redis.xrevrange("oona.normal", count=1)
    envelope = leyline_pb2.BusEnvelope()
    envelope.ParseFromString(entries[0][1][b"payload"])
    assert envelope.attributes["attempt"] == "1"

    await client.consume(failing_handler)
    # Normal stream still retains historical entries but no active workloads
    assert await client.backlog("oona.normal") == 0
    assert await client.stream_length("oona.deadletter") == 1
    metrics = await client.metrics_snapshot()
    assert metrics["dead_letter_total"] >= 1.0
    await client.close()


@pytest.mark.asyncio
async def test_oona_publish_kernel_ready_and_error() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    ready = leyline_pb2.KernelArtifactReady(
        request_id="req-42",
        blueprint_id="BP-42",
        artifact_ref="/tmp/bp42",
        checksum="abc123",
        guard_digest="guard",
        prewarm_p50_ms=12.5,
        prewarm_p95_ms=18.2,
    )
    error = leyline_pb2.KernelArtifactError(
        request_id="req-43",
        blueprint_id="BP-43",
        reason="not found",
    )

    assert await client.publish_kernel_artifact_ready(ready)
    assert await client.publish_kernel_artifact_error(error)

    ready_msgs: list[OonaMessage] = []
    error_msgs: list[OonaMessage] = []

    async def ready_handler(message: OonaMessage) -> None:
        ready_msgs.append(message)

    async def error_handler(message: OonaMessage) -> None:
        error_msgs.append(message)

    await client.consume_kernel_ready(ready_handler)
    await client.consume_kernel_errors(error_handler)

    ready_payload = leyline_pb2.KernelArtifactReady()
    ready_payload.ParseFromString(ready_msgs[0].payload)
    assert ready_payload.request_id == "req-42"

    error_payload = leyline_pb2.KernelArtifactError()
    error_payload.ParseFromString(error_msgs[0].payload)
    assert error_payload.reason == "not found"
    await client.close()


@pytest.mark.asyncio
async def test_oona_kernel_prefetch_stale_requests_are_dropped() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        kernel_freshness_window_ms=1_000,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    request = leyline_pb2.KernelPrefetchRequest(
        request_id="req-stale",
        blueprint_id="BP-STALE",
        training_run_id="run-1",
    )
    stale_instant = datetime.now(timezone.utc) - timedelta(seconds=5)
    request.issued_at.FromDatetime(stale_instant)

    assert await client.publish_kernel_prefetch_request(request)

    collected: list[OonaMessage] = []

    async def handler(message: OonaMessage) -> None:
        collected.append(message)

    await client.consume_kernel_requests(handler, block_ms=0)
    assert not collected
    metrics = await client.metrics_snapshot()
    assert metrics["kernel_stale_dropped"] >= 1.0
    assert await client.backlog(client.kernel_request_stream) == 0
    await client.close()


@pytest.mark.asyncio
async def test_oona_kernel_prefetch_replays_are_suppressed() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    request = leyline_pb2.KernelPrefetchRequest(
        request_id="req-unique",
        blueprint_id="BP-1",
        training_run_id="run-1",
    )
    request.issued_at.GetCurrentTime()

    call_count = 0

    async def handler(_: OonaMessage) -> None:
        nonlocal call_count
        call_count += 1

    assert await client.publish_kernel_prefetch_request(request)
    await client.consume_kernel_requests(handler, block_ms=0)
    assert call_count == 1

    # Replay the same request ID; handler should not run again.
    replay = leyline_pb2.KernelPrefetchRequest()
    replay.CopyFrom(request)
    replay.issued_at.GetCurrentTime()
    assert await client.publish_kernel_prefetch_request(replay)
    await client.consume_kernel_requests(handler, block_ms=0)
    assert call_count == 1
    metrics = await client.metrics_snapshot()
    assert metrics["kernel_replay_dropped"] >= 1.0
    await client.close()


@pytest.mark.asyncio
async def test_oona_housekeeping_trims_old_messages() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
        message_ttl_ms=1,
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    # Inject an old entry with a deterministic ID
    await redis.xadd("oona.normal", {"payload": b"old"}, id="0-1")
    trimmed = await client.housekeeping()
    assert trimmed["oona.normal"] >= 1.0
    await client.close()


@pytest.mark.asyncio
async def test_oona_publish_breaker_records_failures(monkeypatch) -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    async def failing_xadd(*args, **kwargs):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(client._redis, "xadd", failing_xadd)
    packet = leyline_pb2.SystemStatePacket(version=1)
    assert await client.publish_state(packet) is False
    metrics = await client.metrics_snapshot()
    assert metrics["breaker_publish_open"] >= 1.0
    await client.close()


@pytest.mark.asyncio
async def test_oona_emits_metrics_telemetry_packet() -> None:
    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="oona-test",
    )
    client = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await client.ensure_consumer_group()

    assert await client.emit_metrics_telemetry()

    collected: list[OonaMessage] = []

    async def telemetry_handler(message: OonaMessage) -> None:
        collected.append(message)

    await client.consume(telemetry_handler, stream=client.telemetry_stream, block_ms=0)
    assert collected

    packet = leyline_pb2.TelemetryPacket()
    packet.ParseFromString(collected[0].payload)
    metric_names = {metric.name for metric in packet.metrics}
    assert "oona.queue_depth_kernel_requests" in metric_names
    assert "oona.publish_total" in metric_names
    await client.close()
