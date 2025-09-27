from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from esper.core import EsperSettings
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, OonaMessage, StreamConfig
from esper.tamiyo import TamiyoPolicy, TamiyoPolicyConfig, TamiyoService
from esper.weatherlight.service_runner import WeatherlightService

pytestmark = pytest.mark.integration


class _LowCoveragePolicy(TamiyoPolicy):
    def __init__(self, avg: float = 0.1) -> None:
        super().__init__(TamiyoPolicyConfig(enable_compile=False))
        self._forced_avg = max(0.0, min(1.0, float(avg)))

    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_type=leyline_pb2.COMMAND_PAUSE,
        )
        command.issued_by = "tamiyo"
        command.issued_at.GetCurrentTime()
        coverage_value = self._forced_avg
        self._last_feature_coverage = {
            "global.loss": coverage_value,
            "seed.learning_rate": coverage_value,
        }
        self._last_action = {
            "action": 2.0,
            "param_delta": 0.0,
            "value_estimate": 0.0,
            "risk_score": 0.99,
        }
        return command


async def _build_oona_client(redis: FakeRedis, settings: EsperSettings) -> OonaClient:
    cfg = StreamConfig(
        normal_stream=settings.oona_normal_stream,
        emergency_stream=settings.oona_emergency_stream,
        telemetry_stream=settings.oona_telemetry_stream,
        policy_stream=settings.oona_policy_stream,
        group="weatherlight",
        consumer="weatherlight-test",
        dead_letter_stream="oona.deadletter",
        backpressure_drop_threshold=512,
        max_stream_length=2048,
    )
    client = OonaClient("redis://localhost", config=cfg, redis_client=redis)
    await client.ensure_consumer_group()
    return client


@pytest.mark.asyncio
async def test_weatherlight_tamiyo_low_coverage_emergency(monkeypatch) -> None:
    redis = FakeRedis()
    settings = EsperSettings()

    async def _patched_build(self: WeatherlightService) -> OonaClient:  # type: ignore[override]
        return await _build_oona_client(redis, settings)

    async def _noop_loop(self: WeatherlightService) -> None:  # type: ignore[override]
        while not self._shutdown_requested.is_set():
            await asyncio.sleep(0.05)

    monkeypatch.setattr(WeatherlightService, "_build_oona_client", _patched_build)
    monkeypatch.setattr(WeatherlightService, "_register_worker", lambda self, state: None)
    monkeypatch.setattr(WeatherlightService, "_kasmina_supervisor_loop", _noop_loop)
    monkeypatch.setattr(WeatherlightService, "_emergency_stream_loop", _noop_loop)
    monkeypatch.setattr(TamiyoService, "_ensure_blueprint_metadata_for_packet", lambda self, state: None)
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")

    service = WeatherlightService(settings=settings)
    await service.start()

    assert service._tamiyo_service is not None  # type: ignore[attr-defined]
    service._tamiyo_service._policy = _LowCoveragePolicy(avg=0.05)  # type: ignore[attr-defined]

    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-low-coverage",
    )

    service._tamiyo_service.evaluate_step(packet)  # type: ignore[attr-defined]

    await service._flush_telemetry_once()

    # Drain emergency stream entries and feed them back through the handler to emulate the supervisor loop.
    emergency_stream = settings.oona_emergency_stream
    entries = await redis.xrange(emergency_stream, count=16)
    assert entries, "expected emergency telemetry entries"
    for message_id, payload in entries:
        envelope = leyline_pb2.BusEnvelope()
        envelope.ParseFromString(payload[b"payload"])
        assert envelope.message_type == leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_TELEMETRY
        message = OonaMessage(
            stream=emergency_stream,
            message_id=message_id,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_TELEMETRY,
            payload=envelope.payload,
            attributes={"origin": envelope.attributes.get("origin", "tamiyo")},
        )
        await service._on_emergency_message(message)

    metrics_snapshot = await service._oona.metrics_snapshot()  # type: ignore[attr-defined]
    assert metrics_snapshot["emergency_published"] >= 1.0
    assert metrics_snapshot["publish_dropped"] == pytest.approx(0.0)

    telemetry_packet = await service._build_telemetry_packet()  # type: ignore[attr-defined]
    metric_map = {m.name: m.value for m in telemetry_packet.metrics}
    assert metric_map.get("weatherlight.emergency.telemetry_total", 0.0) >= 1.0
    assert metric_map.get("weatherlight.emergency.tamiyo_total", 0.0) >= 1.0

    await service.shutdown()
