from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakeredis.aioredis import FakeRedis

from esper.core import EsperSettings, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.weatherlight.service_runner import WeatherlightService


@pytest.fixture(autouse=True)
def _ensure_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")


@pytest.fixture()
async def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    client = FakeRedis(decode_responses=False)
    monkeypatch.setattr("esper.oona.messaging.aioredis.from_url", lambda url: client)
    yield client
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()
    else:  # pragma: no cover - fallback for older fakeredis
        client.close()


@pytest.fixture()
def weatherlight_settings(tmp_path: Path) -> EsperSettings:
    urza_root = tmp_path / "urza"
    artifact_dir = urza_root / "artifacts"
    database_url = f"sqlite:///{(urza_root / 'catalog.db').as_posix()}"
    return EsperSettings(
        redis_url="redis://localhost:6379/0",
        urza_artifact_dir=str(artifact_dir),
        urza_database_url=database_url,
    )


@pytest.mark.asyncio
async def test_weatherlight_start_stop(fake_redis: FakeRedis, weatherlight_settings: EsperSettings) -> None:
    service = WeatherlightService(settings=weatherlight_settings)
    await service.start()
    run_task = asyncio.create_task(service.run())
    # Allow the workers to enter their run loops.
    await asyncio.sleep(0.1)
    assert all(state.task is not None for state in service._workers.values())
    service.initiate_shutdown()
    await run_task
    assert service._shutdown_complete.is_set()


@pytest.mark.asyncio
async def test_weatherlight_builds_telemetry_packet(fake_redis: FakeRedis, weatherlight_settings: EsperSettings) -> None:
    service = WeatherlightService(settings=weatherlight_settings)
    await service.start()
    try:
        await asyncio.sleep(0.1)
        packet = await service._build_telemetry_packet()
    finally:
        await service.shutdown()
    metric_names = {metric.name for metric in packet.metrics}
    assert "weatherlight.tasks.running" in metric_names
    assert packet.source_subsystem == "weatherlight"
    assert "urza.library.cache_size" in metric_names


@pytest.mark.asyncio
async def test_weatherlight_streams_tezzeret_packets(
    fake_redis: FakeRedis,
    weatherlight_settings: EsperSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = WeatherlightService(settings=weatherlight_settings)
    await service.start()
    try:
        await asyncio.sleep(0.1)

        class _DummyForge:
            def __init__(self) -> None:
                self.count = 0

            def metrics_snapshot(self) -> dict[str, float]:
                return {"tezzeret.jobs.total": 3.0}

            def build_telemetry_packet(self) -> leyline_pb2.TelemetryPacket:
                self.count += 1
                return build_telemetry_packet(
                    packet_id=f"tezzeret-{self.count}",
                    source="tezzeret",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    metrics=[TelemetryMetric("tezzeret.jobs.total", 3.0)],
                    events=[],
                )

        forge = _DummyForge()
        service.connect_tezzeret_forge(forge)

        published: list[tuple[str, leyline_pb2.MessagePriority]] = []

        async def _fake_publish(
            packet: leyline_pb2.TelemetryPacket,
            *,
            priority: leyline_pb2.MessagePriority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL,
        ) -> None:
            published.append((packet.source_subsystem, priority))

        async def _noop_metrics_telemetry(**_kwargs: object) -> None:
            return None

        monkeypatch.setattr(service._oona, "publish_telemetry", _fake_publish)
        monkeypatch.setattr(service._oona, "emit_metrics_telemetry", _noop_metrics_telemetry)

        await service._flush_telemetry_once()

        assert ("weatherlight", leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL) in published
        assert ("tezzeret", leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH) in published

        packet = await service._build_telemetry_packet()
        metric_names = {metric.name for metric in packet.metrics}
        assert "tezzeret.jobs.total" in metric_names
    finally:
        await service.shutdown()
