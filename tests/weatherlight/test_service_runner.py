from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakeredis.aioredis import FakeRedis

from esper.core import EsperSettings
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
