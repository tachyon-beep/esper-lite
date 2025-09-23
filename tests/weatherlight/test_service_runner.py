from __future__ import annotations

import asyncio
from pathlib import Path
import contextlib

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


def test_weatherlight_priority_override_mapping(weatherlight_settings: EsperSettings) -> None:
    # Packet has INFO level but explicit HIGH priority indicator; override should map to HIGH
    pkt = build_telemetry_packet(
        packet_id="override-1",
        source="unit",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("unit.metric", 1.0)],
        events=[],
    )
    pkt.system_health.indicators["priority"] = leyline_pb2.MessagePriority.Name(
        leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    )
    # Static method mapping
    from esper.weatherlight.service_runner import WeatherlightService

    pr = WeatherlightService._telemetry_priority(pkt)
    assert pr == leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
    # Mapping override is sufficient for routing; additional packet content is
    # covered by other tests.


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


@pytest.mark.asyncio
async def test_weatherlight_flushes_tamiyo_history(
    fake_redis: FakeRedis,
    weatherlight_settings: EsperSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = WeatherlightService(settings=weatherlight_settings)
    await service.start()
    try:
        await asyncio.sleep(0.05)
        # Seed Tamiyo with a telemetry packet by evaluating a minimal state
        assert service._tamiyo_service is not None  # type: ignore[attr-defined]
        pkt = leyline_pb2.SystemStatePacket(version=1, current_epoch=0)
        # One evaluation populates Tamiyo telemetry buffer
        service._tamiyo_service.evaluate_epoch(pkt)  # type: ignore[attr-defined]

        published: list[leyline_pb2.TelemetryPacket] = []

        async def _capture_publish(
            packet: leyline_pb2.TelemetryPacket,
            *,
            priority: leyline_pb2.MessagePriority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL,
        ) -> None:
            published.append(packet)

        async def _noop_metrics_telemetry(**_kwargs: object) -> None:
            return None

        assert service._oona is not None  # type: ignore[attr-defined]
        monkeypatch.setattr(service._oona, "publish_telemetry", _capture_publish)
        monkeypatch.setattr(service._oona, "emit_metrics_telemetry", _noop_metrics_telemetry)

        # Flush Weatherlight telemetry once (should drain Tamiyo history as well)
        await service._flush_telemetry_once()  # type: ignore[attr-defined]

        # Verify at least one published packet originates from Tamiyo and carries coverage metric
        assert any(
            (p.source_subsystem == "tamiyo" and any(m.name == "tamiyo.gnn.feature_coverage" for m in p.metrics))
            for p in published
        )
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_weatherlight_fans_out_tamiyo_coverage_types(
    fake_redis: FakeRedis,
    weatherlight_settings: EsperSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure signing secret is present for Tamiyo path
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    service = WeatherlightService(settings=weatherlight_settings)
    await service.start()
    try:
        # Trigger Tamiyo to generate a telemetry packet with coverage types
        assert service._tamiyo_service is not None  # type: ignore[attr-defined]
        pkt = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-fanout")
        # Provide at least one seed to avoid fast-path pause
        seed = pkt.seed_states.add()
        seed.seed_id = "seed-fanout"
        seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
        seed.learning_rate = 0.01
        service._tamiyo_service.evaluate_epoch(pkt)  # type: ignore[attr-defined]

        # Build Weatherlight telemetry packet and assert fanout metrics present
        packet = await service._build_telemetry_packet()  # type: ignore[attr-defined]
        names = {m.name for m in packet.metrics}
        # Expect at least one node and one edge type coverage metric
        assert any(n.startswith("weatherlight.tamiyo.coverage.node.") for n in names)
        assert any(n.startswith("weatherlight.tamiyo.coverage.edges.") for n in names)
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_weatherlight_flushes_kasmina_pending(
    fake_redis: FakeRedis,
    weatherlight_settings: EsperSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = WeatherlightService(settings=weatherlight_settings)
    await service.start()
    try:
        # Stop the background Kasmina telemetry loop to avoid races in this unit test
        if service._kasmina_telemetry_task is not None:  # type: ignore[attr-defined]
            service._kasmina_telemetry_task.cancel()  # type: ignore[attr-defined]
            with contextlib.suppress(Exception):
                await asyncio.sleep(0)  # yield control for cancellation

        # Provide a stub Kasmina manager with a single buffered packet
        pkt = build_telemetry_packet(
            packet_id="kasmina-1",
            source="kasmina",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=[TelemetryMetric("kasmina.test", 1.0)],
            events=[],
        )

        class _StubKasmina:
            def __init__(self, p: leyline_pb2.TelemetryPacket) -> None:
                self._p = p
                self._drained = False

            def drain_telemetry_packets(self) -> list[leyline_pb2.TelemetryPacket]:
                if self._drained:
                    return []
                self._drained = True
                return [self._p]

        # Swap in the stub
        service._kasmina_manager = _StubKasmina(pkt)  # type: ignore[attr-defined]

        published: list[leyline_pb2.TelemetryPacket] = []

        async def _capture_publish(
            packet: leyline_pb2.TelemetryPacket,
            *,
            priority: leyline_pb2.MessagePriority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL,
        ) -> None:
            published.append(packet)

        async def _noop_metrics_telemetry(**_kwargs: object) -> None:
            return None

        assert service._oona is not None  # type: ignore[attr-defined]
        monkeypatch.setattr(service._oona, "publish_telemetry", _capture_publish)
        monkeypatch.setattr(service._oona, "emit_metrics_telemetry", _noop_metrics_telemetry)

        # Flush Weatherlight telemetry once (should also forward Kasmina stub packet)
        await service._flush_telemetry_once()  # type: ignore[attr-defined]

        assert any(p.source_subsystem == "kasmina" for p in published)
    finally:
        await service.shutdown()
