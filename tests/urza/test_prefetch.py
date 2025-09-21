from __future__ import annotations

from pathlib import Path
import hashlib

import pytest
from fakeredis.aioredis import FakeRedis

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.urza import UrzaLibrary, UrzaPrefetchWorker


def _make_descriptor(blueprint_id: str) -> BlueprintDescriptor:
    descriptor = BlueprintDescriptor()
    descriptor.blueprint_id = blueprint_id
    descriptor.name = f"Blueprint {blueprint_id}"
    descriptor.tier = BlueprintTier.BLUEPRINT_TIER_SAFE
    return descriptor


@pytest.mark.asyncio
async def test_urza_prefetch_worker_emits_ready(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pt"
    data = b"model-bytes"
    artifact.write_bytes(data)
    checksum = hashlib.sha256(data).hexdigest()

    library = UrzaLibrary(root=tmp_path)
    descriptor = _make_descriptor("BP-123")
    update = leyline_pb2.KernelCatalogUpdate(
        blueprint_id=descriptor.blueprint_id,
        artifact_ref=str(artifact),
        checksum=checksum,
        guard_digest="guard-123",
        compile_ms=11.0,
        prewarm_ms=24.0,
    )
    library.save(descriptor, artifact, catalog_update=update)

    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="prefetch",
    )
    oona = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    worker = UrzaPrefetchWorker(oona, library)

    request = leyline_pb2.KernelPrefetchRequest(
        request_id="req-123",
        blueprint_id="BP-123",
        training_run_id="run-1",
    )
    request.issued_at.GetCurrentTime()
    assert await oona.publish_kernel_prefetch_request(request)

    await worker.poll_once()

    ready_messages: list[leyline_pb2.KernelArtifactReady] = []

    async def ready_handler(message):
        payload = leyline_pb2.KernelArtifactReady()
        payload.ParseFromString(message.payload)
        ready_messages.append(payload)

    await oona.consume_kernel_ready(ready_handler, block_ms=0)

    assert ready_messages
    ready = ready_messages[0]
    assert ready.request_id == "req-123"
    assert ready.blueprint_id == "BP-123"
    assert ready.artifact_ref.endswith("artifact.pt")
    assert ready.guard_digest == "guard-123"
    assert ready.prewarm_p50_ms == pytest.approx(24.0)
    assert ready.prewarm_p95_ms == pytest.approx(24.0)
    assert worker.metrics.hits == 1
    assert worker.metrics.latency_ms >= 0.0

    await oona.close()


@pytest.mark.asyncio
async def test_urza_prefetch_worker_emits_error_when_missing(tmp_path: Path) -> None:
    library = UrzaLibrary(root=tmp_path)

    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="prefetch",
    )
    oona = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    worker = UrzaPrefetchWorker(oona, library)

    request = leyline_pb2.KernelPrefetchRequest(
        request_id="req-missing",
        blueprint_id="missing",
        training_run_id="run-1",
    )
    request.issued_at.GetCurrentTime()
    assert await oona.publish_kernel_prefetch_request(request)

    await worker.poll_once()

    error_messages: list[leyline_pb2.KernelArtifactError] = []

    async def error_handler(message):
        payload = leyline_pb2.KernelArtifactError()
        payload.ParseFromString(message.payload)
        error_messages.append(payload)

    await oona.consume_kernel_errors(error_handler, block_ms=0)

    assert error_messages
    assert error_messages[0].reason == "missing_artifact"
    assert worker.metrics.misses == 1

    await oona.close()


@pytest.mark.asyncio
async def test_urza_prefetch_worker_reports_checksum_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"original")
    checksum = hashlib.sha256(b"original").hexdigest()

    library = UrzaLibrary(root=tmp_path)
    descriptor = _make_descriptor("BP-MISMATCH")
    update = leyline_pb2.KernelCatalogUpdate(
        blueprint_id=descriptor.blueprint_id,
        artifact_ref=str(artifact),
        checksum=checksum,
    )
    library.save(descriptor, artifact, catalog_update=update)

    # Corrupt artifact on disk
    artifact.write_bytes(b"tampered")

    redis = FakeRedis()
    config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        group="prefetch",
    )
    oona = OonaClient(redis_url="redis://localhost", config=config, redis_client=redis)
    await oona.ensure_consumer_group()

    worker = UrzaPrefetchWorker(oona, library)

    request = leyline_pb2.KernelPrefetchRequest(
        request_id="req-bad",
        blueprint_id="BP-MISMATCH",
        training_run_id="run-1",
    )
    request.issued_at.GetCurrentTime()
    assert await oona.publish_kernel_prefetch_request(request)

    await worker.poll_once()

    error_messages: list[leyline_pb2.KernelArtifactError] = []

    async def error_handler(message):
        payload = leyline_pb2.KernelArtifactError()
        payload.ParseFromString(message.payload)
        error_messages.append(payload)

    await oona.consume_kernel_errors(error_handler, block_ms=0)

    assert error_messages
    assert error_messages[0].reason == "checksum_mismatch"
    assert worker.metrics.errors >= 1

    await oona.close()
