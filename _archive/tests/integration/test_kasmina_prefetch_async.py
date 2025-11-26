from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from esper.core import AsyncWorker
from esper.kasmina.prefetch import KasminaPrefetchCoordinator
from esper.kasmina.seed_manager import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, OonaMessage, StreamConfig


class _Harness:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.requests: list[str] = []
        self.pending_ready: list[str] = []
        self.clients: list[_WorkerClientStub] = []
        self.coordinator: KasminaPrefetchCoordinator | None = None


class _WorkerClientStub:
    def __init__(self, role: str, harness: _Harness) -> None:
        self.role = role
        self.harness = harness
        self.harness.clients.append(self)
        self.closed = False

    async def ensure_consumer_group(self) -> None:  # pragma: no cover - simple stub
        self.harness.events.append(f"ensure:{self.role}")

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        self.harness.requests.append(request.request_id)
        self.harness.pending_ready.append(request.request_id)

    async def consume_kernel_ready(self, handler, *, block_ms: int = 0) -> None:
        if self.harness.pending_ready:
            request_id = self.harness.pending_ready.pop(0)
            ready = leyline_pb2.KernelArtifactReady(request_id=request_id)
            message = OonaMessage(
                stream="ready",
                message_id="1-1",
                message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY,
                payload=ready.SerializeToString(),
            )
            await handler(message)
            self.harness.events.append(f"ready:{request_id}")
            if self.harness.coordinator is not None:
                self.harness.coordinator._running = False  # pylint: disable=protected-access
        await asyncio.sleep(0.01)

    async def consume_kernel_errors(self, handler, *, block_ms: int = 0) -> None:
        await asyncio.sleep(0.01)

    async def close(self) -> None:
        self.closed = True


class _SpawnableOonaStub:
    def __init__(self, harness: _Harness) -> None:
        self.harness = harness
        self.spawn_calls: list[str] = []

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        self.harness.requests.append(request.request_id)
        self.harness.pending_ready.append(request.request_id)

    def spawn(self, *, consumer_suffix: str | None = None) -> _WorkerClientStub:
        suffix = consumer_suffix or "spawn"
        self.spawn_calls.append(suffix)
        return _WorkerClientStub(suffix, self.harness)

    async def ensure_consumer_group(self) -> None:  # pragma: no cover - unused
        return None

    async def close(self) -> None:  # pragma: no cover - unused
        return None


class _FakeRuntime:
    def fetch_kernel(self, blueprint_id: str):  # pragma: no cover - unused stub
        return "module", 0.0


class _FakeSeedManager(KasminaSeedManager):
    def __init__(self) -> None:
        super().__init__(runtime=_FakeRuntime(), nonce_max_entries=100)
        self.processed: list[tuple[str, str]] = []

    def process_prefetch_ready(self, ready: leyline_pb2.KernelArtifactReady) -> None:
        self.processed.append((ready.request_id, "ready"))

    def process_prefetch_error(self, error: leyline_pb2.KernelArtifactError) -> None:
        self.processed.append((error.request_id, "error"))


@pytest.mark.asyncio
async def test_prefetch_async_worker_ready_flow(monkeypatch) -> None:
    harness = _Harness()
    manager = _FakeSeedManager()
    worker = AsyncWorker(max_concurrency=4, name="kasmina-prefetch-test")
    oona = _SpawnableOonaStub(harness)
    coordinator = KasminaPrefetchCoordinator(manager, oona, async_worker=worker)
    harness.coordinator = coordinator
    manager.set_prefetch(coordinator)

    coordinator.start()
    coordinator.request_kernel("seed-1", "bp-1", training_run_id="run-1")
    await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)

    await coordinator.close()
    await oona.close()
    worker.shutdown(cancel_pending=True)

    assert any(event.startswith("ready:") for event in harness.events)
    assert manager.processed and manager.processed[0][1] == "ready"

@pytest.mark.asyncio
async def test_prefetch_async_worker_shutdown_cancels(monkeypatch) -> None:
    harness = _Harness()
    manager = _FakeSeedManager()
    worker = AsyncWorker(max_concurrency=4, name="kasmina-prefetch-test")
    oona = _SpawnableOonaStub(harness)
    coordinator = KasminaPrefetchCoordinator(manager, oona, async_worker=worker)
    harness.coordinator = coordinator
    manager.set_prefetch(coordinator)

    # Prevent ready handler from draining requests so coordinator stays running
    async def consume_noop(*args, **kwargs):
        await asyncio.sleep(0.05)

    coordinator.start()
    coordinator.request_kernel("seed-1", "bp-1", training_run_id="run-1")
    # Monkeypatch the spawned clients to avoid producing ready events
    for client in harness.clients:
        client.consume_kernel_ready = consume_noop  # type: ignore[assignment]
        client.consume_kernel_errors = consume_noop  # type: ignore[assignment]

    await asyncio.sleep(0.1)
    await coordinator.close()
    await oona.close()
    worker.shutdown(cancel_pending=True)

    assert not coordinator.poll_task_issue()
    assert harness.clients and all(client.closed for client in harness.clients)
