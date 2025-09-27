"""Oona-based kernel prefetch coordinator for Kasmina."""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from esper.core import AsyncWorker, AsyncWorkerHandle, DependencyViolationError
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, OonaMessage

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .seed_manager import KasminaSeedManager


class KasminaPrefetchCoordinator:
    """Bridges KasminaSeedManager with Oona kernel prefetch streams."""

    def __init__(
        self,
        manager: "KasminaSeedManager",
        oona: OonaClient,
        *,
        async_worker: AsyncWorker | None = None,
    ) -> None:
        self._manager = manager
        self._oona = oona
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._worker = async_worker
        self._worker_handles: list[AsyncWorkerHandle[Any]] = []
        self._publisher_handles: list[AsyncWorkerHandle[Any]] = []
        self._managed_worker_clients: list[OonaClient] = []
        self._lock = threading.Lock()

    def request_kernel(
        self,
        seed_id: str,
        blueprint_id: str,
        *,
        training_run_id: str | None = None,
    ) -> str:
        if not (training_run_id or "").strip():
            raise DependencyViolationError(
                "kasmina",
                "prefetch request missing training_run_id",
                context={
                    "dependency_type": "training_run_id",
                    "seed_id": seed_id,
                    "blueprint_id": blueprint_id,
                },
            )
        request = leyline_pb2.KernelPrefetchRequest(
            request_id=f"prefetch-{uuid.uuid4()}",
            blueprint_id=blueprint_id,
            training_run_id=training_run_id,
        )
        request.issued_at.GetCurrentTime()
        self._schedule_prefetch_request(request)
        return request.request_id

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._worker is not None:
            self._worker_handles = [
                self._worker.submit(
                    self._consume_ready_worker,
                    name="kasmina-prefetch-ready",
                ),
                self._worker.submit(
                    self._consume_error_worker,
                    name="kasmina-prefetch-error",
                ),
            ]
            self._tasks = []
            return

        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._consume_ready()),
            loop.create_task(self._consume_error()),
        ]

    async def close(self) -> None:
        self._running = False
        if self._worker_handles:
            handles = list(self._worker_handles)
            self._worker_handles.clear()
            for handle in handles:
                handle.cancel()
            for handle in handles:
                try:
                    handle.result(timeout=0)
                except BaseException:
                    continue
        with self._lock:
            publisher_handles = list(self._publisher_handles)
            self._publisher_handles.clear()
        for handle in publisher_handles:
            handle.cancel()
            try:
                handle.result(timeout=0)
            except BaseException:
                continue
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        with self._lock:
            clients = list(self._managed_worker_clients)
            self._managed_worker_clients.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:
                continue

    def _schedule_prefetch_request(
        self, request: leyline_pb2.KernelPrefetchRequest
    ) -> None:
        if self._worker is not None:
            payload = request.SerializeToString()
            handle = self._worker.submit(
                self._publish_request_worker,
                payload,
                name="kasmina-prefetch-publish",
            )
            with self._lock:
                self._publisher_handles.append(handle)
            handle.add_done_callback(lambda _f, h=handle: self._remove_publisher_handle(h))
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._oona.publish_kernel_prefetch_request(request))
        else:
            loop.create_task(self._oona.publish_kernel_prefetch_request(request))

    async def _consume_ready(self, client: OonaClient | None = None) -> None:
        oona = client or self._oona
        async def handler(message: OonaMessage) -> None:
            ready = leyline_pb2.KernelArtifactReady()
            ready.ParseFromString(message.payload)
            self._manager.process_prefetch_ready(ready)

        while self._running:
            await oona.consume_kernel_ready(handler, block_ms=500)

    async def _consume_error(self, client: OonaClient | None = None) -> None:
        oona = client or self._oona
        async def handler(message: OonaMessage) -> None:
            error = leyline_pb2.KernelArtifactError()
            error.ParseFromString(message.payload)
            self._manager.process_prefetch_error(error)

        while self._running:
            await oona.consume_kernel_errors(handler, block_ms=500)

    def poll_task_issue(self) -> BaseException | None:
        """Check background tasks and return an exception if any finished abnormally."""

        for task in self._tasks:
            if not task.done():
                continue
            if task.cancelled():
                if self._running:
                    return RuntimeError("Kasmina prefetch task cancelled unexpectedly")
                continue
            exception = task.exception()
            if exception is None:
                if self._running:
                    return RuntimeError("Kasmina prefetch task exited unexpectedly")
                continue
            return exception
        for handle in self._worker_handles:
            if not handle.done():
                continue
            try:
                result = handle.result(timeout=0)
            except asyncio.CancelledError:
                if self._running:
                    return RuntimeError("Kasmina prefetch worker cancelled unexpectedly")
                continue
            except Exception as exc:
                return exc
            else:
                if self._running:
                    return RuntimeError(
                        f"Kasmina prefetch worker '{handle.name}' exited unexpectedly"
                    )
                continue
        with self._lock:
            publisher_handles = list(self._publisher_handles)
        for handle in publisher_handles:
            if not handle.done():
                continue
            try:
                handle.result(timeout=0)
            except asyncio.CancelledError:
                if self._running:
                    return RuntimeError("Kasmina prefetch publish cancelled unexpectedly")
                continue
            except Exception as exc:
                return exc
        return None

    async def _consume_ready_worker(self) -> None:
        client, managed = self._spawn_worker_client("ready")
        try:
            if managed:
                await client.ensure_consumer_group()
            await self._consume_ready(client)
        finally:
            if managed:
                await client.close()
                self._unregister_worker_client(client)

    async def _consume_error_worker(self) -> None:
        client, managed = self._spawn_worker_client("error")
        try:
            if managed:
                await client.ensure_consumer_group()
            await self._consume_error(client)
        finally:
            if managed:
                await client.close()
                self._unregister_worker_client(client)

    async def _publish_request_worker(self, payload: bytes) -> None:
        client, managed = self._spawn_worker_client("publish")
        try:
            request = leyline_pb2.KernelPrefetchRequest()
            request.ParseFromString(payload)
            await client.publish_kernel_prefetch_request(request)
        finally:
            if managed:
                await client.close()
                self._unregister_worker_client(client)

    def _spawn_worker_client(self, role: str) -> tuple[OonaClient, bool]:
        spawner = getattr(self._oona, "spawn", None)
        if spawner is None:
            return self._oona, False
        suffix = f"{role}-{uuid.uuid4().hex[:6]}"
        client = spawner(consumer_suffix=suffix)
        # Worker-managed clients run on the shared AsyncWorker loop; disable
        # stale-claim scans that rely on redis futures bound to the creator loop.
        config = getattr(client, "_config", None)
        if config is not None and getattr(config, "retry_idle_ms", 0) > 0:
            try:
                client._config = replace(config, retry_idle_ms=0)
            except Exception:  # pragma: no cover - defensive, keep original config
                pass
        with self._lock:
            self._managed_worker_clients.append(client)
        return client, True

    def _unregister_worker_client(self, client: OonaClient) -> None:
        with self._lock:
            if client in self._managed_worker_clients:
                self._managed_worker_clients.remove(client)

    def _remove_publisher_handle(self, handle: AsyncWorkerHandle[Any]) -> None:
        with self._lock:
            if handle in self._publisher_handles:
                self._publisher_handles.remove(handle)


__all__ = ["KasminaPrefetchCoordinator"]
