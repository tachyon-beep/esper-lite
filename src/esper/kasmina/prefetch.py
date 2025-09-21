"""Oona-based kernel prefetch coordinator for Kasmina."""

from __future__ import annotations

import asyncio
import uuid

from typing import TYPE_CHECKING

from esper.leyline import leyline_pb2
from esper.oona import OonaClient, OonaMessage

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .seed_manager import KasminaSeedManager


class KasminaPrefetchCoordinator:
    """Bridges KasminaSeedManager with Oona kernel prefetch streams."""

    def __init__(self, manager: "KasminaSeedManager", oona: OonaClient) -> None:
        self._manager = manager
        self._oona = oona
        self._tasks: list[asyncio.Task] = []
        self._running = False

    def request_kernel(
        self,
        seed_id: str,
        blueprint_id: str,
        *,
        training_run_id: str | None = None,
    ) -> str:
        request = leyline_pb2.KernelPrefetchRequest(
            request_id=f"prefetch-{uuid.uuid4()}",
            blueprint_id=blueprint_id,
            training_run_id=training_run_id or "prototype",
        )
        request.issued_at.GetCurrentTime()
        self._schedule(self._oona.publish_kernel_prefetch_request(request))
        return request.request_id

    def start(self) -> None:
        if self._running:
            return
        loop = asyncio.get_running_loop()
        self._running = True
        self._tasks = [
            loop.create_task(self._consume_ready()),
            loop.create_task(self._consume_error()),
        ]

    async def close(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _schedule(self, coro) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    async def _consume_ready(self) -> None:
        async def handler(message: OonaMessage) -> None:
            ready = leyline_pb2.KernelArtifactReady()
            ready.ParseFromString(message.payload)
            self._manager.process_prefetch_ready(ready)

        while self._running:
            await self._oona.consume_kernel_ready(handler, block_ms=500)

    async def _consume_error(self) -> None:
        async def handler(message: OonaMessage) -> None:
            error = leyline_pb2.KernelArtifactError()
            error.ParseFromString(message.payload)
            self._manager.process_prefetch_error(error)

        while self._running:
            await self._oona.consume_kernel_errors(handler, block_ms=500)

    def poll_task_issue(self) -> BaseException | None:
        """Check background tasks and return an exception if any finished abnormally."""

        for task in self._tasks:
            if not task.done():
                continue
            if task.cancelled():
                return RuntimeError("Kasmina prefetch task cancelled unexpectedly")
            exception = task.exception()
            if exception is None:
                return RuntimeError("Kasmina prefetch task exited unexpectedly")
            return exception
        return None


__all__ = ["KasminaPrefetchCoordinator"]
