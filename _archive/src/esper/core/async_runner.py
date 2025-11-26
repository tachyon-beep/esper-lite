"""Async worker utilities for shared execution infrastructure.

This module provides a cancellable async worker that mirrors the shared
foundations plan documented in
``docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/05_shared_foundations.md``.
The worker exposes a small API for submitting coroutine or synchronous tasks
with timeout support so Tolaria, Tamiyo, and Kasmina can share a single
execution primitive during RC1 remediation work.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Generic, TypeVar

__all__ = ["AsyncTimeoutError", "AsyncWorker", "AsyncWorkerHandle", "AsyncWorkerStats"]


T = TypeVar("T")


class AsyncTimeoutError(TimeoutError):
    """Raised when a submitted task exceeds the configured timeout."""


@dataclass(slots=True)
class AsyncWorkerStats:
    """Simple counter set for harness diagnostics."""

    submitted: int = 0
    completed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    failed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, field_name: str) -> None:
        with self.lock:
            value = getattr(self, field_name)
            setattr(self, field_name, value + 1)


class AsyncWorkerHandle(Generic[T]):
    """Thread-safe handle wrapping the result of a worker submission."""

    def __init__(self, name: str, loop: asyncio.AbstractEventLoop, stats: AsyncWorkerStats) -> None:
        self._name = name
        self._loop = loop
        self._stats = stats
        self._future: Future[T] = Future()
        self._task: asyncio.Task[Any] | None = None
        self._finished = threading.Event()

    @property
    def name(self) -> str:
        return self._name

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        if self._future.done():
            return False

        def _cancel_task() -> None:
            if self._task and not self._task.done():
                self._task.cancel()
            if not self._future.done():
                self._future.cancel()
                if not self._task or self._task.done():
                    # Task was never scheduled or already finished.
                    self._stats.bump("cancelled")
                self._finished.set()

        self._loop.call_soon_threadsafe(_cancel_task)
        return True

    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
        self._future.add_done_callback(callback)

    # Internal helpers -------------------------------------------------
    def _attach_task(self, task: asyncio.Task[Any]) -> None:
        self._task = task

    def _set_result(self, value: T) -> None:
        if not self._future.done():
            self._future.set_result(value)
            self._stats.bump("completed")
        self._finished.set()

    def _set_exception(self, exc: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(exc)
            if isinstance(exc, AsyncTimeoutError):
                self._stats.bump("timed_out")
            elif isinstance(exc, asyncio.CancelledError):
                self._stats.bump("cancelled")
            else:
                self._stats.bump("failed")
        self._finished.set()

    def _set_cancelled(self) -> None:
        if not self._future.done():
            self._future.cancel()
            self._stats.bump("cancelled")
        self._finished.set()


class AsyncWorker:
    """Shared async worker coordinating coroutine execution across subsystems."""

    def __init__(
        self,
        *,
        max_concurrency: int = 4,
        name: str = "esper-async-worker",
        graceful_shutdown_timeout: float = 5.0,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        self._max_concurrency = max_concurrency
        self._name = name
        self._graceful_shutdown_timeout = graceful_shutdown_timeout

        self._loop = asyncio.new_event_loop()
        self._semaphore_ready = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._closing = False

        self._stats = AsyncWorkerStats()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._task_counter = 0

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"{name}-loop",
            daemon=True,
        )
        self._thread.start()
        # Wait until the event loop thread has created the semaphore.
        if not self._semaphore_ready.wait(timeout=graceful_shutdown_timeout):
            raise RuntimeError("AsyncWorker event loop failed to start in time")

    # ------------------------------------------------------------------
    @property
    def stats(self) -> AsyncWorkerStats:
        return self._stats

    def submit(
        self,
        func: Callable[..., Any] | Awaitable[Any],
        *args: Any,
        timeout: float | None = None,
        cancel_on_timeout: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> AsyncWorkerHandle[Any]:
        if self._closing:
            raise RuntimeError("AsyncWorker is shutting down")

        self._stats.bump("submitted")
        task_name = name or f"{self._name}-task-{self._task_counter}"
        self._task_counter += 1
        handle: AsyncWorkerHandle[Any] = AsyncWorkerHandle(task_name, self._loop, self._stats)

        submission = _Submission(
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            cancel_on_timeout=cancel_on_timeout,
            handle=handle,
        )

        def schedule() -> None:
            if self._closing:
                handle._set_exception(RuntimeError("AsyncWorker is shutting down"))
                return

            runner = self._loop.create_task(self._run_submission(submission), name=task_name)
            handle._attach_task(runner)
            self._tasks.add(runner)
            runner.add_done_callback(lambda _: self._tasks.discard(runner))

        self._loop.call_soon_threadsafe(schedule)
        return handle

    def shutdown(self, *, cancel_pending: bool = True, timeout: float | None = None) -> None:
        with self._shutdown_lock:
            if self._closing:
                return
            self._closing = True

        async def _graceful_shutdown() -> None:
            if cancel_pending:
                for task in list(self._tasks):
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._loop.stop()

        join_timeout = timeout if timeout is not None else self._graceful_shutdown_timeout
        future = asyncio.run_coroutine_threadsafe(_graceful_shutdown(), self._loop)
        try:
            future.result(timeout=join_timeout)
        except FuturesTimeout:
            if not self._loop.is_closed():
                try:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                except RuntimeError:
                    pass

        self._thread.join(join_timeout)
        if self._thread.is_alive():
            # Force stop and attempt one final join to avoid hanging callers.
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(join_timeout)

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> "AsyncWorker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.shutdown()

    # Internal helpers -------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._semaphore_ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    async def _run_submission(self, submission: "_Submission") -> None:
        async with self._semaphore:
            try:
                result = await submission.resolve()
            except asyncio.CancelledError as exc:  # worker shutdown cancels task
                submission.handle._set_exception(exc)
                raise
            except AsyncTimeoutError as exc:
                submission.handle._set_exception(exc)
            except Exception as exc:  # noqa: BLE001
                submission.handle._set_exception(exc)
            else:
                submission.handle._set_result(result)


def _is_coroutine_callable(func: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(func):
        return True
    call = getattr(func, "__call__", None)
    if call and inspect.iscoroutinefunction(call):
        return True
    if isinstance(func, functools.partial):
        return _is_coroutine_callable(func.func)
    return False


def _ensure_awaitable(result: Any) -> Awaitable[Any]:
    if inspect.isawaitable(result):
        return result  # type: ignore[return-value]
    return asyncio.sleep(0, result)  # creates completed awaitable


@dataclass(slots=True)
class _Submission:
    func: Callable[..., Any] | Awaitable[Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timeout: float | None
    cancel_on_timeout: bool
    handle: AsyncWorkerHandle[Any]

    async def resolve(self) -> Any:
        start = monotonic()
        awaitable = self._build_awaitable()
        try:
            if self.timeout is not None:
                return await asyncio.wait_for(awaitable, timeout=self.timeout)
            return await awaitable
        except asyncio.TimeoutError as exc:
            if self.cancel_on_timeout:
                # Provide deterministic cancellation when wait_for aborts.
                if hasattr(awaitable, "cancel"):
                    awaitable.cancel()
            raise AsyncTimeoutError(f"Task exceeded timeout after {monotonic() - start:.3f}s") from exc

    def _build_awaitable(self) -> Awaitable[Any]:
        target = self.func
        if inspect.isawaitable(target):
            return _ensure_awaitable(target)

        if callable(target) and _is_coroutine_callable(target):
            coroutine = target(*self.args, **self.kwargs)
            if not inspect.isawaitable(coroutine):
                return _ensure_awaitable(coroutine)
            return coroutine  # type: ignore[return-value]

        if callable(target):
            return asyncio.to_thread(target, *self.args, **self.kwargs)

        raise TypeError("func must be a callable or awaitable")
