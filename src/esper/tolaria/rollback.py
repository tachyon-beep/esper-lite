"""Two-tier rollback support for Tolaria.

- Fast tier: in-memory LRU snapshots (bytes of state_dicts), size-capped.
- Full tier: delegate to trainer's existing checkpoint/WAL restore.

Scope note (prototype):
- Deadline enforcement currently uses a thread pool with timeouts to cancel
  long-running restores. For true cross-process cancellation, introduce a
  shared signaling primitive (e.g., `multiprocessing.Event` or POSIX shared
  memory + atomic flag) owned by Weatherlight and observed by Tolaria.
- Risk profile: The current approach cannot preempt kernel-mode I/O or CPU
  compute in third-party libraries; it only bounds our wait and proceeds with
  conservative mode. In worst case, a blocked restore thread lives until I/O
  returns, but the training loop moves on safely.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import io

import torch


@dataclass(slots=True)
class Snapshot:
    step: int
    size_bytes: int
    payload: bytes


class FastRollbackCache:
    def __init__(self, max_megabytes: int) -> None:
        self._max_bytes = int(max(1, max_megabytes)) * 1024 * 1024
        self._lru: OrderedDict[int, Snapshot] = OrderedDict()
        self._size_bytes = 0

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    def put(self, step: int, model, optimizer) -> None:  # type: ignore[no-untyped-def]
        # Serialize a minimal snapshot to bytes
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buf)
        payload = buf.getvalue()
        snap = Snapshot(step=step, size_bytes=len(payload), payload=payload)
        # Evict if necessary
        self._evict_until_fits(snap.size_bytes)
        self._lru[step] = snap
        self._lru.move_to_end(step)
        self._size_bytes += snap.size_bytes

    def _evict_until_fits(self, incoming: int) -> None:
        while self._size_bytes + incoming > self._max_bytes and self._lru:
            _, old = self._lru.popitem(last=False)
            self._size_bytes -= old.size_bytes

    def get_nearest(self, step_leq: int) -> Snapshot | None:
        candidates = [s for k, s in self._lru.items() if k <= step_leq]
        if not candidates:
            return None
        snap = max(candidates, key=lambda s: s.step)
        # Mark as recently used
        if snap.step in self._lru:
            self._lru.move_to_end(snap.step)
        return snap

    def restore(self, snap: Snapshot, model, optimizer) -> None:  # type: ignore[no-untyped-def]
        buf = io.BytesIO(snap.payload)
        payload = torch.load(buf, map_location=getattr(model, "device", None))
        model.load_state_dict(payload.get("model", {}))
        try:
            optimizer.load_state_dict(payload.get("optimizer", {}))
        except Exception:
            pass


# Placeholder for future shared signaling interface (prototype scope)
class DeadlineSignal:
    """Process-friendly cancellation primitive (scoped).

    Prototype uses a per-process event; future evolution can move to
    multiprocessing.Event or shared_memory for cross-process coordination.
    """

    def __init__(self) -> None:
        import threading

        self._ev = threading.Event()

    def trigger(self) -> None:
        self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set()

    def clear(self) -> None:
        self._ev.clear()


class SharedDeadlineSignal:
    """Cross-process deadline signal using shared memory (1-byte flag).

    Prototype: provides best-effort cross-process coordination without external
    brokers. Uses Python's `multiprocessing.shared_memory` when available.
    """

    def __init__(self, name: str, create: bool = True) -> None:
        try:
            from multiprocessing import shared_memory  # type: ignore
        except Exception as exc:  # pragma: no cover - platform dependent
            raise RuntimeError("shared_memory unavailable") from exc
        self._name = name
        self._created = create
        if create:
            try:
                self._shm = shared_memory.SharedMemory(name=name, create=True, size=1)
            except FileExistsError:
                self._shm = shared_memory.SharedMemory(name=name, create=False)
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
        # Zero-initialize if created
        if create:
            self.clear()

    @classmethod
    def create(cls, name: str) -> "SharedDeadlineSignal":
        return cls(name, create=True)

    @classmethod
    def attach(cls, name: str) -> "SharedDeadlineSignal":
        return cls(name, create=False)

    def trigger(self) -> None:
        self._shm.buf[0] = 1  # type: ignore[attr-defined]

    def clear(self) -> None:
        self._shm.buf[0] = 0  # type: ignore[attr-defined]

    def is_set(self) -> bool:
        return int(self._shm.buf[0]) == 1  # type: ignore[attr-defined]

    def close(self) -> None:
        try:
            self._shm.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def unlink(self) -> None:
        if self._created:
            try:
                self._shm.unlink()  # type: ignore[attr-defined]
            except Exception:
                pass


@dataclass(slots=True)
class RollbackResult:
    used_fast: bool
    latency_ms: float
    hit: bool


def attempt_two_tier_rollback(
    *,
    cache: FastRollbackCache | None,
    deadline_ms: int,
    step: int,
    model,  # type: ignore[no-untyped-def]
    optimizer,  # type: ignore[no-untyped-def]
    full_restore_cb,  # type: ignore[no-untyped-def]
    signal: DeadlineSignal | None = None,
) -> RollbackResult:
    start = perf_counter()
    if cache is not None:
        snap = cache.get_nearest(step)
        if snap is not None:
            cache.restore(snap, model, optimizer)
            return RollbackResult(True, (perf_counter() - start) * 1000.0, True)

    # Fall back to full restore with deadline enforcement
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(full_restore_cb)
            hit = bool(fut.result(timeout=max(0, int(deadline_ms)) / 1000.0))
    except FuturesTimeout:
        try:
            fut.cancel()
        except Exception:
            pass
        if signal is not None:
            try:
                signal.trigger()
            except Exception:
                pass
        return RollbackResult(False, (perf_counter() - start) * 1000.0, False)
    return RollbackResult(False, (perf_counter() - start) * 1000.0, hit)


__all__ = [
    "FastRollbackCache",
    "Snapshot",
    "RollbackResult",
    "attempt_two_tier_rollback",
]
