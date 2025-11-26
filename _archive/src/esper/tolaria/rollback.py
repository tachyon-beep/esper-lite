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

import contextlib
import io
import inspect
import logging
import struct
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic_ns, perf_counter

import torch

from esper.core import AsyncTimeoutError, AsyncWorker


LOGGER = logging.getLogger(__name__)
_WEIGHTS_ONLY_SUPPORTED = "weights_only" in inspect.signature(torch.load).parameters


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
        if step in self._lru:
            existing = self._lru.pop(step)
            self._size_bytes = max(0, self._size_bytes - existing.size_bytes)
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
        device = infer_model_device(model)
        try:
            payload = load_state_dict_from_bytes(snap.payload, device=device)
        except Exception as exc:
            LOGGER.error(
                "Failed to load rollback snapshot for step %s: %s",
                snap.step,
                exc,
                exc_info=True,
            )
            raise
        model.load_state_dict(payload.get("model", {}))
        try:
            optimizer.load_state_dict(payload.get("optimizer", {}))
        except Exception as exc:  # pragma: no cover - exercised via dedicated tests
            LOGGER.error(
                "Failed to restore optimizer state from rollback snapshot %s", snap.step, exc_info=True
            )
            raise RuntimeError(
                "failed to restore optimizer state from rollback snapshot"
            ) from exc


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
        size = 16  # 1 byte flag + 8 byte timestamp (ms) + padding
        if create:
            try:
                self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)
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
        # Set flag
        self._shm.buf[0] = 1  # type: ignore[attr-defined]
        # Write current time in ms into bytes [8:16)
        try:
            ts_ms = int(monotonic_ns() // 1_000_000)
            self._shm.buf[8:16] = struct.pack("<Q", ts_ms)  # type: ignore[attr-defined]
        except Exception:
            pass

    def clear(self) -> None:
        self._shm.buf[0] = 0  # type: ignore[attr-defined]
        try:
            self._shm.buf[8:16] = b"\x00" * 8  # type: ignore[attr-defined]
        except Exception:
            pass

    def is_set(self) -> bool:
        return int(self._shm.buf[0]) == 1  # type: ignore[attr-defined]

    def read_timestamp_ms(self) -> int | None:
        try:
            ts_bytes = bytes(self._shm.buf[8:16])  # type: ignore[attr-defined]
            ts_ms = struct.unpack("<Q", ts_bytes)[0]
            return int(ts_ms) if ts_ms > 0 else None
        except Exception:
            return None

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
    error: str | None = None


def attempt_two_tier_rollback(
    *,
    cache: FastRollbackCache | None,
    deadline_ms: int,
    step: int,
    model,  # type: ignore[no-untyped-def]
    optimizer,  # type: ignore[no-untyped-def]
    full_restore_cb,  # type: ignore[no-untyped-def]
    signal: DeadlineSignal | None = None,
    worker: AsyncWorker | None = None,
    task_name: str = "tolaria-rollback",
) -> RollbackResult:
    start = perf_counter()
    if cache is not None:
        snap = cache.get_nearest(step)
        if snap is not None:
            try:
                cache.restore(snap, model, optimizer)
                return RollbackResult(True, (perf_counter() - start) * 1000.0, True)
            except Exception as exc:
                LOGGER.error(
                    "Fast rollback restore failed for step %s: %s", step, exc, exc_info=True
                )
                return RollbackResult(True, (perf_counter() - start) * 1000.0, False, error=str(exc))

    deadline_s = max(0, int(deadline_ms)) / 1000.0 if deadline_ms > 0 else None
    if worker is not None:
        handle = worker.submit(
            full_restore_cb,
            timeout=deadline_s,
            cancel_on_timeout=True,
            name=task_name,
        )
        try:
            hit = bool(handle.result())
            return RollbackResult(False, (perf_counter() - start) * 1000.0, hit)
        except AsyncTimeoutError:
            if signal is not None:
                with contextlib.suppress(Exception):
                    signal.trigger()
            return RollbackResult(
                False,
                (perf_counter() - start) * 1000.0,
                False,
                error="deadline_exceeded",
            )
        except Exception as exc:  # pragma: no cover - propagation tested separately
            LOGGER.error("Rollback restore encountered error: %s", exc, exc_info=True)
            return RollbackResult(False, (perf_counter() - start) * 1000.0, False, error=str(exc))

    # Fallback path when no worker is available (runs synchronously without deadline enforcement).
    try:
        hit = bool(full_restore_cb())
    except Exception as exc:  # pragma: no cover - propagation tested separately
        LOGGER.error("Rollback restore encountered error: %s", exc, exc_info=True)
        return RollbackResult(False, (perf_counter() - start) * 1000.0, False, error=str(exc))
    return RollbackResult(False, (perf_counter() - start) * 1000.0, hit)


def infer_model_device(model) -> torch.device:
    try:
        param = next(model.parameters())
    except (StopIteration, AttributeError, TypeError):
        param = None
    device = None
    if param is not None:
        device = param.device
    else:
        try:
            buffer = next(model.buffers())
            device = buffer.device
        except (StopIteration, AttributeError, TypeError):
            device = None
    if device is None:
        return torch.device("cpu")
    if not isinstance(device, torch.device):
        try:
            device = torch.device(str(device))
        except Exception:
            return torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _map_location(device: torch.device | None) -> torch.device | str | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    try:
        dev = torch.device(device)
    except Exception:
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def load_state_dict_from_bytes(
    payload: bytes, *, device: torch.device | str | None = None
) -> dict[str, object]:
    buffer = io.BytesIO(payload)
    kwargs: dict[str, object] = {}
    map_location = _map_location(device)
    if map_location is not None:
        kwargs["map_location"] = map_location
    if _WEIGHTS_ONLY_SUPPORTED:
        kwargs["weights_only"] = True
    try:
        return torch.load(buffer, **kwargs)
    except TypeError:
        if kwargs.pop("weights_only", None) is not None:
            buffer.seek(0)
            return torch.load(buffer, **kwargs)
        raise


__all__ = [
    "FastRollbackCache",
    "Snapshot",
    "RollbackResult",
    "attempt_two_tier_rollback",
    "infer_model_device",
    "load_state_dict_from_bytes",
]
