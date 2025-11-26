from __future__ import annotations

import io
import inspect
import time

import pytest
import torch
from torch import nn
from torch.optim import SGD

from esper.core import AsyncWorker
from esper.tolaria.rollback import (
    DeadlineSignal,
    FastRollbackCache,
    Snapshot,
    attempt_two_tier_rollback,
)


def _build_model_optimizer() -> tuple[nn.Module, SGD]:
    model = nn.Linear(8, 4)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = SGD(model.parameters(), lr=0.01)
    return model, optimizer


def test_put_replaces_existing_snapshot_without_leaking_size() -> None:
    cache = FastRollbackCache(1)
    model, optimizer = _build_model_optimizer()

    cache.put(10, model, optimizer)
    size_first = cache.size_bytes

    cache.put(10, model, optimizer)

    assert cache.size_bytes == size_first
    assert list(cache._lru.keys()) == [10]  # type: ignore[attr-defined]


def test_put_evicts_old_entries_when_capacity_exceeded() -> None:
    cache = FastRollbackCache(1)
    # Restrict capacity so two snapshots cannot fit simultaneously.
    cache._max_bytes = 3000  # type: ignore[attr-defined]
    model, optimizer = _build_model_optimizer()

    cache.put(1, model, optimizer)
    cache.put(2, model, optimizer)

    assert cache.size_bytes <= cache._max_bytes  # type: ignore[attr-defined]
    assert list(cache._lru.keys()) == [2]  # type: ignore[attr-defined]


def test_restore_raises_when_optimizer_state_is_corrupt() -> None:
    cache = FastRollbackCache(1)
    model, optimizer = _build_model_optimizer()
    cache.put(1, model, optimizer)

    # Replace stored payload with one missing optimizer state to force failure.
    bad_buf = io.BytesIO()
    torch.save({"model": model.state_dict()}, bad_buf)
    bad_payload = bad_buf.getvalue()
    cache._lru[1] = Snapshot(step=1, size_bytes=len(bad_payload), payload=bad_payload)  # type: ignore[attr-defined]
    cache._size_bytes = len(bad_payload)  # type: ignore[attr-defined]

    snap = cache.get_nearest(1)
    assert snap is not None
    with pytest.raises(RuntimeError):
        cache.restore(snap, model, optimizer)


def test_restore_prefers_weights_only_when_supported(monkeypatch) -> None:
    cache = FastRollbackCache(1)
    model, optimizer = _build_model_optimizer()
    cache.put(5, model, optimizer)
    snap = cache.get_nearest(5)
    assert snap is not None

    original_load = torch.load
    captured: dict[str, object] = {}

    def fake_load(buffer, **kwargs):
        captured.update(kwargs)
        kwargs_without = {k: v for k, v in kwargs.items() if k != "weights_only"}
        return original_load(buffer, **kwargs_without)

    monkeypatch.setattr("esper.tolaria.rollback.torch.load", fake_load)

    cache.restore(snap, model, optimizer)
    if "weights_only" in inspect.signature(original_load).parameters:
        assert captured.get("weights_only") is True
    else:
        assert "weights_only" not in captured or not captured.get("weights_only")


def test_full_restore_timeout_uses_async_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = AsyncWorker(max_concurrency=1)
    model, optimizer = _build_model_optimizer()
    signal = DeadlineSignal()
    called = []

    def slow_restore() -> bool:
        called.append(time.time())
        time.sleep(0.05)
        return True

    try:
        result = attempt_two_tier_rollback(
            cache=None,
            deadline_ms=10,
            step=1,
            model=model,
            optimizer=optimizer,
            full_restore_cb=slow_restore,
            signal=signal,
            worker=worker,
        )
    finally:
        worker.close()

    assert called  # restore attempted
    assert not result.hit
    assert result.error == "deadline_exceeded"
    assert signal.is_set()


def test_full_restore_success_via_async_worker() -> None:
    worker = AsyncWorker(max_concurrency=1)
    model, optimizer = _build_model_optimizer()
    signal = DeadlineSignal()
    calls: list[int] = []

    def restore() -> bool:
        calls.append(1)
        return True

    try:
        result = attempt_two_tier_rollback(
            cache=None,
            deadline_ms=100,
            step=5,
            model=model,
            optimizer=optimizer,
            full_restore_cb=restore,
            signal=signal,
            worker=worker,
        )
    finally:
        worker.close()

    assert result.hit is True
    assert not result.used_fast
    assert result.error is None
    assert calls == [1]
    assert not signal.is_set()
