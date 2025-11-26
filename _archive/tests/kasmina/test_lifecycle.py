from __future__ import annotations

import json

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch import nn

from esper.karn import BlueprintDescriptor, BlueprintTier, KarnCatalog
from esper.core import DependencyViolationError
import asyncio

from esper.kasmina import KasminaLifecycle, KasminaPrefetchCoordinator, KasminaSeedManager, SeedContext
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign
from esper.tezzeret import CompileJobConfig, TezzeretCompiler
from esper.urza import UrzaLibrary, UrzaRuntime
from esper.urza.pipeline import BlueprintPipeline, BlueprintRequest


class _RuntimeStub:
    def __init__(self) -> None:
        self.loaded = []

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        self.loaded.append(blueprint_id)
        return nn.Identity(), 1.0


def test_lifecycle_default_state_is_dormant() -> None:
    lifecycle = KasminaLifecycle()
    assert lifecycle.state == leyline_pb2.SEED_STAGE_DORMANT


def test_lifecycle_transitions_follow_order() -> None:
    lifecycle = KasminaLifecycle()
    ordered = [
        leyline_pb2.SEED_STAGE_GERMINATED,
        leyline_pb2.SEED_STAGE_TRAINING,
        leyline_pb2.SEED_STAGE_BLENDING,
        leyline_pb2.SEED_STAGE_SHADOWING,
        leyline_pb2.SEED_STAGE_PROBATIONARY,
        leyline_pb2.SEED_STAGE_FOSSILIZED,
        leyline_pb2.SEED_STAGE_TERMINATED,
    ]
    for stage in ordered:
        lifecycle.transition(stage)
    assert lifecycle.state == leyline_pb2.SEED_STAGE_TERMINATED


def test_lifecycle_rejects_invalid_transition() -> None:
    lifecycle = KasminaLifecycle()
    with pytest.raises(ValueError):
        lifecycle.transition(leyline_pb2.SEED_STAGE_SHADOWING)


def test_seed_manager_grafts_and_retires_seed() -> None:
    runtime = _RuntimeStub()
    manager = KasminaSeedManager(runtime=runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-1",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "bp-1"
    _sign_command(command)
    manager.handle_command(command)
    seeds = manager.seeds()
    assert "seed-1" in seeds
    assert seeds["seed-1"].lifecycle.state == leyline_pb2.SEED_STAGE_PROBATIONARY
    assert pytest.approx(seeds["seed-1"].alpha, rel=1e-5) == 1.0
    assert runtime.loaded == ["bp-1"]

    retire = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-2",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    retire.seed_operation.operation = leyline_pb2.SEED_OP_CULL
    retire.seed_operation.blueprint_id = "bp-1"
    _sign_command(retire)
    manager.handle_command(retire)
    assert "seed-1" not in manager.seeds()
    payload = manager.rollback_payload("seed-1")
    assert payload is not None
    assert payload["reason"] == "retired"


@pytest.mark.asyncio
async def test_seed_manager_with_urza_runtime() -> None:
    catalog = KarnCatalog(load_defaults=False)
    metadata = BlueprintDescriptor(
        blueprint_id="bp-1",
        name="Test",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )
    bounds = metadata.allowed_parameters["alpha"]
    bounds.min_value = 0.0
    bounds.max_value = 1.0
    catalog.register(metadata)
    with TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "artifacts"
        urza_root = Path(tmp) / "urza"
        library = UrzaLibrary(root=urza_root)
        compiler = TezzeretCompiler(config=CompileJobConfig(artifact_dir=artifact_dir))
        pipeline = BlueprintPipeline(catalog=catalog, compiler=compiler, library=library)
        await pipeline.handle_request(
            BlueprintRequest(
                blueprint_id="bp-1",
                parameters={"alpha": 0.5},
                training_run_id="run-1",
            )
        )
        urza_runtime = UrzaRuntime(library)
        manager = KasminaSeedManager(runtime=urza_runtime, signing_context=_SIGNING_CONTEXT)
        manager.register_host_model(nn.Linear(1, 1))
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-urza",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = "bp-1"
        _sign_command(command)
        manager.handle_command(command)
        assert "seed-urza" in manager.seeds()


def test_full_lifecycle_path() -> None:
    lc = KasminaLifecycle()
    stages = [
        leyline_pb2.SEED_STAGE_GERMINATED,
        leyline_pb2.SEED_STAGE_TRAINING,
        leyline_pb2.SEED_STAGE_BLENDING,
        leyline_pb2.SEED_STAGE_SHADOWING,
        leyline_pb2.SEED_STAGE_PROBATIONARY,
        leyline_pb2.SEED_STAGE_FOSSILIZED,
        leyline_pb2.SEED_STAGE_TERMINATED,
    ]
    for stage in stages:
        lc.transition(stage)
    assert lc.state == leyline_pb2.SEED_STAGE_TERMINATED


def test_cull_path_returns_to_dormant() -> None:
    lc = KasminaLifecycle()
    lc.transition(leyline_pb2.SEED_STAGE_GERMINATED)
    lc.transition(leyline_pb2.SEED_STAGE_TRAINING)
    lc.transition(leyline_pb2.SEED_STAGE_CULLED)
    lc.transition(leyline_pb2.SEED_STAGE_EMBARGOED)
    lc.transition(leyline_pb2.SEED_STAGE_RESETTING)
    lc.transition(leyline_pb2.SEED_STAGE_DORMANT)
    assert lc.state == leyline_pb2.SEED_STAGE_DORMANT


def test_register_prefetch_requires_training_run_id() -> None:
    runtime = _RuntimeStub()
    manager = KasminaSeedManager(runtime=runtime, signing_context=_SIGNING_CONTEXT)
    manager.register_host_model(nn.Linear(1, 1))
    manager._seeds["seed-guard"] = SeedContext("seed-guard")
    manager.set_prefetch(_PrefetchStub())

    with pytest.raises(DependencyViolationError):
        manager._register_prefetch(manager._seeds["seed-guard"], "bp-guard")

    manager._seeds["seed-guard"].metadata["training_run_id"] = "run-123"
    request_id = manager._register_prefetch(manager._seeds["seed-guard"], "bp-guard")
    assert request_id == "prefetch-req"


def test_prefetch_coordinator_requires_training_run_id() -> None:
    manager = _ManagerStub()
    coordinator = KasminaPrefetchCoordinator(manager, _OonaStub())

    with pytest.raises(DependencyViolationError):
        coordinator.request_kernel("seed-guard", "bp-guard")

    coordinator.request_kernel(
        "seed-guard",
        "bp-guard",
        training_run_id="run-123",
    )
    assert manager.events == []  # scheduler shouldn't run in synchronous path


def test_prefetch_coordinator_worker_spawns_clients(monkeypatch) -> None:
    manager = _ManagerStub()
    oona = _SpawnableOonaStub()
    worker = _WorkerStub()
    coordinator = KasminaPrefetchCoordinator(manager, oona, async_worker=worker)

    coordinator.start()
    assert len(worker.calls) == 2

    async def stub_ready(client):
        assert isinstance(client, _SpawnedClientStub)
        coordinator._running = False

    async def stub_error(client):
        assert isinstance(client, _SpawnedClientStub)
        coordinator._running = False

    coordinator._consume_ready = stub_ready  # type: ignore[assignment]
    coordinator._consume_error = stub_error  # type: ignore[assignment]
    coordinator._running = False

    for func, args, kwargs, _handle in worker.calls:
        asyncio.run(func(*args, **kwargs))

    assert any("ready" in call for call in oona.spawn_calls)
    assert any("error" in call for call in oona.spawn_calls)
    asyncio.run(coordinator.close())


def test_prefetch_coordinator_worker_publish_uses_spawn(monkeypatch) -> None:
    manager = _ManagerStub()
    oona = _SpawnableOonaStub()
    worker = _WorkerStub()
    coordinator = KasminaPrefetchCoordinator(manager, oona, async_worker=worker)
    coordinator._running = True

    coordinator.request_kernel(
        "seed-1",
        "bp-1",
        training_run_id="run-1",
    )

    publish_calls = [call for call in worker.calls if call[0] == coordinator._publish_request_worker]
    assert publish_calls, "publish worker not scheduled"

    func, args, kwargs, _handle = publish_calls[0]
    asyncio.run(func(*args, **kwargs))

    assert any("publish" in call for call in oona.spawn_calls)
    asyncio.run(coordinator.close())


_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-test-secret")


def _sign_command(command: leyline_pb2.AdaptationCommand) -> None:
    command.annotations.setdefault("training_run_id", "test-run")
    if "signature" in command.annotations:
        del command.annotations["signature"]
    if (
        command.command_type == leyline_pb2.COMMAND_SEED
        and "mesh_host_layers" not in command.annotations
    ):
        # Default mesh layers align with a 1x1 Linear host model used in tests.
        command.annotations["mesh_host_layers"] = json.dumps(["weight", "bias"])
    command.issued_at.GetCurrentTime()
    command.annotations["signature"] = sign(
        command.SerializeToString(deterministic=True),
        _SIGNING_CONTEXT,
    )


class _PrefetchStub:
    def __init__(self) -> None:
        self.last = None

    def request_kernel(self, seed_id: str, blueprint_id: str, *, training_run_id: str) -> str:
        self.last = (seed_id, blueprint_id, training_run_id)
        return "prefetch-req"


class _ManagerStub:
    def __init__(self) -> None:
        self.events: list[str] = []

    def process_prefetch_ready(self, ready):  # pragma: no cover - unused stub
        self.events.append("ready")

    def process_prefetch_error(self, error):  # pragma: no cover - unused stub
        self.events.append("error")


class _OonaStub:
    async def publish_kernel_prefetch_request(self, _request):
        return None


class _WorkerHandleStub:
    def __init__(self, name: str = "stub") -> None:
        self._name = name
        self._cancelled = False
        self._callbacks: list[callable] = []

    @property
    def name(self) -> str:
        return self._name

    def cancel(self) -> bool:
        if self._cancelled:
            return False
        self._cancelled = True
        for callback in list(self._callbacks):
            callback(self)
        return True

    def result(self, timeout: float | None = None):
        if self._cancelled:
            raise asyncio.CancelledError
        return None

    def done(self) -> bool:
        return self._cancelled

    def add_done_callback(self, callback):
        self._callbacks.append(callback)

    def exception(self, timeout: float | None = None):  # pragma: no cover - stub
        return None


class _WorkerStub:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def submit(self, func, *args, **kwargs):
        name = kwargs.pop("name", "worker")
        handle = _WorkerHandleStub(name)
        self.calls.append((func, args, kwargs, handle))
        return handle


class _SpawnedClientStub:
    def __init__(self, role: str, tracker: list[str]) -> None:
        self.role = role
        self.tracker = tracker
        self.closed = False
        self.published: list[leyline_pb2.KernelPrefetchRequest] = []

    async def ensure_consumer_group(self) -> None:  # pragma: no cover - stub
        return None

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        self.tracker.append(f"publish:{self.role}")
        self.published.append(request)

    async def consume_kernel_ready(self, handler, *, block_ms: int = 0) -> None:
        self.tracker.append(f"ready:{self.role}")

    async def consume_kernel_errors(self, handler, *, block_ms: int = 0) -> None:
        self.tracker.append(f"error:{self.role}")

    async def close(self) -> None:
        self.closed = True


class _SpawnableOonaStub(_OonaStub):
    def __init__(self) -> None:
        self.spawn_calls: list[str] = []
        self.children: list[_SpawnedClientStub] = []

    def spawn(self, *, consumer_suffix: str | None = None) -> _SpawnedClientStub:
        role = consumer_suffix or "spawn"
        self.spawn_calls.append(role)
        child = _SpawnedClientStub(role, self.spawn_calls)
        self.children.append(child)
        return child
