"""Shared helpers for the RC1 cross-system performance harness."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.core import AsyncWorker
from esper.kasmina.prefetch import KasminaPrefetchCoordinator
from esper.kasmina.seed_manager import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.oona import OonaMessage
from esper.security.signing import SignatureContext, sign
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.tolaria import trainer as trainer_module
from esper.tolaria.emergency import EmergencyController
from esper.tolaria.rollback import RollbackResult

# ---------------------------------------------------------------------------
# Data structures


@dataclass(slots=True)
class ScenarioResult:
    """Structured output saved by harness scenarios."""

    scenario: str
    environment: dict[str, Any]
    metrics: dict[str, float]
    telemetry: list[dict[str, Any]]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "environment": self.environment,
            "metrics": self.metrics,
            "telemetry": self.telemetry,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Common helpers


_SIGNATURE_CONTEXT = SignatureContext(secret=b"rc1-harness-secret")


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }


def _serialize_packet(packet: leyline_pb2.TelemetryPacket) -> dict[str, Any]:
    payload = {
        "metrics": {metric.name: metric.value for metric in packet.metrics},
        "events": [event.description for event in packet.events],
    }
    if hasattr(packet, "priority"):
        try:
            payload["priority"] = packet.priority
        except AttributeError:
            pass
    return payload


def _write_results(result: ScenarioResult, output_dir: Path, slug: str) -> Path:
    _ensure_output_dir(output_dir)
    json_path = output_dir / f"{slug}_metrics.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    # Write a tiny CSV summary for quick spreadsheet import.
    csv_path = output_dir / "summary.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["scenario", *sorted(result.metrics.keys())])
        row = [result.scenario] + [f"{result.metrics[key]:.6f}" for key in sorted(result.metrics.keys())]
        writer.writerow(row)
    return json_path


# ---------------------------------------------------------------------------
# Tolaria scenarios


class _SeedCommandTamiyo(TamiyoClient):
    """Generates deterministic SEED commands for the harness."""

    def __init__(self, *, blueprint_id: str, target_seed_id: str) -> None:
        self._blueprint_id = blueprint_id
        self._target_seed_id = target_seed_id
        self.evaluations: list[leyline_pb2.SystemStatePacket] = []

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self.evaluations.append(state)
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.epoch}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id=self._target_seed_id,
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = self._blueprint_id
        command.issued_at.GetCurrentTime()
        command.annotations["signature"] = sign(
            command.SerializeToString(deterministic=True), _SIGNATURE_CONTEXT
        )
        return command


class _RelayKasmina(KasminaClient):
    def __init__(self) -> None:
        self.received: list[leyline_pb2.AdaptationCommand] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.received.append(command)


class _TimeoutTamiyo(TamiyoClient):
    def __init__(self, *, timeout_every: int) -> None:
        self._timeout_every = max(timeout_every, 1)
        self._calls = 0

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        self._calls += 1
        if self._calls % self._timeout_every == 0:
            raise TimeoutError("tamiyo timeout drill")
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=f"cmd-{state.epoch}",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-timeout",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = "bp-timeout"
        command.issued_at.GetCurrentTime()
        return command


class _TimeoutKasmina(KasminaClient):
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        raise TimeoutError("kasmina timeout drill")


def _build_dataset(device: torch.device, *, batch_size: int, samples: int = 64) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(42)
    inputs = torch.randn(samples, 4, generator=generator)
    targets = torch.randint(0, 2, (samples,), generator=generator)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _build_trainer(
    *,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    tamiyo: TamiyoClient,
    kasmina: KasminaClient,
    enable_compile: bool,
    enable_graphs: bool,
) -> TolariaTrainer:
    model = nn.Linear(4, 2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataloader = _build_dataset(device, batch_size=batch_size)
    config = TrainingLoopConfig(
        max_epochs=max_epochs,
        device=device,
        gradient_accumulation_steps=1,
        tamiyo_timeout_s=0.0,
        enable_compile=enable_compile,
        enable_graphs=enable_graphs,
        enable_gpu_prefetch=False,
    )
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=config,
    )
    return trainer


def run_steady_train(
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    enable_compile: bool = False,
    enable_graphs: bool = False,
) -> ScenarioResult:
    tamiyo = _SeedCommandTamiyo(blueprint_id="bp-steady", target_seed_id="seed-steady")
    kasmina = _RelayKasmina()
    trainer = _build_trainer(
        device=device,
        max_epochs=epochs,
        batch_size=batch_size,
        tamiyo=tamiyo,
        kasmina=kasmina,
        enable_compile=enable_compile,
        enable_graphs=enable_graphs,
    )
    try:
        list(trainer.run())
        telemetry = [
            _serialize_packet(packet) for packet in trainer.telemetry_packets
        ]
        latencies = [pkt["metrics"].get("tolaria.training.latency_ms", 0.0) for pkt in telemetry]
        summary = {
            "epochs": float(epochs),
            "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
            "latency_p95_ms": _percentile(latencies, 0.95),
            "accuracy_final": telemetry[-1]["metrics"].get("tolaria.training.accuracy", 0.0)
            if telemetry
            else 0.0,
            "loss_final": telemetry[-1]["metrics"].get("tolaria.training.loss", 0.0)
            if telemetry
            else 0.0,
        }
        return ScenarioResult(
            scenario="steady_train",
            environment=_environment_snapshot(device),
            metrics=summary,
            telemetry=telemetry,
            details={
                "tamiyo_evaluations": len(tamiyo.evaluations),
                "kasmina_commands": len(kasmina.received),
            },
        )
    finally:
        trainer.close()


def run_rollback_deadline(
    *,
    device: torch.device,
    batch_size: int,
    enable_compile: bool = False,
    enable_graphs: bool = False,
    deadline_ms: float = 5.0,
) -> ScenarioResult:
    tamiyo = _SeedCommandTamiyo(blueprint_id="bp-rollback", target_seed_id="seed-rollback")
    kasmina = _TimeoutKasmina()
    trainer = _build_trainer(
        device=device,
        max_epochs=1,
        batch_size=batch_size,
        tamiyo=tamiyo,
        kasmina=kasmina,
        enable_compile=enable_compile,
        enable_graphs=enable_graphs,
    )
    trainer._settings = trainer._settings.model_copy(
        update={
            "tolaria_emergency_enabled": True,
            "tolaria_emergency_l4_on_rollback_deadline": True,
            "tolaria_rollback_deadline_ms": deadline_ms,
            "tolaria_rollback_enabled": True,
        }
    )
    trainer._emergency = EmergencyController(bypass_cap_per_min=1)
    trainer._metrics.setdefault("tolaria.emergency.broadcasts_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.bypass_applied_total", 0.0)
    trainer._metrics.setdefault("tolaria.emergency.halts_total", 0.0)

    class _FastCacheStub:
        def get_nearest(self, step: int):
            return None

    trainer._fast_cache = _FastCacheStub()

    fake_result = RollbackResult(False, 12.0, False, error="deadline_exceeded")

    original = trainer_module.attempt_two_tier_rollback
    try:
        trainer_module.attempt_two_tier_rollback = lambda **kwargs: fake_result  # type: ignore[assignment]
        list(trainer.run())
    finally:
        trainer_module.attempt_two_tier_rollback = original

    telemetry = [_serialize_packet(packet) for packet in trainer.telemetry_packets]
    metrics = telemetry[-1]["metrics"] if telemetry else {}
    fallback_deadline = trainer._metrics.get("tolaria.rollback.deadline_exceeded_total", 0.0)
    summary = {
        "timeout_kasmina": metrics.get("tolaria.timeout.kasmina_total", 0.0),
        "rollback_deadline_exceeded": metrics.get("tolaria.rollback.deadline_exceeded_total", fallback_deadline),
        "rollback_restore_latency_ms": metrics.get("tolaria.rollback.restore_latency_ms", 0.0) or trainer._metrics.get("tolaria.rollback.restore_latency_ms", 0.0),
    }
    result = ScenarioResult(
        scenario="rollback_deadline",
        environment=_environment_snapshot(device),
        metrics=summary,
        telemetry=telemetry,
        details={
            "tamiyo_evaluations": len(tamiyo.evaluations),
            "halted": trainer._halt,
        },
    )
    trainer.close()
    return result


def run_tamiyo_timeout(
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    timeout_every: int = 2,
    enable_compile: bool = False,
    enable_graphs: bool = False,
) -> ScenarioResult:
    tamiyo = _TimeoutTamiyo(timeout_every=timeout_every)
    kasmina = _RelayKasmina()
    trainer = _build_trainer(
        device=device,
        max_epochs=epochs,
        batch_size=batch_size,
        tamiyo=tamiyo,
        kasmina=kasmina,
        enable_compile=enable_compile,
        enable_graphs=enable_graphs,
    )
    try:
        list(trainer.run())
        telemetry = [_serialize_packet(packet) for packet in trainer.telemetry_packets]
        metrics = telemetry[-1]["metrics"] if telemetry else {}
        summary = {
            "timeout_tamiyo": metrics.get("tolaria.timeout.tamiyo_total", 0.0),
            "timeout_kasmina": metrics.get("tolaria.timeout.kasmina_total", 0.0),
            "evaluations": float(tamiyo._calls),
        }
        return ScenarioResult(
            scenario="tamiyo_timeout",
            environment=_environment_snapshot(device),
            metrics=summary,
            telemetry=telemetry,
            details={
                "timeout_every": timeout_every,
            },
        )
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Kasmina prefetch harness


class _PrefetchBackend:
    def __init__(self, *, ready_latency_ms: float, jitter_ms: float, error_rate: float) -> None:
        self._ready_latency = max(ready_latency_ms, 0.0) / 1000.0
        self._jitter = max(jitter_ms, 0.0) / 1000.0
        self._error_rate = min(max(error_rate, 0.0), 1.0)
        self._ready_queue: asyncio.Queue[leyline_pb2.KernelArtifactReady] = asyncio.Queue()
        self._error_queue: asyncio.Queue[leyline_pb2.KernelArtifactError] = asyncio.Queue()
        self._start_times: dict[str, float] = {}
        self._durations: list[float] = []

    def _now(self) -> float:
        return time.perf_counter()

    async def publish(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        request_id = request.request_id
        self._start_times[request_id] = self._now()

        async def _emit() -> None:
            await asyncio.sleep(self._ready_latency)
            started = self._start_times.pop(request_id, None)
            if started is not None:
                duration = (self._now() - started) * 1000.0
                self._durations.append(duration)
            ready = leyline_pb2.KernelArtifactReady(request_id=request_id)
            await self._ready_queue.put(ready)

        asyncio.get_running_loop().create_task(_emit())

    async def next_ready(self, timeout: float | None) -> leyline_pb2.KernelArtifactReady | None:
        try:
            if timeout is None:
                return await self._ready_queue.get()
            return await asyncio.wait_for(self._ready_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def next_error(self, timeout: float | None) -> leyline_pb2.KernelArtifactError | None:
        try:
            if timeout is None:
                return await self._error_queue.get()
            return await asyncio.wait_for(self._error_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def durations(self) -> list[float]:
        return list(self._durations)


class _WorkerClientStub:
    def __init__(self, backend: _PrefetchBackend, role: str) -> None:
        self._backend = backend
        self._role = role

    async def ensure_consumer_group(self) -> None:
        return None

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        await self._backend.publish(request)

    async def consume_kernel_ready(self, handler, *, block_ms: int = 0) -> None:
        timeout = block_ms / 1000.0 if block_ms else None
        ready = await self._backend.next_ready(timeout)
        if ready is None:
            return
        message = OonaMessage(
            stream="kasmina.ready",
            message_id="m-ready",
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY,
            payload=ready.SerializeToString(),
        )
        result = handler(message)
        if asyncio.iscoroutine(result):
            await result

    async def consume_kernel_errors(self, handler, *, block_ms: int = 0) -> None:
        timeout = block_ms / 1000.0 if block_ms else None
        error = await self._backend.next_error(timeout)
        if error is None:
            return
        message = OonaMessage(
            stream="kasmina.errors",
            message_id="m-error",
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR,
            payload=error.SerializeToString(),
        )
        result = handler(message)
        if asyncio.iscoroutine(result):
            await result

    async def close(self) -> None:
        return None


class _OonaStub:
    def __init__(self, backend: _PrefetchBackend) -> None:
        self._backend = backend
        self._workers: list[_WorkerClientStub] = []

    async def publish_kernel_prefetch_request(self, request: leyline_pb2.KernelPrefetchRequest) -> None:
        await self._backend.publish(request)

    def spawn(self, *, consumer_suffix: str | None = None) -> _WorkerClientStub:
        worker = _WorkerClientStub(self._backend, consumer_suffix or "worker")
        self._workers.append(worker)
        return worker

    async def ensure_consumer_group(self) -> None:
        return None

    async def close(self) -> None:
        for worker in self._workers:
            await worker.close()


async def _run_prefetch_async(
    *,
    requests: int,
    concurrency: int,
    ready_latency_ms: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    backend = _PrefetchBackend(
        ready_latency_ms=ready_latency_ms,
        jitter_ms=0.0,
        error_rate=0.0,
    )
    worker = AsyncWorker(max_concurrency=concurrency, name="kasmina-prefetch-harness")
    manager = KasminaSeedManager(runtime=_RuntimeStub(), nonce_max_entries=64)
    oona = _OonaStub(backend)
    coordinator = KasminaPrefetchCoordinator(manager, oona, async_worker=worker)
    manager.set_prefetch(coordinator)
    coordinator.start()

    try:
        for idx in range(requests):
            coordinator.request_kernel(f"seed-{idx}", f"bp-{idx}", training_run_id="harness")
        await asyncio.sleep(ready_latency_ms / 1000.0 + 0.1)
    finally:
        await coordinator.close()
        await oona.close()
        worker.shutdown(cancel_pending=True)

    packets = manager.drain_telemetry_packets()
    telemetry = [_serialize_packet(packet) for packet in packets]
    return backend.durations(), telemetry


class _RuntimeStub:
    def fetch_kernel(self, blueprint_id: str):  # pragma: no cover - trivial stub
        return nn.Identity(), 0.0


def run_kasmina_prefetch_burst(
    *,
    requests: int,
    concurrency: int,
    ready_latency_ms: float = 40.0,
    device: torch.device | None = None,
) -> ScenarioResult:
    durations, telemetry = asyncio.run(
        _run_prefetch_async(
            requests=requests,
            concurrency=concurrency,
            ready_latency_ms=ready_latency_ms,
        )
    )
    summary = {
        "requests": float(requests),
        "latency_mean_ms": statistics.fmean(durations) if durations else 0.0,
        "latency_p95_ms": _percentile(durations, 0.95),
        "latency_max_ms": max(durations) if durations else 0.0,
    }
    return ScenarioResult(
        scenario="kasmina_prefetch_burst",
        environment=_environment_snapshot(device or torch.device("cpu")),
        metrics=summary,
        telemetry=telemetry,
        details={
            "durations_ms": durations,
            "concurrency": concurrency,
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point


def _percentile(data: Iterable[float], percentile: float) -> float:
    values = list(data)
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = percentile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = rank - lower
    return lower_value + (upper_value - lower_value) * fraction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RC1 cross-system performance harness")
    parser.add_argument("scenario", choices=[
        "steady-train",
        "rollback",
        "tamiyo-timeout",
        "kasmina-prefetch",
    ])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-every", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--ready-latency-ms", type=float, default=40.0)
    parser.add_argument("--deadline-ms", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("rc1_harness_results"))
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--enable-graphs", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    device = torch.device(args.device)
    enable_compile = not args.disable_compile

    if args.scenario == "steady-train":
        result = run_steady_train(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            enable_compile=enable_compile,
            enable_graphs=args.enable_graphs,
        )
        slug = "steady_train"
    elif args.scenario == "rollback":
        result = run_rollback_deadline(
            device=device,
            batch_size=args.batch_size,
            enable_compile=enable_compile,
            enable_graphs=args.enable_graphs,
            deadline_ms=args.deadline_ms,
        )
        slug = "rollback_deadline"
    elif args.scenario == "tamiyo-timeout":
        result = run_tamiyo_timeout(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            timeout_every=args.timeout_every,
            enable_compile=enable_compile,
            enable_graphs=args.enable_graphs,
        )
        slug = "tamiyo_timeout"
    else:
        result = run_kasmina_prefetch_burst(
            requests=args.requests,
            concurrency=args.concurrency,
            ready_latency_ms=args.ready_latency_ms,
            device=device,
        )
        slug = "kasmina_prefetch"

    path = _write_results(result, args.output_dir, slug)
    return path


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
