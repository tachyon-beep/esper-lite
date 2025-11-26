#!/usr/bin/env python3
"""Capture Phase 0 performance baselines for Tolaria/Tamiyo/Kasmina."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import esper.tolaria.trainer as trainer_mod
from esper.leyline import leyline_pb2
from esper.tamiyo import FieldReportStoreConfig, TamiyoPolicy, TamiyoPolicyConfig, TamiyoService
from esper.security.signing import SignatureContext
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.tolaria.emergency import EmergencyController
from esper.tolaria.rollback import RollbackResult
from esper.urza import UrzaLibrary

from scripts.bench_kasmina_prefetch import run_benchmark


class _TamiyoServiceClient(TamiyoClient):
    def __init__(self, service: TamiyoService) -> None:
        self._service = service

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        return self._service.evaluate_epoch(state)


class _KasminaPassthrough(KasminaClient):
    def __init__(self) -> None:
        self.commands: list[leyline_pb2.AdaptationCommand] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.commands.append(command)


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Phase 0 baseline capture; enable GPU access and retry")


def _build_training_artifacts(
    device: torch.device,
    *,
    dataset_on_gpu: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int | None,
    pin_memory: bool,
    pin_memory_device: str | None,
    enable_graphs: bool,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    torch.manual_seed(1337)
    features = 16
    samples = 16
    model = nn.Linear(features, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data_device = device if dataset_on_gpu else torch.device("cpu")
    inputs = torch.randn(samples, features, device=data_device)
    targets = torch.randint(0, 4, (samples,), device=data_device)
    dataset = TensorDataset(inputs, targets)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "pin_memory": pin_memory,
    }
    if dataset_on_gpu:
        # Pinned memory and CPU prefetching are unnecessary when tensors reside on GPU.
        loader_kwargs["pin_memory"] = False
        loader_kwargs["num_workers"] = 0
        loader_kwargs.pop("shuffle", None)
        loader_kwargs["shuffle"] = False
    # Disable pinning when graphs are enabled to avoid capture conflicts.
    if enable_graphs:
        loader_kwargs["pin_memory"] = False
        loader_kwargs["num_workers"] = 0
        loader_kwargs["shuffle"] = False

    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        if pin_memory_device:
            loader_kwargs["pin_memory_device"] = pin_memory_device
    else:
        loader_kwargs.pop("pin_memory", None)
        loader_kwargs["pin_memory"] = False
    dataloader = DataLoader(dataset, **loader_kwargs)
    return model, optimizer, dataloader


def _build_trainer(
    *,
    tamiyo: TamiyoClient,
    kasmina: KasminaClient,
    device: torch.device,
    tamiyo_timeout_s: float = 2.0,
    dataset_on_gpu: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int | None,
    pin_memory: bool,
    pin_memory_device: str | None,
    disable_compile: bool,
    compile_mode: str,
    compile_dynamic: bool,
    compile_warmup_steps: int,
    enable_graphs: bool,
    graph_warmup_batches: int,
    enable_gpu_prefetch: bool,
) -> TolariaTrainer:
    model, optimizer, dataloader = _build_training_artifacts(
        device,
        dataset_on_gpu=dataset_on_gpu,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        enable_graphs=enable_graphs,
    )
    config = TrainingLoopConfig(
        max_epochs=1,
        device=device,
        gradient_accumulation_steps=1,
        epoch_budget_ms=18.0,
        hook_budget_ms=5.0,
        tamiyo_timeout_s=tamiyo_timeout_s,
        enable_compile=not disable_compile,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_warmup_steps=compile_warmup_steps,
        enable_graphs=enable_graphs and disable_compile,
        graph_warmup_batches=graph_warmup_batches,
        enable_amp=False,
        enable_tf32=True,
        enable_foreach_optim=True,
    )
    return TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=config,
    )


def _build_tamiyo_service(tmp_root: Path, *, timeout_ms: float = 1000.0) -> TamiyoService:
    policy_cfg = TamiyoPolicyConfig(
        enable_compile=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy = TamiyoPolicy(policy_cfg)
    store_cfg = FieldReportStoreConfig(path=tmp_root / "field_reports.log")
    urza = UrzaLibrary(root=tmp_root / "urza")
    signature = SignatureContext(secret=b"phase0-tamiyo-secret")
    service = TamiyoService(
        policy=policy,
        store_config=store_cfg,
        urza=urza,
        signature_context=signature,
        step_timeout_ms=timeout_ms,
    )
    return service


def _packet_summary(packet: leyline_pb2.TelemetryPacket) -> Dict[str, Any]:
    return {
        "metrics": {metric.name: metric.value for metric in packet.metrics},
        "events": [
            {
                "description": event.description,
                "level": int(event.level),
                "attributes": dict(event.attributes),
            }
            for event in packet.events
        ],
        "priority": packet.system_health.indicators.get("priority"),
    }


def _run_tolaria_scenario(
    *,
    tamiyo_service: TamiyoService,
    kasmina_client: KasminaClient,
    device: torch.device,
    tamiyo_timeout_s: float,
    rollback_simulator: Any | None = None,
    dataset_on_gpu: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int | None,
    pin_memory: bool,
    pin_memory_device: str | None,
    disable_compile: bool,
    compile_mode: str,
    compile_dynamic: bool,
    compile_warmup_steps: int,
    enable_graphs: bool,
    graph_warmup_batches: int,
    enable_gpu_prefetch: bool,
) -> Dict[str, Any]:
    tamiyo_client = _TamiyoServiceClient(tamiyo_service)
    trainer = _build_trainer(
        tamiyo=tamiyo_client,
        kasmina=kasmina_client,
        device=device,
        tamiyo_timeout_s=tamiyo_timeout_s,
        dataset_on_gpu=dataset_on_gpu,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        disable_compile=disable_compile,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_warmup_steps=compile_warmup_steps,
        enable_graphs=enable_graphs,
        graph_warmup_batches=graph_warmup_batches,
        enable_gpu_prefetch=enable_gpu_prefetch,
    )
    if rollback_simulator is not None:
        trainer._settings = trainer._settings.model_copy(update=rollback_simulator["settings"])  # type: ignore[attr-defined]
        trainer._emergency = EmergencyController(bypass_cap_per_min=1)
        trainer._metrics.setdefault("tolaria.emergency.broadcasts_total", 0.0)
        trainer._metrics.setdefault("tolaria.emergency.bypass_applied_total", 0.0)
        trainer._metrics.setdefault("tolaria.emergency.halts_total", 0.0)
        trainer._metrics.setdefault("tolaria.emergency.halt", 0.0)
        trainer._fast_cache = object()
        original = trainer_mod.attempt_two_tier_rollback  # type: ignore[name-defined]

        def _fake_attempt(**kwargs):
            return RollbackResult(False, 12_000.0, False, error="deadline_exceeded")

        trainer_mod.attempt_two_tier_rollback = _fake_attempt  # type: ignore[name-defined]
    else:
        original = None
    try:
        list(trainer.run())
        tolaria_packet = trainer.telemetry_packets[0]
    finally:
        trainer.close()
        if original is not None:
            trainer_mod.attempt_two_tier_rollback = original  # type: ignore[name-defined]
    tamiyo_packets = tamiyo_service.telemetry_packets
    tamiyo_summary = _packet_summary(tamiyo_packets[-1]) if tamiyo_packets else {}
    return {
        "tolaria": _packet_summary(tolaria_packet),
        "tamiyo": tamiyo_summary,
    }


def _capture_normal_epoch(
    device: torch.device,
    dataloader_opts: dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase0-tamiyo-") as tmp:
        service = _build_tamiyo_service(Path(tmp))
        kasmina = _KasminaPassthrough()
        result = _run_tolaria_scenario(
            tamiyo_service=service,
            kasmina_client=kasmina,
            device=device,
            tamiyo_timeout_s=2.0,
            **dataloader_opts,
        )
        service.close()
    return result


def _capture_tamiyo_timeout(
    device: torch.device,
    dataloader_opts: dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase0-tamiyo-timeout-") as tmp:
        service = _build_tamiyo_service(Path(tmp), timeout_ms=1.0)
        kasmina = _KasminaPassthrough()
        result = _run_tolaria_scenario(
            tamiyo_service=service,
            kasmina_client=kasmina,
            device=device,
            tamiyo_timeout_s=0.001,
            **dataloader_opts,
        )
        service.close()
    return result


def _capture_rollback_deadline(
    device: torch.device,
    dataloader_opts: dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase0-tamiyo-rollback-") as tmp:
        service = _build_tamiyo_service(Path(tmp))
        kasmina = _KasminaPassthrough()
        result = _run_tolaria_scenario(
            tamiyo_service=service,
            kasmina_client=kasmina,
            device=device,
            tamiyo_timeout_s=0.001,
            rollback_simulator={
                "settings": {
                    "tolaria_emergency_enabled": True,
                    "tolaria_emergency_l4_on_rollback_deadline": True,
                    "tolaria_rollback_deadline_ms": 5,
                    "tolaria_rollback_enabled": True,
                }
            },
            **dataloader_opts,
        )
        service.close()
    return result


def _capture_kasmina_prefetch() -> Dict[str, Any]:
    async def _run() -> Dict[str, Any]:
        stats = await run_benchmark(
            requests=256,
            ready_latency_ms=45.0,
            jitter_ms=8.0,
            error_rate=0.0,
            concurrency=6,
            issue_delay_ms=1.0,
            timeout_s=45.0,
        )
        return {
            "requests": stats.total_requests,
            "ready": stats.ready,
            "errors": stats.errors,
            "latency_ms": stats.latency_summary(),
        }

    return asyncio.run(_run())


def _gather_env() -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_devices": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Phase 0 performance baselines")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/phase0"),
    )
    parser.add_argument("--gpu-dataset", action="store_true", help="Pre-materialise synthetic dataset on CUDA before loading")
    parser.add_argument("--dataloader-workers", type=int, default=2, help="Number of DataLoader workers for GPU runs (ignored if dataset on GPU)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor when workers > 0")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent_workers even when workers > 0")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory when using CPU dataset")
    parser.add_argument("--pin-memory-device", type=str, default="cuda", help="pin_memory_device to use when pin_memory is enabled")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile path for Tolaria trainer")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", help="torch.compile mode to use when enabled")
    parser.add_argument("--compile-dynamic", action="store_true", help="Enable dynamic=True when compiling the training step")
    parser.add_argument("--compile-warmup-steps", type=int, default=1, help="Number of eager batches to run before compiling")
    parser.add_argument("--enable-graphs", action="store_true", help="Enable CUDA graph capture (only active when compile is disabled)")
    parser.add_argument("--graph-warmup-batches", type=int, default=1, help="Number of eager batches before attempting graph capture")
    parser.add_argument("--enable-gpu-prefetch", action="store_true", help="Enable Tolaria GPU prefetch staging buffers")
    args = parser.parse_args()

    os.environ.setdefault("ESPER_LEYLINE_SECRET", "phase0-secret")

    _ensure_cuda()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    dataset_on_gpu = bool(args.gpu_dataset)
    num_workers = max(0, args.dataloader_workers)
    persistent_workers = not args.no_persistent_workers
    pin_memory = not args.no_pin_memory
    pin_memory_device = args.pin_memory_device if pin_memory and not dataset_on_gpu else None
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        prefetch_factor = max(1, args.prefetch_factor)

    enable_graphs = bool(args.enable_graphs)
    enable_gpu_prefetch = bool(args.enable_gpu_prefetch)
    disable_compile = bool(args.no_compile) or enable_graphs

    dataloader_opts = {
        "dataset_on_gpu": dataset_on_gpu,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "pin_memory_device": pin_memory_device,
        "disable_compile": disable_compile,
        "compile_mode": args.compile_mode,
        "compile_dynamic": bool(args.compile_dynamic),
        "compile_warmup_steps": max(0, args.compile_warmup_steps),
        "enable_graphs": enable_graphs,
        "graph_warmup_batches": max(0, args.graph_warmup_batches),
        "enable_gpu_prefetch": enable_gpu_prefetch,
    }

    baselines = {
        "tolaria_normal_epoch": _capture_normal_epoch(device, dataloader_opts),
        "tolaria_tamiyo_timeout": _capture_tamiyo_timeout(device, dataloader_opts),
        "tolaria_rollback_deadline": _capture_rollback_deadline(device, dataloader_opts),
        "kasmina_prefetch_benchmark": _capture_kasmina_prefetch(),
    }

    for name, data in baselines.items():
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_gather_env(), handle, indent=2, sort_keys=True)

    print("Captured baselines:")
    for name in baselines:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
