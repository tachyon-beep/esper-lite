#!/usr/bin/env python3
"""Esper-Lite end-to-end demo.

This script exercises the Tolaria → Tamiyo → Kasmina → Simic → Nissa control loop
on a synthetic workload. It:

1. Seeds the Karn/Urza blueprint pipeline and Kasmina runtime.
2. Runs Tolaria for a small number of epochs, letting Tamiyo make decisions.
3. Publishes telemetry and field reports into Oona/Redis and ingests them with Nissa.
4. Runs Simic offline training, emits a PolicyUpdate, and has Tamiyo hot-reload it.
5. Runs Tolaria again to demonstrate the updated policy in action.

Requirements:
- Redis running at the location specified by EsperSettings (see README).
- Optional: Prometheus/Grafana stack from infra/ for live metrics.

Run with:
    python scripts/run_demo.py
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.core import EsperSettings
from esper.karn import KarnCatalog
from esper.kasmina import KasminaPrefetchCoordinator, KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.nissa import NissaIngestor, NissaIngestorConfig
from esper.oona import OonaClient, StreamConfig
from esper.simic import FieldReportReplayBuffer, SimicTrainer, SimicTrainerConfig
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext
from esper.tamiyo import TamiyoService
from esper.tezzeret import CompileJobConfig, TezzeretCompiler
from esper.tolaria import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.urza import UrzaLibrary, UrzaRuntime
from esper.urza.pipeline import BlueprintPipeline, BlueprintRequest

logger = logging.getLogger(__name__)


INPUT_DIM = 8
OUTPUT_DIM = 4
DATASET_SIZE = 64
MAX_EPOCHS = 3


class MemoryElasticsearch:
    """Minimal stub to collect indexed documents during the demo."""

    def __init__(self) -> None:
        self.documents: list[tuple[str, dict]] = []

    def index(self, index: str, document: dict) -> None:
        self.documents.append((index, document))


class DemoKasminaAdapter(KasminaClient):
    """Adapter that forwards commands to KasminaSeedManager and records them."""

    def __init__(self, manager: KasminaSeedManager) -> None:
        self._manager = manager
        self.commands: list[leyline_pb2.AdaptationCommand] = []

    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        self.commands.append(command)
        self._manager.handle_command(command)


class DemoTamiyoClient(TamiyoClient):
    """TamiyoService already implements the protocol; this wrapper is for typing."""

    def __init__(self, service: TamiyoService) -> None:
        self._service = service

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        return self._service.evaluate_epoch(state)


def build_dataloader() -> DataLoader:
    inputs = torch.randn(DATASET_SIZE, INPUT_DIM)
    targets = torch.randint(0, OUTPUT_DIM, (DATASET_SIZE,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=8, shuffle=True)


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(INPUT_DIM, 16), nn.ReLU(), nn.Linear(16, OUTPUT_DIM))


async def initialise_blueprint_pipeline(
    root: Path,
    *,
    catalog_notifier: Callable[[leyline_pb2.KernelCatalogUpdate], Awaitable[None]] | None = None,
) -> tuple[KarnCatalog, UrzaLibrary, UrzaRuntime]:
    catalog = KarnCatalog()
    blueprint_id = "BP001"
    blueprint = catalog.get(blueprint_id)
    if blueprint is None:
        raise RuntimeError("Default blueprint catalog missing BP001")

    settings = EsperSettings()
    artifact_dir = root / "artifacts"
    library = UrzaLibrary(
        root=root / "urza",
        cache_ttl_seconds=settings.urza_cache_ttl_seconds,
    )
    compiler = TezzeretCompiler(
        config=CompileJobConfig(
            artifact_dir=artifact_dir,
            inductor_cache_dir=(
                Path(settings.tezzeret_inductor_cache_dir)
                if settings.tezzeret_inductor_cache_dir
                else root / "inductor_cache"
            ),
        )
    )
    pipeline = BlueprintPipeline(
        catalog=catalog,
        compiler=compiler,
        library=library,
        catalog_notifier=catalog_notifier,
    )
    await pipeline.handle_request(
        BlueprintRequest(
            blueprint_id=blueprint_id,
            parameters={
                key: (bounds[0] + bounds[1]) / 2.0 for key, bounds in blueprint.allowed_parameters.items()
            }
            if blueprint.allowed_parameters
            else {},
            training_run_id="run-demo",
        )
    )
    runtime = UrzaRuntime(library)
    return catalog, library, runtime


def generate_field_reports(
    tamiyo: TamiyoService,
    commands: Iterable[leyline_pb2.AdaptationCommand],
    states: Iterable[leyline_pb2.SystemStatePacket],
) -> None:
    previous_loss: float | None = None
    for command, state in zip(commands, states, strict=False):
        metrics_delta = {}
        if previous_loss is not None:
            metrics_delta["loss_delta"] = state.validation_loss - previous_loss
        blueprint_id = (
            command.seed_operation.blueprint_id
            if command.HasField("seed_operation")
            else ""
        )
        report = tamiyo.generate_field_report(
            command=command,
            outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
            metrics_delta=metrics_delta,
            training_run_id=state.training_run_id,
            seed_id=command.target_seed_id or "seed-unset",
            blueprint_id=blueprint_id,
        )
        logger.debug("Generated field report %s", report.report_id)
        previous_loss = state.validation_loss


async def publish_history(
    tol: TolariaTrainer,
    tamiyo: TamiyoService,
    oona: OonaClient,
    nissa: NissaIngestor,
    settings: EsperSettings,
) -> None:
    await tol.publish_history(oona)
    await tamiyo.publish_history(oona)
    await nissa.consume_from_oona(oona, stream=settings.oona_normal_stream)
    await nissa.consume_from_oona(oona, stream=settings.oona_telemetry_stream)


async def consume_policy_updates(tamiyo: TamiyoService, oona: OonaClient) -> None:
    await tamiyo.consume_policy_updates(oona)


async def run_demo() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = EsperSettings()

    stream_config = StreamConfig(
        normal_stream=settings.oona_normal_stream,
        emergency_stream=settings.oona_emergency_stream,
        telemetry_stream=settings.oona_telemetry_stream,
        policy_stream=settings.oona_policy_stream,
        group="esper-demo",
    )
    oona = OonaClient(redis_url=settings.redis_url, config=stream_config)
    await oona.ensure_consumer_group()

    nissa = NissaIngestor(
        NissaIngestorConfig(
            prometheus_gateway=settings.prometheus_pushgateway,
            elasticsearch_url=settings.elasticsearch_url,
        ),
        es_client=MemoryElasticsearch(),
    )

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        catalog, urza_library, urza_runtime = await initialise_blueprint_pipeline(
            root,
            catalog_notifier=lambda update: oona.publish_kernel_catalog_update(update),
        )

        kasmina_manager = KasminaSeedManager(runtime=urza_runtime)
        prefetch_coordinator = KasminaPrefetchCoordinator(kasmina_manager, oona)
        kasmina_manager.set_prefetch(prefetch_coordinator)
        prefetch_coordinator.start()
        kasmina_adapter = DemoKasminaAdapter(kasmina_manager)
        try:
            tamiyo_signature = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
        except RuntimeError:
            logger.warning("ESPER_LEYLINE_SECRET not set; using demo signing secret")
            tamiyo_signature = SignatureContext(secret=b"demo-signing-secret")
        tamiyo_service = TamiyoService(urza=urza_library, signature_context=tamiyo_signature)
        tamiyo_client = DemoTamiyoClient(tamiyo_service)

        model = build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        dataloader = build_dataloader()
        trainer = TolariaTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            tamiyo=tamiyo_client,
            kasmina=kasmina_adapter,
            config=TrainingLoopConfig(max_epochs=MAX_EPOCHS),
        )

        logger.info("Running initial training loop (%s epochs)", MAX_EPOCHS)
        list(trainer.run())
        generate_field_reports(tamiyo_service, kasmina_adapter.commands, trainer.state_packets)
        await publish_history(trainer, tamiyo_service, oona, nissa, settings)

        buffer = FieldReportReplayBuffer(capacity=256)
        buffer.extend(tamiyo_service.field_reports)
        simic_trainer = SimicTrainer(
            policy=None,
            buffer=buffer,
            config=SimicTrainerConfig(
                epochs=2,
                batch_size=16,
                hidden_size=32,
                use_lora=True,
                lora_rank=4,
            ),
        )
        logger.info("Running Simic offline training")
        simic_trainer.run_training()
        simic_trainer.create_policy_update(
            policy_id="policy-demo",
            training_run_id="run-demo",
            policy_version="policy-updated",
        )
        await simic_trainer.publish_metrics(oona, training_run_id="run-demo")
        await simic_trainer.publish_policy_updates(oona)
        await consume_policy_updates(tamiyo_service, oona)
        logger.info(
            "Simic training complete: loss=%.4f, policy updates=%s",
            simic_trainer.last_loss,
            len(simic_trainer.policy_updates),
        )
        logger.info("Tamiyo applied %s policy updates", len(tamiyo_service.policy_updates))

        command_offset = len(kasmina_adapter.commands)
        state_offset = len(trainer.state_packets)

        logger.info("Running training loop after policy update")
        list(trainer.run())
        generate_field_reports(
            tamiyo_service,
            kasmina_adapter.commands[command_offset:],
            trainer.state_packets[state_offset:],
        )
        await publish_history(trainer, tamiyo_service, oona, nissa, settings)

        logger.info(
            "Demo complete: %s telemetry packets, %s field reports, %s policy updates",
            len(trainer.telemetry_packets),
            len(tamiyo_service.field_reports),
            len(tamiyo_service.policy_updates),
        )

        await prefetch_coordinator.close()
    await oona.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
