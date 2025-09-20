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
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.core import EsperSettings
from esper.karn import BlueprintMetadata, BlueprintTier, KarnCatalog
from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.nissa import NissaIngestor, NissaIngestorConfig
from esper.oona import OonaClient, StreamConfig
from esper.simic import FieldReportReplayBuffer, SimicTrainer, SimicTrainerConfig
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


def initialise_blueprint_pipeline(root: Path) -> tuple[KarnCatalog, UrzaLibrary, UrzaRuntime]:
    catalog = KarnCatalog()
    metadata = BlueprintMetadata(
        blueprint_id="bp-demo",
        name="Demo Blueprint",
        tier=BlueprintTier.SAFE,
        description="Synthetic graft for demo",
        allowed_parameters={"alpha": (0.0, 1.0)},
    )
    catalog.register(metadata)

    artifact_dir = root / "artifacts"
    library = UrzaLibrary(root=root / "urza")
    compiler = TezzeretCompiler(config=CompileJobConfig(artifact_dir=artifact_dir))
    pipeline = BlueprintPipeline(catalog=catalog, compiler=compiler, library=library)
    pipeline.handle_request(
        BlueprintRequest(
            blueprint_id="bp-demo",
            parameters={"alpha": 0.5},
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
        _, _, urza_runtime = initialise_blueprint_pipeline(root)

        kasmina_manager = KasminaSeedManager(runtime=urza_runtime)
        kasmina_adapter = DemoKasminaAdapter(kasmina_manager)
        tamiyo_service = TamiyoService()
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

    await oona.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
