from __future__ import annotations

from pathlib import Path

import pytest

from esper.karn import BlueprintDescriptor, BlueprintTier, KarnCatalog
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext
from esper.tamiyo import TamiyoPolicy, TamiyoService
from esper.tamiyo.persistence import FieldReportStoreConfig
from esper.tezzeret import CompileJobConfig, TezzeretCompiler
from esper.urza import UrzaLibrary
from esper.urza.pipeline import BlueprintPipeline, BlueprintRequest
from esper.oona import OonaClient, StreamConfig
from fakeredis.aioredis import FakeRedis


class _StaticPolicy:
    """Deterministic Tamiyo policy stub emitting a fixed blueprint command."""

    def __init__(self, blueprint_id: str) -> None:
        self._blueprint_id = blueprint_id
        self._last_action: dict[str, float] = {"action": 0.0, "param_delta": 0.0}

    def select_action(self, _: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd-static",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-static",
        )
        command.issued_at.GetCurrentTime()
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = self._blueprint_id
        command.seed_operation.parameters["alpha"] = 0.25
        self._last_action = {"action": 0.0, "param_delta": 0.0}
        return command

    @property
    def last_action(self) -> dict[str, float]:
        return dict(self._last_action)

    def load_state_dict(self, *_args, **_kwargs) -> None:  # pragma: no cover - stub compliance
        return None

    def state_dict(self) -> dict[str, float]:  # pragma: no cover - stub compliance
        return {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tamiyo_end_to_end_blueprint_pipeline(tmp_path) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-end",
        name="End-to-End",
        tier=BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL,
        description="Pipeline integration",
        risk=0.85,
        stage=3,
        quarantine_only=False,
        approval_required=False,
    )
    alpha_bounds = metadata.allowed_parameters["alpha"]
    alpha_bounds.min_value = 0.1
    alpha_bounds.max_value = 0.9

    catalog = KarnCatalog(load_defaults=False)
    catalog.register(metadata)

    artifact_dir = tmp_path / "artifacts"
    urza_root = tmp_path / "urza"
    library = UrzaLibrary(root=urza_root)
    compiler = TezzeretCompiler(CompileJobConfig(artifact_dir=artifact_dir))
    pipeline = BlueprintPipeline(catalog=catalog, compiler=compiler, library=library)

    request = BlueprintRequest(
        blueprint_id=metadata.blueprint_id,
        parameters={"alpha": 0.2},
        training_run_id="run-123",
    )
    response = await pipeline.handle_request(request)

    assert Path(response.artifact_path).exists()
    record = library.get(metadata.blueprint_id)
    assert record is not None
    assert record.metadata.stage == metadata.stage

    policy = _StaticPolicy(metadata.blueprint_id)
    store_path = tmp_path / "field_reports.log"
    service = TamiyoService(
        policy=policy,
        store_config=FieldReportStoreConfig(path=store_path),
        urza=library,
        signature_context=SignatureContext(secret=b"tamiyo-test-secret"),
    )

    state = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=2,
        training_run_id="run-123",
        packet_id="pkt-123",
    )
    state.timestamp_ns = 42

    command = service.evaluate_epoch(state)

    assert command.command_type == leyline_pb2.COMMAND_PAUSE
    assert command.annotations["blueprint_tier"] == leyline_pb2.BlueprintTier.Name(metadata.tier)
    assert command.annotations["blueprint_risk"] == f"{metadata.risk:.2f}"
    assert command.annotations["blueprint_stage"] == str(metadata.stage)

    telemetry = service.telemetry_packets[-1]
    metric_names = {metric.name for metric in telemetry.metrics}
    assert "tamiyo.blueprint.risk" in metric_names
    assert any(event.description in {"bp_quarantine", "pause_triggered"} for event in telemetry.events)


@pytest.mark.asyncio
async def test_tamiyo_telemetry_flows_to_oona_emergency(tmp_path) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-route",
        name="Route",
        tier=BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL,
        description="Emergency routing",
        risk=0.95,
        stage=3,
        quarantine_only=True,
        approval_required=False,
    )
    artifact = tmp_path / "artifact.pt"
    artifact.write_text("demo")
    library = UrzaLibrary(root=tmp_path / "urza")
    library.save(metadata, artifact)

    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=library,
        signature_context=SignatureContext(secret=b"tamiyo-test-secret"),
    )
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-routing",
    )
    service.evaluate_step(packet)
    telemetry = service.telemetry_packets[-1]
    telemetry.system_health.indicators["priority"] = "MESSAGE_PRIORITY_CRITICAL"
    if telemetry.events:
        telemetry.events[0].level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
    else:
        telemetry.events.add(
            description="bp_quarantine",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
        )

    redis = FakeRedis()
    stream_config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="tamiyo-routing",
    )
    oona = OonaClient("redis://localhost", config=stream_config, redis_client=redis)
    await oona.ensure_consumer_group()
    await service.publish_history(oona)
    assert await oona.stream_length("oona.emergency") >= 1
    await oona.close()
