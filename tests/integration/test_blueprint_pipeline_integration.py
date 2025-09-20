from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintMetadata, BlueprintTier, KarnCatalog
from esper.leyline import leyline_pb2
from esper.tamiyo import TamiyoService
from esper.tamiyo.persistence import FieldReportStoreConfig
from esper.tezzeret import CompileJobConfig, TezzeretCompiler
from esper.urza import UrzaLibrary
from esper.urza.pipeline import BlueprintPipeline, BlueprintRequest


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


def test_tamiyo_end_to_end_blueprint_pipeline(tmp_path) -> None:
    metadata = BlueprintMetadata(
        blueprint_id="bp-end",
        name="End-to-End",
        tier=BlueprintTier.EXPERIMENTAL,
        description="Pipeline integration",
        allowed_parameters={"alpha": (0.1, 0.9)},
        risk=0.85,
        stage=3,
        quarantine_only=False,
        approval_required=False,
    )

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
    response = pipeline.handle_request(request)

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
    assert command.annotations["blueprint_tier"] == metadata.tier.value
    assert command.annotations["blueprint_risk"] == f"{metadata.risk:.2f}"
    assert command.annotations["blueprint_stage"] == str(metadata.stage)

    telemetry = service.telemetry_packets[-1]
    metric_names = {metric.name for metric in telemetry.metrics}
    assert "tamiyo.blueprint.risk" in metric_names
    assert any(event.description in {"bp_quarantine", "pause_triggered"} for event in telemetry.events)
