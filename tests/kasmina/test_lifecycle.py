from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from torch import nn

from esper.karn import BlueprintDescriptor, BlueprintTier, KarnCatalog
from esper.kasmina import KasminaLifecycle, KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.tezzeret import CompileJobConfig, TezzeretCompiler
from esper.urza import UrzaLibrary, UrzaRuntime
from esper.urza.pipeline import BlueprintPipeline, BlueprintRequest
from esper.security.signing import SignatureContext, sign


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
    _sign_command(retire)
    manager.handle_command(retire)
    assert "seed-1" not in manager.seeds()
    payload = manager.rollback_payload("seed-1")
    assert payload is not None
    assert payload["reason"] == "retired"


def test_seed_manager_with_urza_runtime() -> None:
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
        pipeline.handle_request(
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
_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-test-secret")


def _sign_command(command: leyline_pb2.AdaptationCommand) -> None:
    command.issued_at.GetCurrentTime()
    command.annotations["signature"] = sign(command.SerializeToString(), _SIGNING_CONTEXT)
