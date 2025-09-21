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


class _RuntimeStub:
    def __init__(self) -> None:
        self.loaded = []

    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        self.loaded.append(blueprint_id)
        return nn.Identity(), 1.0


def test_lifecycle_transitions_follow_order() -> None:
    lifecycle = KasminaLifecycle()
    lifecycle.transition(leyline_pb2.SEED_STAGE_GERMINATING)
    lifecycle.transition(leyline_pb2.SEED_STAGE_TRAINING)
    assert lifecycle.state == leyline_pb2.SEED_STAGE_TRAINING


def test_lifecycle_rejects_invalid_transition() -> None:
    lifecycle = KasminaLifecycle()
    with pytest.raises(ValueError):
        lifecycle.transition(leyline_pb2.SEED_STAGE_TRAINING)


def test_seed_manager_grafts_and_retires_seed() -> None:
    runtime = _RuntimeStub()
    manager = KasminaSeedManager(runtime=runtime)
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-1",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "bp-1"
    manager.handle_command(command)
    assert "seed-1" in manager.seeds()
    assert runtime.loaded == ["bp-1"]

    retire = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-2",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    retire.seed_operation.operation = leyline_pb2.SEED_OP_CULL
    manager.handle_command(retire)
    assert "seed-1" not in manager.seeds()


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
        manager = KasminaSeedManager(runtime=urza_runtime)
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-urza",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.blueprint_id = "bp-1"
        manager.handle_command(command)
        assert "seed-urza" in manager.seeds()


def test_full_lifecycle_path() -> None:
    lc = KasminaLifecycle()
    assert lc.state == leyline_pb2.SEED_STAGE_UNKNOWN
    lc.transition(leyline_pb2.SEED_STAGE_GERMINATING)
    lc.transition(leyline_pb2.SEED_STAGE_GRAFTING)
    lc.transition(leyline_pb2.SEED_STAGE_STABILIZING)
    lc.transition(leyline_pb2.SEED_STAGE_TRAINING)
    lc.transition(leyline_pb2.SEED_STAGE_EVALUATING)
    lc.transition(leyline_pb2.SEED_STAGE_FINE_TUNING)
    lc.transition(leyline_pb2.SEED_STAGE_EVALUATING)
    lc.transition(leyline_pb2.SEED_STAGE_FOSSILIZED)
    lc.transition(leyline_pb2.SEED_STAGE_CULLING)
    lc.transition(leyline_pb2.SEED_STAGE_CANCELLED)
    assert lc.state == leyline_pb2.SEED_STAGE_CANCELLED


def test_fast_path_to_training_and_cull() -> None:
    lc = KasminaLifecycle()
    lc.transition(leyline_pb2.SEED_STAGE_GERMINATING)
    lc.transition(leyline_pb2.SEED_STAGE_TRAINING)  # fast path allowed
    lc.transition(leyline_pb2.SEED_STAGE_CULLING)
    lc.transition(leyline_pb2.SEED_STAGE_CANCELLED)
    assert lc.state == leyline_pb2.SEED_STAGE_CANCELLED


def test_cancel_from_unknown() -> None:
    lc = KasminaLifecycle()
    lc.transition(leyline_pb2.SEED_STAGE_CANCELLED)
    assert lc.state == leyline_pb2.SEED_STAGE_CANCELLED
