from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch import nn

from esper.karn import BlueprintDescriptor, BlueprintTier, KarnCatalog
from esper.leyline import leyline_pb2
from esper.tezzeret import CompileJobConfig, TezzeretCompiler
from esper.urza import UrzaLibrary
from esper.urza.pipeline import BlueprintPipeline, BlueprintRequest


class _UrzaStoreAdapter:
    def __init__(self, library: UrzaLibrary) -> None:
        self._library = library

    def save_kernel(self, blueprint_id: str, artifact_path: Path) -> None:
        metadata = self._library.get(blueprint_id)
        if metadata:
            return
        # loader expects save to be delegated to library.save afterwards
        # pipeline handles actual persistence


@pytest.mark.asyncio
async def test_blueprint_pipeline_compiles_and_stores() -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-1",
        name="Test",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )
    bounds = metadata.allowed_parameters["alpha"]
    bounds.min_value = 0.0
    bounds.max_value = 1.0
    catalog = KarnCatalog(load_defaults=False)
    catalog.register(metadata)

    with TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "artifacts"
        urza_root = Path(tmp) / "urza"
        library = UrzaLibrary(root=urza_root)
        compiler = TezzeretCompiler(
            config=CompileJobConfig(artifact_dir=artifact_dir),
        )
        pipeline = BlueprintPipeline(
            catalog=catalog,
            compiler=compiler,
            library=library,
        )
        response = await pipeline.handle_request(
            BlueprintRequest(
                blueprint_id="bp-1",
                parameters={"alpha": 0.5},
                training_run_id="run-1",
            )
        )
        assert response.metadata.blueprint_id == "bp-1"
        assert Path(response.artifact_path).exists()
        artifact = torch.load(response.artifact_path)
        assert isinstance(artifact, nn.Module)
        assert artifact.blueprint_params["alpha"] == 0.5
        sample = torch.randn(8, 32)
        assert artifact(sample).shape == sample.shape
        stored = library.get("bp-1")
        assert stored is not None
        assert stored.guard_digest
        assert stored.compile_ms is not None
        assert stored.guard_spec
        assert stored.guard_spec_summary


@pytest.mark.asyncio
async def test_blueprint_pipeline_notifies_catalog_update(tmp_path) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-notify",
        name="Notify",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )
    catalog = KarnCatalog(load_defaults=False)
    catalog.register(metadata)
    artifact_dir = tmp_path / "artifacts"
    library = UrzaLibrary(root=tmp_path / "urza")
    compiler = TezzeretCompiler(CompileJobConfig(artifact_dir=artifact_dir))

    received: list[leyline_pb2.KernelCatalogUpdate] = []

    async def notifier(update: leyline_pb2.KernelCatalogUpdate) -> None:
        received.append(update)

    pipeline = BlueprintPipeline(
        catalog=catalog,
        compiler=compiler,
        library=library,
        catalog_notifier=notifier,
    )

    await pipeline.handle_request(
        BlueprintRequest(
            blueprint_id="bp-notify",
            parameters={},
            training_run_id="run-2",
        )
    )

    assert received
    assert received[0].blueprint_id == "bp-notify"
    record = library.get("bp-notify")
    assert record is not None
    assert record.compile_ms is not None
