from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from esper.karn import BlueprintMetadata, BlueprintTier
from esper.tezzeret import CompileJobConfig, TezzeretCompiler


def test_compiler_persists_artifact() -> None:
    metadata = BlueprintMetadata(
        blueprint_id="bp-1",
        name="Test",
        tier=BlueprintTier.SAFE,
        description="",
        allowed_parameters={},
    )
    with TemporaryDirectory() as tmp:
        config = CompileJobConfig(artifact_dir=Path(tmp))
        compiler = TezzeretCompiler(config=config)
        path = compiler.compile(metadata, parameters={"alpha": 0.1})
        assert path.exists()
        module = torch.load(path)
        assert module.blueprint_params["alpha"] == 0.1


def test_compiler_retries_and_clears_wal(tmp_path) -> None:
    metadata = BlueprintMetadata(
        blueprint_id="BP001",
        name="identity_passthrough",
        tier=BlueprintTier.SAFE,
        description="Identity",
    )

    failures: list[bool] = [True, False]

    def sampler(_: BlueprintMetadata) -> bool:
        return failures.pop(0)

    config = CompileJobConfig(artifact_dir=tmp_path, max_retries=1)
    compiler = TezzeretCompiler(config=config, error_sampler=sampler)
    path = compiler.compile(metadata, parameters={})
    assert path.exists()
    assert not (tmp_path / "tezzeret_wal.json").exists()


def test_compiler_raises_after_retries(tmp_path) -> None:
    metadata = BlueprintMetadata(
        blueprint_id="BP002",
        name="linear_projection",
        tier=BlueprintTier.SAFE,
        description="Linear",
    )

    def sampler(_: BlueprintMetadata) -> bool:
        return True

    config = CompileJobConfig(artifact_dir=tmp_path, max_retries=1)
    compiler = TezzeretCompiler(config=config, error_sampler=sampler)
    with pytest.raises(RuntimeError):
        compiler.compile(metadata, parameters={})
    assert (tmp_path / "tezzeret_wal.json").exists()
