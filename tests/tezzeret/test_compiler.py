from pathlib import Path
from tempfile import TemporaryDirectory

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
