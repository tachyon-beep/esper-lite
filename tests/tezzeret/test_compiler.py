from pathlib import Path
from tempfile import TemporaryDirectory

from esper.karn import BlueprintMetadata, BlueprintTier
from esper.tezzeret import CompileJobConfig, TezzeretCompiler


class _UrzaStoreStub:
    def __init__(self) -> None:
        self.saved: dict[str, Path] = {}

    def save_kernel(self, blueprint_id: str, artifact_path: Path) -> None:
        self.saved[blueprint_id] = artifact_path


def test_compiler_persists_artifact() -> None:
    metadata = BlueprintMetadata(
        blueprint_id="bp-1",
        name="Test",
        tier=BlueprintTier.SAFE,
        description="",
        allowed_parameters={},
    )
    store = _UrzaStoreStub()
    with TemporaryDirectory() as tmp:
        config = CompileJobConfig(artifact_dir=Path(tmp))
        compiler = TezzeretCompiler(artifact_store=store, config=config)
        path = compiler.compile(metadata)
        assert path.exists()
        assert store.saved["bp-1"] == path
