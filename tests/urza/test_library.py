from pathlib import Path
from tempfile import TemporaryDirectory

from esper.karn import BlueprintMetadata, BlueprintTier
from esper.urza import UrzaLibrary


def test_urza_library_saves_artifact() -> None:
    metadata = BlueprintMetadata(
        blueprint_id="bp-1",
        name="Test",
        tier=BlueprintTier.SAFE,
        description="",
        allowed_parameters={},
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        library = UrzaLibrary(root=root)
        artifact = root / "artifact.pt"
        artifact.write_bytes(b"data")
        library.save(metadata, artifact)
        record = library.get("bp-1")
        assert record is not None
        assert record.artifact_path.exists()
