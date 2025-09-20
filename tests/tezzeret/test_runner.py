from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from esper.karn import KarnCatalog
from esper.tezzeret import CompileJobConfig, TezzeretCompiler, TezzeretForge
from esper.urza import UrzaLibrary


def _catalog_subset(count: int) -> KarnCatalog:
    source = KarnCatalog()
    subset = KarnCatalog(load_defaults=False)
    for idx, metadata in enumerate(source.all()):
        if idx >= count:
            break
        subset.register(metadata)
    return subset


def _setup(tmp_path: Path, catalog: KarnCatalog, *, error_sampler=None, max_retries: int = 0):
    artifact_dir = tmp_path / "artifacts"
    config = CompileJobConfig(artifact_dir=artifact_dir, max_retries=max_retries)
    compiler = TezzeretCompiler(config, error_sampler=error_sampler)
    library = UrzaLibrary(root=tmp_path / "urza")
    wal_path = tmp_path / "forge_wal.json"
    forge = TezzeretForge(catalog, library, compiler, wal_path=wal_path)
    return forge, library, compiler, wal_path


def test_forge_compiles_catalog(tmp_path) -> None:
    catalog = _catalog_subset(5)
    forge, library, _, wal_path = _setup(tmp_path, catalog)
    forge.run()
    assert len(library.list_all()) == 5
    assert not wal_path.exists()


def test_forge_skips_existing(tmp_path) -> None:
    catalog = _catalog_subset(3)
    forge, library, compiler, wal_path = _setup(tmp_path, catalog)
    first = next(iter(catalog.all()))
    artifact = compiler.compile(first, {})
    library.save(first, artifact)
    forge.run()
    assert len(library.list_all()) == 3
    assert not wal_path.exists()


def test_forge_resumes_from_wal(tmp_path) -> None:
    catalog = _catalog_subset(2)
    failures = {"during": True}

    def sampler(metadata):
        if failures.get("during") and metadata.blueprint_id.endswith("1"):
            failures["during"] = False
            return True
        return False

    forge, library, compiler, wal_path = _setup(tmp_path, catalog, error_sampler=sampler, max_retries=0)

    with pytest.raises(RuntimeError):
        forge.run()
    assert wal_path.exists()
    pending = wal_path.read_text(encoding="utf-8")
    assert "BP001" in pending or "BP" in pending

    forge.run()
    assert len(library.list_all()) == 2
    assert not wal_path.exists()
