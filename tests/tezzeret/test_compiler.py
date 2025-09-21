from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.tezzeret import CompileJobConfig, TezzeretCompiler


def test_compiler_persists_artifact() -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-1",
        name="Test",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )
    with TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        config = CompileJobConfig(artifact_dir=Path(tmp), inductor_cache_dir=cache_dir)
        compiler = TezzeretCompiler(config=config)
        path = compiler.compile(metadata, parameters={"alpha": 0.1})
        assert path.exists()
        module = torch.load(path)
        assert module.blueprint_params["alpha"] == 0.1
        sample = torch.randn(8, 32)
        output = module(sample)
        assert output.shape == sample.shape
        update = compiler.latest_catalog_update()
        assert update is not None
        assert update.blueprint_id == "bp-1"
        assert update.artifact_ref == str(path)
        assert update.checksum
    result = compiler.latest_result()
    assert result is not None
    assert result.guard_spec
    assert result.guard_digest == update.guard_digest
    assert result.guard_summary
    assert all(isinstance(entry, str) for entry in result.guard_summary)
    if result.compile_strategy == "standard":
        assert result.eager_fallback is False
    else:
        assert result.compile_strategy == "eager"
        assert result.eager_fallback is True


def test_compiler_falls_back_to_eager(monkeypatch) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-fallback",
        name="Fallback",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )

    def failing_compile(module, dynamic=True):  # type: ignore[unused-argument]
        raise RuntimeError("compile failure")

    monkeypatch.setattr(torch, "compile", failing_compile)

    with TemporaryDirectory() as tmp:
        config = CompileJobConfig(artifact_dir=Path(tmp))
        compiler = TezzeretCompiler(config=config)
        path = compiler.compile(metadata, parameters={})
        assert path.exists()
        module = torch.load(path)
        sample = torch.randn(8, 32)
        _ = module(sample)
    result = compiler.latest_result()
    assert result is not None
    assert result.eager_fallback is True
    assert result.compile_strategy == "eager"
    assert result.guard_summary


def test_compiler_retries_and_clears_wal(tmp_path) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="BP001",
        name="identity_passthrough",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="Identity",
    )

    failures: list[bool] = [True, False]

    def sampler(_: BlueprintDescriptor) -> bool:
        return failures.pop(0)

    config = CompileJobConfig(artifact_dir=tmp_path, max_retries=1)
    compiler = TezzeretCompiler(config=config, error_sampler=sampler)
    path = compiler.compile(metadata, parameters={})
    assert path.exists()
    assert not (tmp_path / "tezzeret_wal.json").exists()


def test_compiler_raises_after_retries(tmp_path) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="BP002",
        name="linear_projection",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="Linear",
    )

    def sampler(_: BlueprintDescriptor) -> bool:
        return True

    config = CompileJobConfig(artifact_dir=tmp_path, max_retries=1)
    compiler = TezzeretCompiler(config=config, error_sampler=sampler)
    with pytest.raises(RuntimeError):
        compiler.compile(metadata, parameters={})
    assert (tmp_path / "tezzeret_wal.json").exists()


def test_compile_job_config_uses_env_cache(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    config = CompileJobConfig(artifact_dir=tmp_path)
    assert config.inductor_cache_dir == cache_dir
    compiler = TezzeretCompiler(config=config)
    metadata = BlueprintDescriptor(
        blueprint_id="bp-env",
        name="Env",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )
    path = compiler.compile(metadata, parameters={})
    assert path.exists()
    result = compiler.latest_result()
    assert result is not None
    assert result.inductor_cache_dir == str(cache_dir)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)


def test_guard_digest_consistency(tmp_path) -> None:
    metadata = BlueprintDescriptor(
        blueprint_id="bp-digest",
        name="Digest",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        description="",
    )
    compiler = TezzeretCompiler(CompileJobConfig(artifact_dir=tmp_path))
    compiler.compile(metadata, parameters={})
    first = compiler.latest_result()
    assert first is not None
    compiler.compile(metadata, parameters={})
    second = compiler.latest_result()
    assert second is not None
    assert first.guard_digest == second.guard_digest
    assert first.guard_summary == second.guard_summary
