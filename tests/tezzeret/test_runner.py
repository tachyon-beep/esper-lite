import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from esper.karn import BlueprintDescriptor, KarnCatalog
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


def _setup(
    tmp_path: Path,
    catalog: KarnCatalog,
    *,
    error_sampler=None,
    max_retries: int = 0,
    breaker_threshold: int = 3,
    compile_timeout_s: float = 60.0,
):
    artifact_dir = tmp_path / "artifacts"
    config = CompileJobConfig(artifact_dir=artifact_dir, max_retries=max_retries)
    compiler = TezzeretCompiler(config, error_sampler=error_sampler)
    library = UrzaLibrary(root=tmp_path / "urza")
    wal_path = tmp_path / "forge_wal.json"
    forge = TezzeretForge(
        catalog,
        library,
        compiler,
        wal_path=wal_path,
        compile_timeout_s=compile_timeout_s,
        breaker_threshold=breaker_threshold,
    )
    return forge, library, compiler, wal_path


def test_forge_compiles_catalog(tmp_path) -> None:
    catalog = _catalog_subset(5)
    forge, library, _, wal_path = _setup(tmp_path, catalog)
    forge.run()
    assert len(library.list_all()) == 5
    assert not wal_path.exists()
    record = library.get(next(iter(catalog.all())).blueprint_id)
    assert record is not None
    assert record.compile_ms is not None
    assert record.guard_spec
    metrics = forge.metrics_snapshot()
    assert metrics["tezzeret.compilation.total"] >= 5.0
    assert metrics["tezzeret.jobs.total"] >= 5.0
    assert "tezzeret.compilation.last_compile_ms" in metrics


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
    first_meta = next(iter(catalog.all()))

    with pytest.raises(RuntimeError):
        forge.run()
    assert wal_path.exists()
    pending_payload = json.loads(wal_path.read_text(encoding="utf-8"))
    assert first_meta.blueprint_id in pending_payload.get("pending", [])

    compiler_wal = compiler._wal_path  # type: ignore[attr-defined]
    assert compiler_wal.exists()
    compiler_record = json.loads(compiler_wal.read_text(encoding="utf-8"))
    assert compiler_record.get("blueprint_id") == first_meta.blueprint_id

    forge.run()
    assert len(library.list_all()) == 2
    assert not wal_path.exists()
    assert not compiler_wal.exists()


def test_forge_retries_failed_job(tmp_path) -> None:
    catalog = _catalog_subset(1)
    attempt_counter = {"count": 0}

    def sampler(metadata):
        attempt_counter["count"] += 1
        return attempt_counter["count"] == 1

    forge, library, compiler, wal_path = _setup(
        tmp_path,
        catalog,
        error_sampler=sampler,
        max_retries=2,
    )

    forge.run()

    assert attempt_counter["count"] >= 2
    assert len(library.list_all()) == 1
    assert not wal_path.exists()
    compiler_wal = compiler._wal_path  # type: ignore[attr-defined]
    assert not compiler_wal.exists()
    record = library.get(next(iter(catalog.all())).blueprint_id)
    assert record is not None
    assert record.compile_ms is not None
    metrics = forge.metrics_snapshot()
    assert metrics["tezzeret.compilation.failed_attempts"] >= 1.0
    assert metrics["tezzeret.jobs.retried"] >= 1.0


def test_forge_builds_telemetry_packet(tmp_path) -> None:
    catalog = _catalog_subset(1)
    forge, _, _, _ = _setup(tmp_path, catalog)
    forge.run()
    packet = forge.build_telemetry_packet(packet_id="pkt-1")
    assert packet.source_subsystem == "tezzeret"
    assert any(metric.name == "tezzeret.jobs.total" for metric in packet.metrics)
    assert any(event.description == "tezzeret.compile_succeeded" for event in packet.events)


def test_forge_enters_conservative_mode_after_breaker(tmp_path) -> None:
    catalog = _catalog_subset(1)
    attempts = {"count": 0}

    def sampler(_: BlueprintDescriptor) -> bool:
        attempts["count"] += 1
        return attempts["count"] == 1

    forge, library, compiler, wal_path = _setup(
        tmp_path,
        catalog,
        error_sampler=sampler,
        max_retries=0,
        breaker_threshold=1,
        compile_timeout_s=0.01,
    )

    with pytest.raises(RuntimeError):
        forge.run()

    assert wal_path.exists()
    metrics = forge.metrics_snapshot()
    assert metrics["tezzeret.mode.conservative"] == 1.0
    failed_events = forge.drain_telemetry_events()
    assert any(event.description == "tezzeret.compile_failed" for event in failed_events)
    assert any(event.description == "tezzeret.breaker_open" for event in failed_events)

    forge._breaker_open_until = None
    forge.run()

    record = library.get(next(iter(catalog.all())).blueprint_id)
    assert record is not None
    result = compiler.latest_result()
    assert result is not None
    assert result.compile_strategy == "conservative"
    metrics = forge.metrics_snapshot()
    assert metrics["tezzeret.mode.conservative"] == 0.0
    success_events = forge.drain_telemetry_events()
    assert any(
        event.description == "tezzeret.compile_succeeded"
        and event.attributes.get("strategy") == "conservative"
        for event in success_events
    )
    assert not wal_path.exists()
