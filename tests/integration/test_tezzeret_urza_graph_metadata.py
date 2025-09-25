from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from esper.karn import BlueprintDescriptor, BlueprintParameterBounds, BlueprintTier, KarnCatalog
from esper.leyline import leyline_pb2
from esper.tamiyo.service import TamiyoService
from esper.tezzeret.compiler import CompileJobConfig, TezzeretCompiler
from esper.tezzeret.runner import TezzeretForge
from esper.urza import UrzaLibrary


def _make_descriptor(blueprint_id: str = "BP_META") -> BlueprintDescriptor:
    desc = BlueprintDescriptor(
        blueprint_id=blueprint_id,
        name="linear_meta",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
    )
    # Add one allowed parameter for parameter metadata
    bounds = BlueprintParameterBounds(min_value=0.0, max_value=1.0)
    desc.allowed_parameters["alpha"].CopyFrom(bounds)
    return desc


@pytest.mark.asyncio
async def test_tezzeret_saves_graph_metadata_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid heavy torch.compile during test (force eager fallback path)
    def _raise_compile(*_args, **_kwargs):  # pragma: no cover - deliberately triggered
        raise RuntimeError("compile disabled for test")

    monkeypatch.setattr(torch, "compile", _raise_compile)

    catalog = KarnCatalog(load_defaults=False)
    desc = _make_descriptor("BP_META1")
    catalog.register(desc)

    urza_root = tmp_path / "urza"
    urza = UrzaLibrary(root=urza_root)
    artifacts = tmp_path / "artifacts"
    compiler = TezzeretCompiler(
        CompileJobConfig(artifact_dir=artifacts, use_cuda=False, max_retries=0)
    )
    forge = TezzeretForge(
        catalog, urza, compiler, wal_path=tmp_path / "tezzeret_wal.json", compile_timeout_s=5.0
    )

    forge.run()

    record = urza.get(desc.blueprint_id)
    assert record is not None
    assert "graph_metadata" in (record.extras or {})
    gm = record.extras.get("graph_metadata", {})  # type: ignore[assignment]
    assert isinstance(gm, dict)
    assert isinstance(gm.get("layers"), list) and len(gm["layers"]) >= 1
    assert isinstance(gm.get("activations"), list) and len(gm["activations"]) >= 1
    assert isinstance(gm.get("parameters"), list) and any(p.get("name") == "alpha" for p in gm["parameters"])  # type: ignore[index]
    adj = gm.get("adjacency", {})
    assert isinstance(adj, dict) and "layer" in adj


def test_tamiyo_consumes_urza_graph_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")

    # Avoid heavy torch.compile during test (force eager fallback path)
    def _raise_compile(*_args, **_kwargs):  # pragma: no cover - deliberately triggered
        raise RuntimeError("compile disabled for test")

    monkeypatch.setattr(torch, "compile", _raise_compile)

    catalog = KarnCatalog(load_defaults=False)
    desc = _make_descriptor("BP_META2")
    catalog.register(desc)
    urza_root = tmp_path / "urza2"
    urza = UrzaLibrary(root=urza_root)
    artifacts = tmp_path / "artifacts2"
    compiler = TezzeretCompiler(
        CompileJobConfig(artifact_dir=artifacts, use_cuda=False, max_retries=0)
    )
    forge = TezzeretForge(
        catalog, urza, compiler, wal_path=tmp_path / "tezzeret_wal.json", compile_timeout_s=5.0
    )
    forge.run()

    # Build a minimal packet that references the compiled blueprint via packet_id
    svc = TamiyoService(urza=urza, step_timeout_ms=100.0)
    pkt = leyline_pb2.SystemStatePacket(
        version=1, current_epoch=1, training_run_id="run", packet_id=desc.blueprint_id
    )
    seed = pkt.seed_states.add()
    seed.seed_id = "seed-x"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed.learning_rate = 0.01
    cmd = svc.evaluate_step(pkt)

    # Assert per-type coverage includes layer connectivity from extras
    telemetry = svc.telemetry_packets[-1]
    names = {m.name for m in telemetry.metrics}
    assert any(
        name.startswith("tamiyo.gnn.feature_coverage.edges.layer_connects") for name in names
    )
    # Also ensure annotations include typed coverage json
    assert "coverage_types" in cmd.annotations
