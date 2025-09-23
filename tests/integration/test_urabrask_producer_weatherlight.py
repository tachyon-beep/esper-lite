from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from esper.core import EsperSettings
from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.urza import UrzaLibrary
from esper.weatherlight.service_runner import WeatherlightService


class _UrzaWorkerStub:
    class _Metrics:
        hits = 0
        misses = 0
        errors = 0
        latency_ms = 0.0

    @property
    def metrics(self):  # pragma: no cover - trivial
        return self._Metrics()


class _OonaStub:
    async def metrics_snapshot(self):  # pragma: no cover - trivial
        return {"queue.depth": 0.0}


def _urza_without_bsds(tmp_path: Path, count: int = 3) -> UrzaLibrary:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    for i in range(count):
        artifact = tmp_path / f"bp-{i}.pt"
        artifact.write_bytes(b"dummy")
        d = BlueprintDescriptor(
            blueprint_id=f"bp-{i}",
            name=f"bp-{i}",
            tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
            risk=0.4,
            stage=2,
            quarantine_only=False,
            approval_required=False,
            description="wl urabrask test",
        )
        lib.save(d, artifact, extras={})
    return lib


@pytest.mark.asyncio
async def test_weatherlight_urabrask_producer_attaches_bsds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable producer with fast interval
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    monkeypatch.setenv("URABRASK_ENABLED", "true")
    monkeypatch.setenv("URABRASK_PRODUCER_INTERVAL_S", "1")
    monkeypatch.setenv("URABRASK_TOPN_PER_CYCLE", "2")
    monkeypatch.setenv("URABRASK_ONLY_SAFE_TIER", "true")
    monkeypatch.setenv("URABRASK_TIMEOUT_MS", "0")

    urza = _urza_without_bsds(tmp_path, count=3)

    wl = WeatherlightService(settings=EsperSettings())
    wl._oona = _OonaStub()  # type: ignore[attr-defined]
    wl._urza_library = urza  # type: ignore[attr-defined]
    wl._urza_worker = _UrzaWorkerStub()  # type: ignore[attr-defined]

    # Manually construct the producer from settings and run once
    # (avoids booting full Weatherlight worker loops)
    assert wl._settings.urabrask_enabled  # type: ignore[attr-defined]
    wl._urabrask_producer = __import__("esper.urabrask.producer", fromlist=["UrabraskProducer"]).UrabraskProducer(  # type: ignore[attr-defined]
        urza,
        interval_s=1,
        topn=2,
        only_safe=True,
        timeout_ms=0,
    )
    stats = wl._urabrask_producer.run_once()  # type: ignore[attr-defined]
    assert stats["processed"] >= 1
    # Verify extras were attached for at least one record
    attached = 0
    for rec in urza.list_all().values():
        if "bsds" in (rec.extras or {}):
            attached += 1
    assert attached >= 1

    # Weatherlight telemetry should expose producer metrics
    pkt = await wl._build_telemetry_packet()
    names = {m.name for m in pkt.metrics}
    assert "urabrask.produced_total" in names
    assert "urabrask.last_processed" in names

