from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.tamiyo import FieldReportStoreConfig, TamiyoPolicy, TamiyoService
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


def _urza_with_bsds(tmp_path: Path, bp_id: str) -> UrzaLibrary:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / f"{bp_id}.pt"
    artifact.write_bytes(b"dummy")
    descriptor = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=bp_id,
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.35,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="bsds-indicator-test",
    )
    lib.save(
        descriptor,
        artifact,
        extras={
            "bsds": {
                "risk_score": 0.9,
                "hazard_band": "CRITICAL",
                "handling_class": "quarantine",
                "resource_profile": "gpu",
                "provenance": "URABRASK",
            }
        },
    )
    return lib


@pytest.mark.asyncio
async def test_weatherlight_surfaces_bsds_indicators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")

    urza = _urza_with_bsds(tmp_path, "bp-wl")
    service = TamiyoService(
        policy=TamiyoPolicy(),
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=urza,
        step_timeout_ms=500.0,
        metadata_timeout_ms=500.0,
    )
    # Evaluate once to generate Tamiyo telemetry containing BSDS events
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=1,
        training_run_id="run-wl",
        packet_id="bp-wl",
    )
    seed = packet.seed_states.add()
    seed.seed_id = "seed-wl"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed.learning_rate = 1e-3
    seed.risk_score = 0.1
    _ = service.evaluate_epoch(packet)

    wl = WeatherlightService()
    wl._tamiyo_service = service  # type: ignore[attr-defined]
    wl._oona = _OonaStub()  # type: ignore[attr-defined]
    wl._urza_worker = _UrzaWorkerStub()  # type: ignore[attr-defined]

    pkt = await wl._build_telemetry_packet()
    indicators = pkt.system_health.indicators
    assert indicators.get("bsds_provenance") == "urabrask"
    assert indicators.get("bsds_hazard") == "CRITICAL"
