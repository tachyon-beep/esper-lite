from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from esper.leyline import leyline_pb2
from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urza import UrzaLibrary
from esper.tamiyo import FieldReportStoreConfig, TamiyoPolicy, TamiyoService
from esper.security.signing import SignatureContext


_SIG = SignatureContext(secret=b"tamiyo-test-secret")


class _SeedPolicy(TamiyoPolicy):
    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:  # pragma: no cover - simple stub
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-bsds",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        cmd.seed_operation.blueprint_id = packet.packet_id or "bp-bsds"
        self._last_action = {"action": 0.0, "param_delta": 0.0}
        return cmd


def _descriptor(bp_id: str) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.35,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="bsds fixture",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def _service_with_urza(tmp_path: Path, bp_id: str, bsds: dict) -> TamiyoService:
    urza_root = tmp_path / "urza"
    urza_root.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / f"{bp_id}.pt"
    artifact.write_bytes(b"dummy")
    lib = UrzaLibrary(root=urza_root)
    lib.save(_descriptor(bp_id), artifact, extras={"bsds": bsds})
    return TamiyoService(
        policy=_SeedPolicy(),
        store_config=FieldReportStoreConfig(path=tmp_path / "field_reports.log"),
        urza=lib,
        signature_context=_SIG,
        step_timeout_ms=100.0,
    )


def _packet(bp_id: str) -> leyline_pb2.SystemStatePacket:
    pkt = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-bsds", packet_id=bp_id)
    seed = pkt.seed_states.add()
    seed.seed_id = "seed-bsds"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    return pkt


def test_bsds_critical_triggers_pause_and_annotations(tmp_path: Path) -> None:
    bsds = {"hazard_band": "CRITICAL", "risk_score": 0.7, "provenance": "URABRASK", "handling_class": "standard"}
    service = _service_with_urza(tmp_path, "bp-crit", bsds)
    cmd = service.evaluate_epoch(_packet("bp-crit"))
    assert cmd.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    assert any(e.description == "bsds_hazard_critical" for e in telemetry.events)
    assert cmd.annotations["bsds_hazard_band"].upper() == "CRITICAL"
    assert cmd.annotations["bsds_risk"] == "0.70"
    # blueprint_risk should reflect bsds risk override
    assert cmd.annotations["blueprint_risk"] == "0.70"


def test_bsds_high_downgrades_seed_to_optimizer(tmp_path: Path) -> None:
    bsds = {"hazard_band": "HIGH", "risk_score": 0.6}
    service = _service_with_urza(tmp_path, "bp-high", bsds)
    cmd = service.evaluate_epoch(_packet("bp-high"))
    assert cmd.command_type == leyline_pb2.COMMAND_OPTIMIZER
    telemetry = service.telemetry_packets[-1]
    assert any(e.description == "bsds_hazard_high" for e in telemetry.events)


def test_bsds_handling_quarantine_treated_critical(tmp_path: Path) -> None:
    bsds = {"hazard_band": "LOW", "handling_class": "quarantine"}
    service = _service_with_urza(tmp_path, "bp-quar", bsds)
    cmd = service.evaluate_epoch(_packet("bp-quar"))
    assert cmd.command_type == leyline_pb2.COMMAND_PAUSE
    telemetry = service.telemetry_packets[-1]
    assert any(e.description == "bsds_handling_quarantine" for e in telemetry.events)


def test_bsds_present_event_and_annotations(tmp_path: Path) -> None:
    bsds = {"hazard_band": "MEDIUM", "risk_score": 0.42, "handling_class": "restricted", "provenance": "HEURISTIC", "resource_profile": "gpu"}
    service = _service_with_urza(tmp_path, "bp-present", bsds)
    cmd = service.evaluate_epoch(_packet("bp-present"))
    telemetry = service.telemetry_packets[-1]
    assert any(e.description == "bsds_present" for e in telemetry.events)
    for key in ("bsds_hazard_band", "bsds_handling_class", "bsds_resource_profile", "bsds_provenance", "bsds_risk"):
        assert key in cmd.annotations
    assert cmd.annotations["blueprint_risk"] == "0.42"

