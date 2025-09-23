from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urza import UrzaLibrary
from esper.tamiyo import FieldReportStoreConfig, TamiyoPolicy, TamiyoService
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext
from esper.urabrask import metrics as ura_metrics
from esper.urabrask.service import produce_and_attach_bsds


class _Policy(TamiyoPolicy):
    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:  # pragma: no cover - stub
        cmd = leyline_pb2.AdaptationCommand(version=1, command_type=leyline_pb2.COMMAND_SEED, target_seed_id="seed-int")
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        cmd.seed_operation.blueprint_id = packet.packet_id or "bp-int"
        return cmd


def _descriptor(bp_id: str, *, risk: float = 0.4) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=risk,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="integrity fail tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def _packet(bp_id: str) -> leyline_pb2.SystemStatePacket:
    return leyline_pb2.SystemStatePacket(version=1, packet_id=bp_id, current_epoch=1)


def test_tamiyo_verification_increments_integrity_failures(tmp_path: Path, monkeypatch) -> None:
    # Enable signing so verification path is active
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    monkeypatch.setenv("URABRASK_SIGNING_ENABLED", "true")

    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")

    # Initial valid BSDS (signed)
    lib.save(_descriptor("bp-int"), artifact, extras={})
    _ = produce_and_attach_bsds(lib, "bp-int")

    # Tamper bsds without updating signature
    rec = lib.get("bp-int")
    assert rec is not None
    extras = dict(rec.extras or {})
    bsds = dict(extras.get("bsds") or {})
    bsds["hazard_band"] = "LOW" if str(bsds.get("hazard_band", "")).upper() != "LOW" else "HIGH"
    extras["bsds"] = bsds
    # keep bsds_sig unchanged
    lib.save(rec.metadata, rec.artifact_path, extras=extras)

    before = ura_metrics.snapshot().get("integrity_failures", 0.0)

    service = TamiyoService(policy=_Policy(), store_config=FieldReportStoreConfig(path=tmp_path / "fr.log"), urza=lib, signature_context=SignatureContext(secret=b"t"))
    _ = service.evaluate_epoch(_packet("bp-int"))

    after = ura_metrics.snapshot().get("integrity_failures", 0.0)
    assert after >= before + 1.0

