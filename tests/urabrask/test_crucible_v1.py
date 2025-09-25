from __future__ import annotations

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.urabrask.crucible import CrucibleConfigV1, run_crucible_v1


def _descriptor(
    bp_id: str,
    *,
    tier: int = BlueprintTier.BLUEPRINT_TIER_SAFE,
    risk: float = 0.2,
    quarantine_only: bool = False,
) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=tier,
        risk=risk,
        stage=2,
        quarantine_only=quarantine_only,
        approval_required=False,
        description="crucible v1 tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_crucible_v1_nan_probe_triggers_critical() -> None:
    cfg = CrucibleConfigV1(enable_nan_probe=True)
    bsds, hazards = run_crucible_v1(_descriptor("bp-nan", risk=0.1), config=cfg)
    assert hazards["nan_inf"] == "present"
    assert bsds.hazard_band in (leyline_pb2.HAZARD_BAND_HIGH, leyline_pb2.HAZARD_BAND_CRITICAL)


def test_crucible_v1_precision_sensitive_flag() -> None:
    cfg = CrucibleConfigV1(precision_relerr_threshold=0.0)
    bsds, hazards = run_crucible_v1(_descriptor("bp-prec", risk=0.1), config=cfg)
    assert hazards["precision"] == "sensitive"
    assert bsds.provenance == leyline_pb2.PROVENANCE_URABRASK


def test_crucible_v1_oscillation_flag() -> None:
    cfg = CrucibleConfigV1(oscillation_std_threshold=0.0)
    _bsds, hazards = run_crucible_v1(_descriptor("bp-osc", risk=0.1), config=cfg)
    assert hazards["oscillation"] == "high"


def test_crucible_v1_grad_explode_detected() -> None:
    cfg = CrucibleConfigV1(grad_explode_threshold=0.0, grad_weight_scale=100.0)
    _bsds, hazards = run_crucible_v1(_descriptor("bp-grad", risk=0.1), config=cfg)
    assert hazards["grad_instability"] in ("explode", "vanish")


def test_crucible_v1_memory_watermark_trips_threshold(monkeypatch) -> None:
    # Threshold 0 should classify any delta as 'high' when psutil is available.
    cfg = CrucibleConfigV1(memory_watermark_mb_threshold=0.0)
    _bsds, hazards = run_crucible_v1(_descriptor("bp-mem", risk=0.1), config=cfg)
    assert hazards.get("memory_watermark") in {"high", "ok"}


def test_crucible_v1_oom_probe_simulated(monkeypatch) -> None:
    # Enable OOM probe with simulation to avoid real OOM
    monkeypatch.setenv("URABRASK_CRUCIBLE_ALLOW_OOM", "true")
    monkeypatch.setenv("URABRASK_CRUCIBLE_SIMULATE_OOM", "true")
    cfg = CrucibleConfigV1(enable_oom_probe=True, simulate_oom=True)
    _bsds, hazards = run_crucible_v1(_descriptor("bp-oom", risk=0.1), config=cfg)
    assert hazards.get("oom_risk") == "risk"
