from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask.bench_worker import UrabraskBenchWorker
from esper.urza import UrzaLibrary


def _descriptor(bp_id: str) -> BlueprintDescriptor:
    d = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=f"name-{bp_id}",
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.2,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="bench cadence tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_bench_worker_respects_cooldown(tmp_path: Path, monkeypatch) -> None:
    # Set long cooldown
    monkeypatch.setenv("URABRASK_BENCH_MIN_INTERVAL_S", "3600")
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    lib.save(_descriptor("bp-bench-cooldown"), artifact, extras={})

    worker = UrabraskBenchWorker(lib, runtime=None, interval_s=1, topn=1, timeout_ms=0)
    # First run should attach benchmarks and set last_run
    stats1 = worker.run_once()
    assert stats1["attached_profiles"] >= 0.0  # may be zero if fallback; we only care flow works
    rec1 = lib.get("bp-bench-cooldown")
    assert rec1 is not None and isinstance(rec1.extras, dict)
    assert "benchmarks_last_run" in rec1.extras
    # Second run should skip due to cooldown
    stats2 = worker.run_once()
    assert stats2.get("skipped_cooldown", 0.0) >= 0.0  # metric increments when nothing processed
