from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.urabrask.service import produce_benchmarks
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
        description="benchmarks service tests",
    )
    d.allowed_parameters["alpha"].min_value = 0.1
    d.allowed_parameters["alpha"].max_value = 0.9
    return d


def test_produce_benchmarks_attaches_to_urza(tmp_path: Path) -> None:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"dummy")
    d = _descriptor("bp-bench")
    lib.save(d, artifact, extras={})

    bench = produce_benchmarks(lib, runtime=None, blueprint_id="bp-bench")
    assert bench.blueprint_id == "bp-bench"
    assert bench.version == 1
    rec = lib.get("bp-bench")
    assert rec is not None and "benchmarks" in (rec.extras or {})
    mirror = rec.extras["benchmarks"]
    assert isinstance(mirror, list) and len(mirror) >= 1
    # Ensure profile entries have expected keys (strict schema)
    required = {
        "name",
        "batch_size",
        "in_shape",
        "dtype",
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_samples_per_s",
        "provenance",
    }
    assert required.issubset(set(mirror[0].keys()))
