from __future__ import annotations

from esper.leyline import leyline_pb2
from esper.urabrask.benchmarks import BenchmarkConfig, BenchmarkProfile, run_benchmarks


def test_run_benchmarks_fallback_basic_shape() -> None:
    proto, mirror = run_benchmarks("bp-001", runtime=None)

    assert isinstance(proto, leyline_pb2.BlueprintBenchmark)
    assert proto.blueprint_id == "bp-001"
    assert proto.version == 1
    assert proto.device in {"cpu", "cuda:0"}
    assert isinstance(proto.torch_version, str)
    assert len(proto.profiles) >= 1
    first = proto.profiles[0]
    assert first.p50_latency_ms > 0.0
    assert first.p95_latency_ms >= first.p50_latency_ms
    assert first.throughput_samples_per_s > 0.0

    # JSON mirror checks (list-of-dicts per profile)
    assert isinstance(mirror, list) and len(mirror) >= 1
    entry = mirror[0]
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
    assert required.issubset(entry.keys())
    assert entry["provenance"] in {"fallback", "runtime"}
    assert entry["p50_latency_ms"] > 0.0
    assert entry["p95_latency_ms"] >= entry["p50_latency_ms"]
    assert entry["throughput_samples_per_s"] > 0.0

    # Proto round-trip
    payload = proto.SerializeToString()
    parsed = leyline_pb2.BlueprintBenchmark()
    parsed.ParseFromString(payload)
    assert parsed.blueprint_id == proto.blueprint_id
    assert len(parsed.profiles) == len(proto.profiles)


def test_run_benchmarks_custom_profiles() -> None:
    cfg = BenchmarkConfig(
        profiles=(
            BenchmarkProfile(name="small", batch_size=2, in_shape=(16,), dtype="float32"),
            BenchmarkProfile(name="large", batch_size=32, in_shape=(64,), dtype="float32"),
        )
    )
    proto, mirror = run_benchmarks("bp-xyz", runtime=None, config=cfg)
    names = [p.name for p in proto.profiles]
    assert names[:2] == ["small", "large"]  # optional bf16 profile may be appended on CUDA
    mirror_names = [m["name"] for m in mirror]
    assert "small" in mirror_names and "large" in mirror_names

    # CPU latency loose bound to catch anomalies (very generous)
    if proto.device == "cpu":
        for p in proto.profiles:
            assert p.p50_latency_ms <= 50.0
