from __future__ import annotations

from esper.leyline import leyline_pb2


def test_bsds_enums_present() -> None:
    # Ensure enums exist and support Name/Value mapping
    assert leyline_pb2.HazardBand.Value("HAZARD_BAND_HIGH") == leyline_pb2.HAZARD_BAND_HIGH
    assert leyline_pb2.HandlingClass.Value("HANDLING_CLASS_RESTRICTED") == leyline_pb2.HANDLING_CLASS_RESTRICTED
    assert leyline_pb2.ResourceProfile.Value("RESOURCE_PROFILE_GPU") == leyline_pb2.RESOURCE_PROFILE_GPU
    assert leyline_pb2.Provenance.Value("PROVENANCE_URABRASK") == leyline_pb2.PROVENANCE_URABRASK


def test_bsds_message_round_trip() -> None:
    bsds = leyline_pb2.BSDS(
        version=1,
        blueprint_id="bp-xyz",
        risk_score=0.82,
        hazard_band=leyline_pb2.HAZARD_BAND_HIGH,
        handling_class=leyline_pb2.HANDLING_CLASS_RESTRICTED,
        resource_profile=leyline_pb2.RESOURCE_PROFILE_GPU,
        recommendation="Prefer optimizer downgrade; avoid aggressive grafting",
        provenance=leyline_pb2.PROVENANCE_URABRASK,
    )
    payload = bsds.SerializeToString()
    out = leyline_pb2.BSDS()
    out.ParseFromString(payload)
    assert out.blueprint_id == "bp-xyz"
    assert out.hazard_band == leyline_pb2.HAZARD_BAND_HIGH
    assert out.provenance == leyline_pb2.PROVENANCE_URABRASK


def test_benchmark_messages_present() -> None:
    bench = leyline_pb2.BlueprintBenchmark(
        version=1,
        blueprint_id="bp-xyz",
        device="cuda:0",
        torch_version="2.8.0",
    )
    bench.profiles.add(name="batch16_f32", p50_latency_ms=12.3, p95_latency_ms=20.5, throughput_samples_per_s=1500.0)
    data = bench.SerializeToString()
    clone = leyline_pb2.BlueprintBenchmark()
    clone.ParseFromString(data)
    assert clone.blueprint_id == "bp-xyz"
    assert clone.profiles[0].name == "batch16_f32"

