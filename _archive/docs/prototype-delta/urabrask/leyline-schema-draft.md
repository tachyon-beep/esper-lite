# Leyline Additions — BSDS & Benchmarks (Day 1 Draft)

This draft specifies the minimal additions to `contracts/leyline/leyline.proto` to support Urabrask from day 1. It aligns with BSDS‑Lite and provides canonical enums/messages for producers/consumers.

```
// Enumerations
enum HazardBand {
  HAZARD_BAND_UNSPECIFIED = 0;
  HAZARD_BAND_LOW = 1;
  HAZARD_BAND_MEDIUM = 2;
  HAZARD_BAND_HIGH = 3;
  HAZARD_BAND_CRITICAL = 4;
}

enum HandlingClass {
  HANDLING_CLASS_UNSPECIFIED = 0;
  HANDLING_CLASS_STANDARD = 1;
  HANDLING_CLASS_RESTRICTED = 2;
  HANDLING_CLASS_QUARANTINE = 3;
}

enum ResourceProfile {
  RESOURCE_PROFILE_UNSPECIFIED = 0;
  RESOURCE_PROFILE_CPU = 1;
  RESOURCE_PROFILE_GPU = 2;
  RESOURCE_PROFILE_MEMORY_HEAVY = 3;
  RESOURCE_PROFILE_IO_HEAVY = 4;
  RESOURCE_PROFILE_MIXED = 5;
}

enum Provenance {
  PROVENANCE_UNSPECIFIED = 0;
  PROVENANCE_URABRASK = 1;
  PROVENANCE_CURATED = 2;
  PROVENANCE_HEURISTIC = 3;
  PROVENANCE_EXTERNAL = 4;
}

// Messages
message BSDS {
  uint32 version = 1;
  string blueprint_id = 2;
  float risk_score = 3; // 0..1
  HazardBand hazard_band = 4;
  HandlingClass handling_class = 5;
  ResourceProfile resource_profile = 6;
  string recommendation = 7; // optional free‑text mitigation
  google.protobuf.Timestamp issued_at = 8;
  Provenance provenance = 9;
}

message BlueprintBenchmarkProfile {
  string name = 1;         // e.g. batch16_f32
  float p50_latency_ms = 2;
  float p95_latency_ms = 3;
  float throughput_samples_per_s = 4;
}

message BlueprintBenchmark {
  uint32 version = 1;
  string blueprint_id = 2;
  repeated BlueprintBenchmarkProfile profiles = 3;
  string device = 4; // cuda:0, cpu
  string torch_version = 5;
}

message BSDSIssued {
  BSDS bsds = 1;
}

message BSDSFailed {
  string blueprint_id = 1;
  string reason = 2;
}

message BenchmarkReport {
  BlueprintBenchmark benchmark = 1;
}
```

Migration Notes
- Add these under new number ranges to avoid field collisions.
- Generate Python bindings and ensure import paths unchanged.
- Consumers may continue to use `Urza.record.extras["bsds"]` as a transitional transport while wiring direct protobuf flows.

